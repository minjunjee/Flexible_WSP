"""Batched sampling with per-sample weight edits on mlp.down_proj.

Every edit used in these experiments is a change to a down_proj matrix W (neuron j's write vector is column j):
  zero    : set columns idx to 0                      -> same as zeroing h[idx] before down_proj
  delta   : add D to columns cols                     -> out += h[cols] @ D^T
  rank1   : W += u v^T                                -> out += (h . v) u
  push    : constant vector added to the MLP output   (steering vector; not a pure weight edit)
Applying them through hooks is mathematically the same as editing W, and lets one batch hold several edited models.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/claude/poc")
import qw

EOS_ID = 248044
END_IDS = {248044, 248046}


class Editor:
    """Holds several edited models in one batch.

    A condition is a list of ops on down_proj of a layer:
      ("zero",  layer, idx)        zero columns idx
      ("delta", layer, cols, D)    add D [1024, len(cols)] to columns cols
      ("rank1", layer, u, v)       add outer(u, v)
      ("push",  layer, vec)        add a constant vector to the MLP output (steering vector, not a weight edit)
    Weight ops are materialized as an edited bfloat16 matrix W' = bf16(W + dW), exactly what a saved checkpoint
    would contain, and the rows of the batch that use it are recomputed with W'.
    """

    def __init__(self, model):
        self.lm = qw.lm_of(model)
        self.wops, self.pops, self.sched, self.ctx = {}, {}, {}, None
        self.handles = [layer.mlp.down_proj.register_forward_hook(self._post(l)) for l, layer in enumerate(self.lm.layers)]

    def clear(self):
        self.wops, self.pops, self.sched, self.ctx = {}, {}, {}, None

    def remove(self):
        for h in self.handles:
            h.remove()

    def edited_weight(self, layer_ops, l):
        W = self.lm.layers[l].mlp.down_proj.weight.detach().float().clone()
        for op in layer_ops:
            if op[0] == "zero":
                W[:, op[2]] = 0
            elif op[0] == "delta":
                W[:, op[2]] += op[3]
            elif op[0] == "rank1":
                W += torch.outer(op[2], op[3])
        return W.to(torch.bfloat16)

    def set(self, B, conditions):
        """conditions: list of length B; each a list of ops. Identical condition objects share one edited matrix."""
        self.clear()
        cache = {}
        for b, ops in enumerate(conditions):
            by_layer = {}
            for op in ops:
                by_layer.setdefault(op[1], []).append(op)
            for l, lops in by_layer.items():
                w = [o for o in lops if o[0] not in ("push", "sched")]
                if w:
                    key = (id(ops), l)
                    if key not in cache:
                        cache[key] = [self.edited_weight(w, l), []]
                        self.wops.setdefault(l, []).append(cache[key])
                    cache[key][1].append(b)
                for o in lops:
                    if o[0] == "push":
                        self.pops.setdefault(l, []).append((b, o[2].to(torch.bfloat16)))
                    if o[0] == "sched":
                        # ("sched", layer, {j: vec}): add vec only at answer position j
                        # (j = 0 is the last prompt token, j >= 1 is where the j-th generated token is the input)
                        self.sched.setdefault(l, []).append((b, {j: v.to(torch.bfloat16) for j, v in o[2].items()}))

    def _post(self, l):
        def hook(mod, inp, out):
            w, p = self.wops.get(l), self.pops.get(l)
            sc = self.sched.get(l) if self.ctx is not None else None
            if not w and not p and not sc:
                return None
            h = inp[0]
            out = out.clone()
            for We, rows in (w or []):
                out[rows] = torch.nn.functional.linear(h[rows], We)
            for b, vec in (p or []):
                out[b] += vec[None, :]
            for b, table in (sc or []):
                if self.ctx[0] == "prompt_last":
                    if 0 in table:
                        out[b, -1] += table[0]
                elif self.ctx[1] in table:
                    out[b, 0] += table[self.ctx[1]]
            return out
        return hook


class DupTracker:
    """Online version of the 16-token duplicate detector."""

    def __init__(self, history=(), k=qw.K_DUP):
        self.k, self.ids, self.seen, self.dup = k, [], set(), []
        for t in history:
            self.push(int(t))

    def push(self, t):
        self.ids.append(t)
        d = False
        if len(self.ids) >= self.k:
            w = tuple(self.ids[-self.k:])
            d = w in self.seen
            self.seen.add(w)
        self.dup.append(d)
        return d


def short_onset(dup, win=128, frac=0.8):
    """Loop onset for short generations: first duplicate token t with >= frac duplicates in dup[t:t+win]."""
    dup = np.asarray(dup, bool)
    n = len(dup)
    if n < win:
        return None
    c = np.concatenate([[0], np.cumsum(dup)])
    dens = (c[win:] - c[:-win]) / win
    ok = np.flatnonzero((dens >= frac) & dup[: n - win + 1])
    return int(ok[0]) if len(ok) else None


@torch.inference_mode()
def generate(model, prompt_ids, B, seeds, max_new, history=None, temperature=0.6, top_p=0.95, top_k=20,
             presence=1.5, stop_loop=True, loop_patience=96, log_every=0, chunk=128, editor=None):
    """Sample B continuations of the same prompt. `history` = response tokens already in the prompt
    (they count for the presence penalty and for duplicate detection). Returns list of dicts."""
    ids = torch.tensor([prompt_ids] * B)
    lm = qw.lm_of(model)

    def fwd(x, pkv=None):
        # body only, then the output head on the last position: logits for every prompt position would need gigabytes
        o = lm(input_ids=x, past_key_values=pkv, use_cache=True)
        return model.lm_head(o.last_hidden_state[:, -1:])[:, 0].float(), o.past_key_values

    pkv = None
    for s0 in range(0, ids.shape[1], chunk):
        if editor is not None:
            editor.ctx = ("prompt_last",) if s0 + chunk >= ids.shape[1] else None
        logits, pkv = fwd(ids[:, s0:s0 + chunk], pkv)
    V = logits.shape[-1]
    seen = torch.zeros(B, V, dtype=torch.bool)
    hist = list(history) if history is not None else []
    if hist:
        seen[:, torch.tensor(sorted(set(hist)))] = True
    gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    trackers = [DupTracker(hist) for _ in range(B)]
    toks = [[] for _ in range(B)]
    done = [None] * B
    looped_at = [None] * B
    for step in range(max_new):
        greedy = temperature == 0
        lg = (logits - presence * seen.float()) / (1.0 if greedy else temperature)
        topv, topi = lg.topk(top_k, dim=-1)
        p = torch.softmax(topv, -1)
        keep = (p.cumsum(-1) - p) < top_p
        p = p * keep
        p = p / p.sum(-1, keepdim=True)
        nxt = torch.empty(B, dtype=torch.long)
        for b in range(B):
            j = 0 if greedy else torch.multinomial(p[b], 1, generator=gens[b]).item()
            nxt[b] = topi[b, j]
            if done[b] is None:
                t = int(nxt[b])
                toks[b].append(t)
                seen[b, t] = True
                trackers[b].push(t)
                if t in END_IDS:
                    done[b] = "eos"
                elif stop_loop:
                    d = trackers[b].dup[len(hist):]
                    o = short_onset(d) if len(d) >= 128 and len(d) % 16 == 0 else None
                    if o is not None and len(d) - o >= 128 + loop_patience:
                        done[b] = "loop"
        if all(d is not None for d in done):
            break
        if log_every and step % log_every == 0:
            print(f"   step {step} done={sum(d is not None for d in done)}/{B}", flush=True)
        if editor is not None:
            editor.ctx = ("gen", step + 1)
        logits, pkv = fwd(nxt[:, None], pkv)
    if editor is not None:
        editor.ctx = None
    res = []
    for b in range(B):
        d = np.array(trackers[b].dup[len(hist):], bool)
        res.append(dict(tokens=toks[b], end=done[b] or "limit", dup=d, onset=short_onset(d)))
    return res
