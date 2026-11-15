"""Model wrapper shared by the experiments.

What it gives:
  - chat prompts as token ids, with or without an instruction appended to the question
  - the residual stream after every block and the inputs of every MLP output matrix (down_proj), for one sequence
  - additions to the MLP output at chosen positions (used to build edits and as inference-time baselines)
  - rank-1 changes to down_proj.weight, stored in the weight's own dtype, and a way to restore the original weights
  - greedy generation without padding (prompts of equal length are batched together)

"Answer-start position" means the last prompt token after the chat template is applied: the position whose output
decides the first token of the answer.
"""
import contextlib
import json
import os

import torch
from transformers import AutoTokenizer

DATA_ROOT = os.environ.get("WSP_DATA_ROOT", "/data/minjun_dev/Introduction_LLM/Flexible_WSP")
MODELS = {
    "qwen35-0.8b": dict(path=f"{DATA_ROOT}/models/qwen35_08b", loader="image_text", template=dict(enable_thinking=False)),
    "qwen35-2b": dict(path="Qwen/Qwen3.5-2B", loader="image_text", template=dict(enable_thinking=False)),
    "qwen35-4b": dict(path="Qwen/Qwen3.5-4B", loader="image_text", template=dict(enable_thinking=False)),
    "phi35-mini": dict(path="microsoft/Phi-3.5-mini-instruct", loader="causal", template={}),
    # the revision IWIBench fixes for its target model
    "gemma3-4b": dict(path="google/gemma-3-4b-it", loader="image_text", template={}, revision="093f9f388b31de276ce2de164bdc2081324b9767"),
    "gemma2-9b": dict(path="google/gemma-2-9b-it", loader="causal", template={}),
}
END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<|end|>", "<eos>")


class LM:
    def __init__(self, name, device="cuda", dtype=torch.float32):
        """The released weights are bfloat16. They are loaded as float32 (the same values), so the forward pass runs in
        float32: with bfloat16 arithmetic the greedy output of this model changed with the batch size in about one
        answer out of ten. Edited weights are still rounded to bfloat16 values, as a saved checkpoint would hold them."""
        spec = MODELS[name]
        self.name, self.device, self.template = name, device, dict(spec["template"])
        rev = dict(revision=spec["revision"]) if "revision" in spec else {}
        self.tok = AutoTokenizer.from_pretrained(spec["path"], **rev)
        if spec["loader"] == "image_text":
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(spec["path"], dtype=dtype, **rev).eval()
            if hasattr(model.model, "visual"):
                model.model.visual = None
            self.body = model.model.language_model
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(spec["path"], dtype=dtype, **rev).eval()
            self.body = model.model
        self.model = model.to(device)
        self.head = model.lm_head
        self.layers = list(self.body.layers)
        self.L = len(self.layers)
        self.mlp_out = [layer.mlp.down_proj for layer in self.layers]
        # Gemma puts an RMSNorm between the MLP output and the residual stream; Qwen and Phi add the MLP output as it is
        self.post_norm = [getattr(layer, "post_feedforward_layernorm", None) for layer in self.layers]
        self.has_post_norm = self.post_norm[0] is not None
        self.d_model, self.d_mlp = self.mlp_out[0].weight.shape
        self.W0 = [m.weight.detach().clone() for m in self.mlp_out]
        vocab = self.tok.get_vocab()
        self.end_ids = {vocab[t] for t in END_TOKENS if t in vocab} | {self.tok.eos_token_id}
        self.cap_h, self.cap_res, self.adds, self.phase, self.cap_last = None, None, [], None, False
        for l in range(self.L):
            self.mlp_out[l].register_forward_hook(self._mlp_hook(l))
            self.layers[l].register_forward_hook(self._block_hook(l))

    # ------------------------------------------------------------ hooks
    def _mlp_hook(self, l):
        def hook(mod, inp, out):
            if self.cap_h is not None:
                self.cap_h[l] = inp[0][:, -1:].detach() if self.cap_last else inp[0].detach()
            todo = [a for a in self.adds if a[0] == l]
            if not todo:
                return None
            out = out.clone()
            for _, where, vec in todo:
                vec = vec.to(out.dtype)
                if where == "all":                               # every position, every forward pass
                    out += vec if vec.dim() == 1 else vec[:, None, :]
                elif where == "start":                           # last prompt token only (first pass of generation)
                    if self.phase == "prompt":
                        out[:, -1] += vec
                else:                                            # an absolute position in a single full-sequence pass
                    out[:, where] += vec
            return out
        return hook

    def _block_hook(self, l):
        def hook(mod, inp, out):
            if self.cap_res is not None:
                x = out[0] if isinstance(out, tuple) else out
                self.cap_res[l] = x[:, -1:].detach() if self.cap_last else x.detach()
        return hook

    # ------------------------------------------------------------ prompts
    def prompt_ids(self, question, instruction=None, **template):
        text = question if instruction is None else question + " " + instruction
        kw = dict(self.template)
        kw.update(template)
        s = self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, **kw)
        return self.tok(s, add_special_tokens=False).input_ids

    def decode(self, tokens):
        return self.tok.decode([t for t in tokens if t not in self.end_ids]).strip()

    # ------------------------------------------------------------ one full sequence
    @torch.inference_mode()
    def run(self, ids, adds=(), logits=False, last_only=False):
        """One sequence, no cache. Returns (res, hh[, logits]):
        res [L, T, d_model] residual after each block (the last one is the input of the final norm),
        hh  [L, T, d_mlp]   inputs of each down_proj, both float32.
        adds: (layer, position, vector) added to that layer's MLP output at that position.
        last_only: keep only the last position (T = 1); for long inputs."""
        self.cap_h, self.cap_res, self.adds, self.phase, self.cap_last = {}, {}, list(adds), None, last_only
        try:
            out = self.body(input_ids=torch.tensor([ids], device=self.device))
            res = torch.stack([self.cap_res[l][0].float() for l in range(self.L)])
            hh = torch.stack([self.cap_h[l][0].float() for l in range(self.L)])
            if logits:
                return res, hh, self.head(out.last_hidden_state[0]).float()
            return res, hh
        finally:
            self.cap_h, self.cap_res, self.adds, self.cap_last = None, None, [], False

    @torch.inference_mode()
    def mlp_stats(self, ids, position):
        """For models with a norm after the MLP: at one position of one sequence, the output of that norm for every
        layer [L, d_model] (what the MLP adds to the residual stream) and the norm of the MLP output before it [L]."""
        post, pre, hooks = {}, {}, []
        for l in range(self.L):
            hooks.append(self.post_norm[l].register_forward_hook(lambda m, i, o, l=l: post.__setitem__(l, o[0, position].detach().float())))
            hooks.append(self.mlp_out[l].register_forward_hook(lambda m, i, o, l=l: pre.__setitem__(l, o[0, position].detach().float().norm())))
        try:
            self.body(input_ids=torch.tensor([ids], device=self.device))
        finally:
            for h in hooks:
                h.remove()
        return torch.stack([post[l] for l in range(self.L)]), torch.stack([pre[l] for l in range(self.L)])

    def to_pre_norm(self, l, u, ybar, mnorm, strength=4.0):
        """The vector to add to the MLP output (before the norm) so that what reaches the residual stream at layer l
        is close to ybar + u: ybar is the mean of what the MLP adds there now, u the wanted shift. The norm removes
        the scale of its input, so only a direction can be set: the direction whose normed image is ybar + u. It is
        added with `strength` times the usual length of the MLP output, enough to dominate it where the edit fires."""
        g = 1.0 + self.post_norm[l].weight.detach().float()
        g = torch.where(g.abs() < 0.05, 0.05 * torch.where(g < 0, -torch.ones_like(g), torch.ones_like(g)), g)
        z = (ybar.to(g.device) + u.to(g.device)) / g
        return strength * mnorm.to(g.device) * z / z.norm()

    @torch.inference_mode()
    def logprobs(self, ids, adds=()):
        """log-probabilities of the next token at every position of one sequence [T, V]"""
        self.adds, self.phase = list(adds), None
        try:
            out = self.body(input_ids=torch.tensor([ids], device=self.device))
            return torch.log_softmax(self.head(out.last_hidden_state[0]).float(), -1)
        finally:
            self.adds = []

    # ------------------------------------------------------------ generation
    @torch.inference_mode()
    def generate(self, prompts, max_new=40, adds=(), batch=64):
        """Greedy. prompts: list of token-id lists. Returns a list of generated token lists (end token included).
        adds: (layer, "all" | "start", vector). Prompts are batched only with prompts of the same length, so no padding."""
        out = [None] * len(prompts)
        by_len = {}
        for i, p in enumerate(prompts):
            by_len.setdefault(len(p), []).append(i)
        self.adds = list(adds)
        try:
            for idx_all in by_len.values():
                for s in range(0, len(idx_all), batch):
                    idx = idx_all[s:s + batch]
                    ids = torch.tensor([prompts[i] for i in idx], device=self.device)
                    self.phase = "prompt"
                    o = self.body(input_ids=ids, use_cache=True)
                    self.phase = "gen"
                    toks = [[] for _ in idx]
                    done = [False] * len(idx)
                    for _ in range(max_new):
                        nxt = self.head(o.last_hidden_state[:, -1]).float().argmax(-1)
                        for b, t in enumerate(nxt.tolist()):
                            if not done[b]:
                                toks[b].append(t)
                                done[b] = t in self.end_ids
                        if all(done):
                            break
                        o = self.body(input_ids=nxt[:, None], past_key_values=o.past_key_values, use_cache=True)
                    for b, i in enumerate(idx):
                        out[i] = toks[b]
        finally:
            self.adds, self.phase = [], None
        return out

    @torch.inference_mode()
    def generate_rows(self, prompt, row_adds, max_new=40):
        """One prompt, one row per condition. row_adds[b] = [(layer, "all" | "start", vector)] for row b."""
        B, table = len(row_adds), {}
        for b, lst in enumerate(row_adds):
            for l, where, vec in lst:
                M = table.setdefault((l, where), torch.zeros(B, self.d_model, device=self.device))
                M[b] += vec.to(self.device).float()
        return self.generate([prompt] * B, max_new, adds=[(l, w, M) for (l, w), M in table.items()], batch=B)

    # ------------------------------------------------------------ weight edits
    def add_rank1(self, l, u, v, scale=1.0):
        """down_proj.weight of layer l += scale * outer(u, v), rounded to bfloat16 values (what a checkpoint would hold)"""
        W = self.mlp_out[l].weight
        new = W.data.float() + scale * torch.outer(u.float().to(W.device), v.float().to(W.device))
        W.data.copy_(new.to(torch.bfloat16).to(W.dtype))

    def apply(self, edits, scale=1.0):
        """edits: list of dicts with keys layer, u, v"""
        for e in edits:
            self.add_rank1(e["layer"], torch.as_tensor(e["u"]), torch.as_tensor(e["v"]), scale)

    def restore(self):
        for m, w0 in zip(self.mlp_out, self.W0):
            m.weight.data.copy_(w0)

    @contextlib.contextmanager
    def original(self):
        """inside this block the model has its original weights; the current edits come back afterwards"""
        saved = [m.weight.data.clone() for m in self.mlp_out]
        self.restore()
        try:
            yield
        finally:
            for m, w in zip(self.mlp_out, saved):
                m.weight.data.copy_(w)

    def changed_weights(self):
        return int(sum((m.weight.data != w0).sum() for m, w0 in zip(self.mlp_out, self.W0)))


def load_questions(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "questions_v1.json")
    return json.load(open(path))["questions"]
