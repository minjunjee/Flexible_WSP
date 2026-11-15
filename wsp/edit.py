"""Conditional edits: rank-1 changes to MLP output matrices that add a vector only at chosen positions.

Follows PLAN.md, "조건부 편집 하나를 만드는 법". For a layer l of the layer set S,

    down_proj.weight += outer(u[l], v[l])

u[l] is the part of the instruction's hidden-state difference that layer l has to add; v[l] reads the inputs of
down_proj and is fitted so that v[l] . h is 1 at the positions where the edit must fire and 0 at every other position.
"""
import numpy as np
import torch


def layer_sets(L):
    """named layer sets used in the experiments (0-based block indices)"""
    every4 = sorted(set(range(3, L, 4)) | {L - 1})
    return {
        "all layers": list(range(L)),
        "every 4th layer": every4,
        "later half": list(range(L // 2, L)),
        "last quarter": list(range(L - L // 4, L)),
        "last 2 layers": [L - 2, L - 1],
        "last layer": [L - 1],
        "earlier half": list(range(L // 2)),
        "every 8th layer": sorted(set(range(7, L, 8)) | {L - 1}),
    }


def start_effect(lm, questions, instruction):
    """[Q, L, d_model]: residual after each block at the answer-start position, with the instruction minus without"""
    out = []
    for q in questions:
        r0, _ = lm.run(lm.prompt_ids(q))
        r1, _ = lm.run(lm.prompt_ids(q, instruction))
        out.append(r1[:, -1] - r0[:, -1])
    return torch.stack(out)


def shared_fraction(diffs):
    """diffs [Q, d]: squared norm of the mean difference divided by the mean squared norm of the differences"""
    return float(diffs.mean(0).pow(2).sum() / diffs.pow(2).sum(1).mean())


def writes(d, S):
    """d [L, d_model] mean difference after each block. u[l] = d[l] - d[previous layer of S] (d[l] for the first)."""
    u, prev = {}, None
    for l in S:
        u[l] = d[l] - (d[prev] if prev is not None else 0)
        prev = l
    return u


def fit_read_vector(H, y, lam):
    """ridge regression of y on H; lam is relative to the mean squared norm of the rows of H"""
    H, y = H.double(), y.double()
    reg = lam * H.pow(2).sum(1).mean()
    if H.shape[0] <= H.shape[1]:
        G = H @ H.T
        return (H.T @ torch.linalg.solve(G + reg * torch.eye(len(G), dtype=G.dtype, device=G.device), y)).float()
    C = H.T @ H
    return torch.linalg.solve(C + reg * torch.eye(len(C), dtype=C.dtype, device=C.device), H.T @ y).float()


def collect(lm, seqs, layers):
    """seqs: dicts with q (question index), ids, adds [(layer, position, vector)], targets (positions that get label 1).
    Returns H {layer: [N, d_mlp]}, y [N], q [N] over every position of every sequence."""
    H, y, q = {l: [] for l in layers}, [], []
    for s in seqs:
        _, hh = lm.run(s["ids"], adds=s.get("adds", ()))
        for l in layers:
            H[l].append(hh[l])
        lab = torch.zeros(len(s["ids"]), device=hh.device)
        lab[list(s["targets"])] = 1.0
        y.append(lab)
        q.append(torch.full((len(s["ids"]),), s["q"], device=hh.device))
    return {l: torch.cat(H[l]) for l in layers}, torch.cat(y), torch.cat(q)


def fit_gates(H, y, q, lams=(1e-3, 1e-2, 1e-1), held_fraction=0.25):
    """One read vector per layer. The ridge strength is chosen on the last quarter of the questions: the one with the
    largest (smallest value at a target position) - (largest absolute value at any other position). Then refit on all."""
    qs = sorted(set(q.tolist()))
    held = set(qs[len(qs) - max(1, int(len(qs) * held_fraction)):])
    tr = torch.tensor([int(i) not in held for i in q.tolist()], device=y.device)
    V, stats = {}, []
    for l, Hl in H.items():
        best = None
        for lam in lams:
            g = Hl[~tr] @ fit_read_vector(Hl[tr], y[tr], lam)
            pos, neg = g[y[~tr] == 1], g[y[~tr] == 0]
            r = dict(layer=l, lam=lam, target_mean=float(pos.mean()), target_min=float(pos.min()),
                     other_max_abs=float(neg.abs().max()), other_p99_abs=float(neg.abs().quantile(0.99)),
                     margin=float(pos.min() - neg.abs().max()))
            if best is None or r["margin"] > best["margin"]:
                best = r
        V[l] = fit_read_vector(Hl, y, best["lam"])
        stats.append(best)
    return V, stats


def build_start_edit(lm, items, d, S, lams=(1e-3, 1e-2, 1e-1)):
    """Conditional edit for the answer-start position.
    items: dicts with ids0 (prompt without instruction), y_instr (answer of the instructed model), y_plain (answer of
    the plain model). d [L, d_model]: mean hidden-state difference at the answer-start position after each block.
    Returns (edits, fit statistics); edits = [dict(layer, u, v)]."""
    u = writes(d, S)
    seqs = []
    for qi, it in enumerate(items):
        p0 = len(it["ids0"]) - 1
        # the plain prompt with the differences switched on at the answer-start position, followed by the instructed answer
        seqs.append(dict(q=qi, ids=it["ids0"] + it["y_instr"], adds=[(l, p0, u[l]) for l in S], targets=[p0]))
        # the plain prompt as it is, followed by the plain answer
        seqs.append(dict(q=qi, ids=it["ids0"] + it["y_plain"], targets=[p0]))
    H, y, q = collect(lm, seqs, S)
    V, stats = fit_gates(H, y, q, lams)
    return [dict(layer=l, u=u[l].float().cpu(), v=V[l].cpu()) for l in S], stats


def save_edits(path, edits, **meta):
    z = {}
    for e in edits:
        z[f"u_L{e['layer']}"] = np.asarray(e["u"], np.float32)
        z[f"v_L{e['layer']}"] = np.asarray(e["v"], np.float32)
    for k, v in meta.items():
        z[f"meta_{k}"] = np.asarray(v)
    np.savez_compressed(path, **z)


def load_edits(path):
    z = np.load(path, allow_pickle=False)
    layers = sorted(int(k[3:]) for k in z.files if k.startswith("u_L"))
    return [dict(layer=l, u=torch.from_numpy(z[f"u_L{l}"]), v=torch.from_numpy(z[f"v_L{l}"])) for l in layers]
