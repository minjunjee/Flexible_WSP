"""Build the edit conditions compared in the intervention experiments.

All statistics used to choose neurons / directions come from half 0 of the recordings (A, A2);
half 1 is kept for held-out checks.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw

A = f"{qw.ROOT}/outA"
A2 = f"{qw.ROOT}/outA2"


def stats(half=0):
    accA = torch.load(f"{A}/acc.pt")
    accT = torch.load(f"{A2}/acc.pt")
    hs = [0, 1] if half is None else [half]
    t, m, b = accT["tight"], accT["prod_mid"], accA["base"]
    n_t, n_m, n_b = sum(t["n"][h] for h in hs), sum(m["n"][h] for h in hs), sum(b["n"][h, 0] for h in hs)
    S = dict(
        act_tight=sum(t["sum_act"][h] for h in hs) / n_t, act_prod=sum(m["sum_act"][h] for h in hs) / n_m,
        act_base=sum(b["sum_act"][h, 0] for h in hs) / n_b,
        h_tight=sum(t["sum_h"][h] for h in hs) / n_t, h_prod=sum(m["sum_h"][h] for h in hs) / n_m,
        res_tight=sum(t["sum_res"][h] for h in hs) / n_t, res_prod=sum(m["sum_res"][h] for h in hs) / n_m,
        h2_tight=sum(t["sum_h2"][h] for h in hs) / n_t, h2_prod=sum(m["sum_h2"][h] for h in hs) / n_m)
    return {k: v.float() for k, v in S.items()}


def gc_sets(S, ks=(128, 1024), seed=0):
    """Loop-specific neurons: active in tight loops, not in productive reasoning. Returns {k: [(layer, idx array)]}
    plus an activity-matched random control for each k."""
    score = S["act_tight"] - torch.maximum(S["act_prod"], S["act_base"])
    prod = torch.maximum(S["act_prod"], S["act_base"])
    flat = score.flatten()
    order = flat.argsort(descending=True)
    rng = np.random.default_rng(seed)
    out = {}
    top5 = set(order[: int(0.05 * len(order))].tolist())
    for k in ks:
        sel = order[:k]
        layers, idx = (sel // qw.D_MLP).numpy(), (sel % qw.D_MLP).numpy()
        gc = {int(l): idx[layers == l] for l in np.unique(layers)}
        ctrl = {}
        for l, ii in gc.items():
            pl = prod[l].numpy()
            pool = np.array([j for j in range(qw.D_MLP) if (l * qw.D_MLP + j) not in top5])
            chosen = []
            used = set()
            for j in ii:
                d = np.abs(pl[pool] - pl[j])
                cand = pool[np.argsort(d)[:20]]
                cand = [c for c in cand if c not in used]
                c = int(rng.choice(cand)) if cand else int(rng.choice(pool))
                used.add(c)
                chosen.append(c)
            ctrl[l] = np.array(chosen)
        out[k] = dict(gc=gc, rand=ctrl)
    return out, score


def heuristic_direction(S, l):
    u = S["res_tight"][l + 1] - S["res_prod"][l + 1]
    return u / u.norm()


def recipe_edit(model, S, layers, k=64, alpha=6.0):
    """The published recipe (top-k writer columns, subtract alpha * (u.w_j) u) with a direction from our own statistics."""
    lm = qw.lm_of(model)
    out = {}
    for l in layers:
        W = lm.layers[l].mlp.down_proj.weight.detach().float()
        u = heuristic_direction(S, l)
        dh = S["h_tight"][l] - S["h_prod"][l]
        proj = u @ W
        cols = (proj * dh).topk(k).indices
        D = -alpha * torch.outer(u, proj[cols])
        out[l] = (cols, D, u)
    return out


def layer_write_strength(model, S):
    """For each layer: how strongly its MLP writes along that layer's loop direction in loops relative to productive text."""
    lm = qw.lm_of(model)
    rows = []
    for l in range(qw.N_LAYERS):
        W = lm.layers[l].mlp.down_proj.weight.detach().float()
        u = heuristic_direction(S, l)
        dh = S["h_tight"][l] - S["h_prod"][l]
        contrib = (u @ W) * dh
        gap = (S["res_tight"][l + 1] - S["res_prod"][l + 1]).norm().item()
        scale = S["res_prod"][l + 1].norm().item()
        srt = contrib.sort(descending=True).values
        pos = srt[srt > 0]
        rows.append(dict(layer=l, mlp_write=float(contrib.sum()), gap=gap, rel_gap=gap / scale,
                         top8=float(srt[:8].sum()), top64=float(srt[:64].sum()), pos_total=float(pos.sum())))
    return rows


def all_conditions(model, alt_layers):
    S = stats(0)
    astra = qw.astra_deltas(model)
    lm = qw.lm_of(model)
    sets, score = gc_sets(S)
    steer = np.load(f"{qw.ROOT}/ext/weight-space-lp/steer/realshift_k8.npz")
    conds = {"base": []}
    conds["astra_edit"] = [("delta", l, cols, D) for l, (cols, D) in astra.items()]
    conds["lp_push"] = [("push", int(l), torch.tensor(steer[l]).float()) for l in ("15", "19")]
    ab = []
    for l, (cols, D) in astra.items():
        u = torch.linalg.svd(D, full_matrices=False)[0][:, 0]
        W = lm.layers[l].mlp.down_proj.weight.detach().float()
        ab.append(("rank1", l, u, -(u @ W)))
    conds["remove_direction"] = ab
    for k in (128, 1024):
        conds[f"gc_{k}"] = [("zero", l, torch.tensor(ii)) for l, ii in sets[k]["gc"].items()]
    conds["random_1024"] = [("zero", l, torch.tensor(ii)) for l, ii in sets[1024]["rand"].items()]
    r = recipe_edit(model, S, [15, 19])
    conds["recipe_own_15_19"] = [("delta", l, cols, D) for l, (cols, D, u) in r.items()]
    r2 = recipe_edit(model, S, alt_layers)
    conds[f"recipe_own_{alt_layers[0]}_{alt_layers[1]}"] = [("delta", l, cols, D) for l, (cols, D, u) in r2.items()]
    return conds, sets, S
