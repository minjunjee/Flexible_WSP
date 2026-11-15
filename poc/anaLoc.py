"""Where is the loop 'located'? Compare what the residual-direction view and the neuron view point to,
and how unique that location is. Uses A (base productive) and A2 (tight loop / productive mid-trace) statistics."""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw
from conds import gc_sets, heuristic_direction, stats

R = {}
f = safe_open(f"{qw.MP}/model.safetensors-00001-of-00001.safetensors", "pt")
W = [f.get_tensor(qw.down_proj_name(l)).float() for l in range(qw.N_LAYERS)]


class FakeLM:
    pass


S0, S1, SA = stats(0), stats(1), stats(None)
# ---- published edit: direction and columns
z = np.load(f"{qw.LOGS}/runs/p50c/edit/edit.npz")
man = json.load(open(f"{qw.LOGS}/runs/p50c/edit/manifest.json"))
astra = {}
for i, t in enumerate(man["tensors"]):
    l = int(t["name"].split(".layers.")[1].split(".")[0])
    base = W[l].to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16).copy()
    e = base.reshape(-1).copy()
    e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
    We = torch.from_numpy(e.reshape(base.shape).view(np.int16).copy()).view(torch.bfloat16).float()
    d = We - W[l]
    cols = torch.nonzero(d.abs().sum(0) > 0).squeeze(1)
    u = torch.linalg.svd(d[:, cols], full_matrices=False)[0][:, 0]
    astra[l] = (set(cols.tolist()), u)

print("=" * 100)
print("L1  Residual-direction view: how different is the residual stream in tight loops vs productive reasoning, by layer")
print(f"  {'layer':>5s} {'rel. gap':>9s} {'cos(half0,half1)':>17s} {'MLP write along u':>18s} {'top8 share':>11s} {'top64 share':>12s}")
rows = []
for l in range(qw.N_LAYERS):
    u0, u1, ua = heuristic_direction(S0, l), heuristic_direction(S1, l), heuristic_direction(SA, l)
    gap = (SA["res_tight"][l + 1] - SA["res_prod"][l + 1]).norm() / SA["res_prod"][l + 1].norm()
    dh = SA["h_tight"][l] - SA["h_prod"][l]
    contrib = (ua @ W[l]) * dh
    srt = contrib.sort(descending=True).values
    pos = srt[srt > 0].sum()
    rows.append(dict(layer=l, rel_gap=float(gap), stab=float(u0 @ u1), write=float(contrib.sum()), top8=float(srt[:8].sum() / pos),
                     top64=float(srt[:64].sum() / pos)))
    print(f"  {l:5d} {gap:9.3f} {float(u0 @ u1):17.3f} {float(contrib.sum()):18.3f} {float(srt[:8].sum() / pos):11.3f} {float(srt[:64].sum() / pos):12.3f}")
R["layers"] = rows
order = sorted(rows, key=lambda r: -r["write"])
print("  layers ranked by MLP write along the loop direction:", [r["layer"] for r in order][:10])
R["rank_15"] = [r["layer"] for r in order].index(15) + 1
R["rank_19"] = [r["layer"] for r in order].index(19) + 1
print(f"  rank of the published layers: 15 -> {R['rank_15']}, 19 -> {R['rank_19']} (of 24)")

print("=" * 100)
print("L2  Is 'the location' unique? Same recipe as the published edit, applied to our own statistics")
for l in (15, 19):
    cols_a, u_a = astra[l]
    ua = heuristic_direction(SA, l)
    print(f"  layer {l}: cosine between our loop direction and the published one = {float(ua @ u_a):.3f}")
    R[f"cos_dir_{l}"] = float(ua @ u_a)
    sel = {}
    for name, S in (("half0", S0), ("half1", S1), ("all", SA)):
        u = heuristic_direction(S, l)
        dh = S["h_tight"][l] - S["h_prod"][l]
        sel[name] = set(((u @ W[l]) * dh).topk(64).indices.tolist())
    # the published recipe with the published direction but our activations
    dh = SA["h_tight"][l] - SA["h_prod"][l]
    sel["published_dir_our_acts"] = set(((u_a @ W[l]) * dh).topk(64).indices.tolist())
    print(f"     top-64 writer neurons: overlap with the published 64 -> ours {len(sel['all'] & cols_a)}/64, "
          f"published direction + our activations {len(sel['published_dir_our_acts'] & cols_a)}/64; "
          f"our two data halves agree on {len(sel['half0'] & sel['half1'])}/64  (chance ~ {64 * 64 / 3584:.1f})")
    R[f"overlap_{l}"] = dict(ours=len(sel["all"] & cols_a), pubdir=len(sel["published_dir_our_acts"] & cols_a), halves=len(sel["half0"] & sel["half1"]))

print("=" * 100)
print("L3  Neuron view: neurons that are active in tight loops but not in productive reasoning (selected on half 0, checked on half 1)")
sets, score0 = gc_sets(S0)
score1 = S1["act_tight"] - torch.maximum(S1["act_prod"], S1["act_base"])
for k in (128, 1024):
    gc, rd = sets[k]["gc"], sets[k]["rand"]
    def held(d):
        t = np.concatenate([S1["act_tight"][l][ii].numpy() for l, ii in d.items()])
        p = np.concatenate([torch.maximum(S1["act_prod"], S1["act_base"])[l][ii].numpy() for l, ii in d.items()])
        return t.mean(), p.mean()
    gt, gp = held(gc)
    rt, rp = held(rd)
    lay = {int(l): int(len(ii)) for l, ii in gc.items()}
    print(f"  top-{k}: held-out active rate in loops {gt:.3f} vs productive {gp:.3f} | matched random control: loops {rt:.3f} vs productive {rp:.3f}")
    print(f"          layers (count): {dict(sorted(lay.items()))}")
    R[f"gc_{k}"] = dict(loop=float(gt), prod=float(gp), rand_loop=float(rt), rand_prod=float(rp), layers=lay)
sc = (SA["act_tight"] - torch.maximum(SA["act_prod"], SA["act_base"])).flatten()
R["n_specific"] = {str(th): int((sc > th).sum()) for th in (0.2, 0.3, 0.5, 0.7)}
print(f"  number of neurons (of 86,016) whose active rate is higher in loops by more than 0.2 / 0.3 / 0.5 / 0.7: {list(R['n_specific'].values())}")
neg = (torch.maximum(SA["act_prod"], SA["act_base"]) - SA["act_tight"]).flatten()
R["n_silenced"] = {str(th): int((neg > th).sum()) for th in (0.2, 0.3, 0.5)}
print(f"  number whose active rate is LOWER in loops by more than 0.2 / 0.3 / 0.5: {list(R['n_silenced'].values())}")
print(f"  mean active rate per neuron: loops {float(SA['act_tight'].mean()):.4f}  productive mid-trace {float(SA['act_prod'].mean()):.4f}  productive early {float(SA['act_base'].mean()):.4f}")
R["mean_act"] = dict(tight=float(SA["act_tight"].mean()), prod=float(SA["act_prod"].mean()), base=float(SA["act_base"].mean()))
# how many of the published 64 columns are loop-specific by the neuron view?
for l in (15, 19):
    cols = sorted(astra[l][0])
    s = (SA["act_tight"][l] - torch.maximum(SA["act_prod"], SA["act_base"])[l])[cols]
    rank = (sc.argsort(descending=True).argsort().reshape(qw.N_LAYERS, qw.D_MLP)[l][cols]).float()
    print(f"  published 64 columns in layer {l}: mean loop-specificity {float(s.mean()):.3f}; in our top-1024 loop-specific set: {int((rank < 1024).sum())}/64")
    R[f"pub_in_gc_{l}"] = int((rank < 1024).sum())
json.dump(R, open(f"{qw.ROOT}/results_loc.json", "w"))
print("saved results_loc.json")
