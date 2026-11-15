"""Experiment A2: neuron statistics inside tight verbatim loops vs. productive mid-trace reasoning.

Tight loop start = first response position where the following 512 tokens are >= 90% 16-gram duplicates.
Because traces are 32k tokens long, each segment is fed with a truncated context: prompt + 1,200 response tokens.
  class 'tight'    : base-model traces, segment starting 300 tokens after the tight-loop start
  class 'prod_mid' : edited-model traces that finished with a correct answer, a mid-trace segment with < 5% duplicates
The first 256 tokens of every segment are burn-in and are excluded from statistics.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw

torch.set_grad_enabled(False)
OUT = f"{qw.ROOT}/outA2"
import os
os.makedirs(OUT, exist_ok=True)
N_TIGHT, N_PROD, SEG, BURN = int(sys.argv[1]) if len(sys.argv) > 1 else 40, int(sys.argv[2]) if len(sys.argv) > 2 else 30, 1200, 256
rng = np.random.default_rng(1)
tok, model = qw.load_model()
lm = qw.lm_of(model)
MATH = json.load(open(f"{qw.ROOT}/data/math500.json"))
GSM = json.load(open(f"{qw.ROOT}/data/gsm8k_test.json"))
WN = [lm.layers[l].mlp.down_proj.weight.float().norm(dim=0) for l in range(qw.N_LAYERS)]
TAU = torch.load(f"{qw.ROOT}/outA/tau.pt")
H = {}
for l, layer in enumerate(lm.layers):
    layer.mlp.down_proj.register_forward_hook(lambda mod, inp, out, l=l: H.__setitem__(l, inp[0][0].detach()))


def tight_start(d16, w=512, frac=0.9):
    c = np.concatenate([[0], np.cumsum(d16)])
    if len(d16) <= w:
        return None
    dens = (c[w:] - c[:-w]) / w
    ok = np.flatnonzero(dens >= frac)
    return int(ok[0]) if len(ok) else None


jobs = []
for bench in ["gsm8k", "math500"]:
    rows = qw.load_released("unedited", bench)
    cand = []
    for r in rows:
        ids = np.array(tok(r["text"], add_special_tokens=False).input_ids)
        if len(ids) < 32000:
            continue
        ts = tight_start(qw.dup_mask(ids))
        if ts is not None and ts + 300 + SEG < len(ids):
            cand.append((r, ids, ts))
    for i in rng.permutation(len(cand))[: N_TIGHT // 2]:
        r, ids, ts = cand[i]
        jobs.append(dict(cls="tight", bench=bench, id=r["id"], seg=ids[ts + 300: ts + 300 + SEG], start=ts + 300, ts=ts))
    rows = qw.load_released("p50c", bench)
    cand = []
    for r in rows:
        if r["correct_first"] != 1:
            continue
        ids = np.array(tok(r["text"], add_special_tokens=False).input_ids)
        if len(ids) >= 32000 or len(ids) < 2600:
            continue
        d16 = qw.dup_mask(ids)
        s = min(1200, len(ids) - SEG - 200)
        if d16[s: s + SEG].mean() < 0.05:
            cand.append((r, ids, s))
    for i in rng.permutation(len(cand))[: N_PROD // 2]:
        r, ids, s = cand[i]
        jobs.append(dict(cls="prod_mid", bench=bench, id=r["id"], seg=ids[s: s + SEG], start=s, ts=None))
print("jobs", {c: sum(j["cls"] == c for j in jobs) for c in ["tight", "prod_mid"]}, flush=True)

acc = {c: dict(sum_h=torch.zeros(2, qw.N_LAYERS, qw.D_MLP, dtype=torch.float64),
               sum_h2=torch.zeros(2, qw.N_LAYERS, qw.D_MLP, dtype=torch.float64),
               sum_act=torch.zeros(2, qw.N_LAYERS, qw.D_MLP, dtype=torch.float64),
               sum_res=torch.zeros(2, qw.N_LAYERS + 1, qw.D_MODEL, dtype=torch.float64),
               n=torch.zeros(2, dtype=torch.float64)) for c in ["tight", "prod_mid"]}
meta = []
t0 = time.time()
for ji, job in enumerate(jobs):
    prob = (MATH if job["bench"] == "math500" else GSM)[job["id"]]
    p_ids = tok(qw.build_prompt(tok, prob["problem"]), add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + [int(x) for x in job["seg"]]])
    with torch.inference_mode():
        out = lm(input_ids=ids, output_hidden_states=True)
    p0 = len(p_ids) + BURN
    half = ji % 2
    a = acc[job["cls"]]
    inst = np.zeros(SEG - BURN)
    for l in range(qw.N_LAYERS):
        h = H[l][p0:].float()
        act = (h.abs() * WN[l]) > TAU[l]
        a["sum_h"][half, l] += h.sum(0).double()
        a["sum_h2"][half, l] += (h ** 2).sum(0).double()
        a["sum_act"][half, l] += act.sum(0).double()
        inst += act.sum(1).numpy()
    for li in range(qw.N_LAYERS + 1):
        a["sum_res"][half, li] += out.hidden_states[li][0, p0:].float().sum(0).double()
    a["n"][half] += SEG - BURN
    np.savez(f"{OUT}/seg_{ji:03d}.npz", h15=H[15][p0:].float().numpy().astype(np.float16), h19=H[19][p0:].float().numpy().astype(np.float16),
             res15=out.hidden_states[16][0, p0:].float().numpy().astype(np.float16),
             res19=out.hidden_states[20][0, p0:].float().numpy().astype(np.float16), inst=inst.astype(np.float32))
    meta.append(dict(ji=ji, cls=job["cls"], bench=job["bench"], id=job["id"], start=int(job["start"]), ts=job["ts"], half=half,
                     seg=[int(x) for x in job["seg"]]))
    if ji % 10 == 0:
        print(f"[{ji + 1}/{len(jobs)}] {job['cls']} elapsed {time.time() - t0:.0f}s", flush=True)
json.dump(meta, open(f"{OUT}/meta.json", "w"))
torch.save(acc, f"{OUT}/acc.pt")
print("DONE", round(time.time() - t0, 1), flush=True)
