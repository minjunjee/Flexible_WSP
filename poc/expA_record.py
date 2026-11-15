"""Experiment A: teacher-forced recording on released Qwen3.5-0.8B traces.

Components = MLP neurons (one rank-1 read/write pair per neuron), all 24 layers.
For every response token we record component-level complexity measures and activation-level baselines.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw

ap = argparse.ArgumentParser()
ap.add_argument("--n_base", type=int, default=110)
ap.add_argument("--n_math", type=int, default=110)
ap.add_argument("--n_gsm", type=int, default=110)
ap.add_argument("--out", default=f"{qw.ROOT}/outA")
ap.add_argument("--q", type=float, default=0.95)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(0)
torch.set_grad_enabled(False)

tok, model = qw.load_model()
lm = qw.lm_of(model)
ASTRA = qw.astra_deltas(model)
MATH = json.load(open(f"{qw.ROOT}/data/math500.json"))
GSM = json.load(open(f"{qw.ROOT}/data/gsm8k_test.json"))
WN = [lm.layers[l].mlp.down_proj.weight.float().norm(dim=0) for l in range(qw.N_LAYERS)]
U_ASTRA = {l: torch.linalg.svd(d, full_matrices=False)[0][:, 0] for l, (c, d) in ASTRA.items()}

H = {}


def mk(l):
    def hook(mod, inp, out):
        H[l] = inp[0][0].detach()
    return hook


for l, layer in enumerate(lm.layers):
    layer.mlp.down_proj.register_forward_hook(mk(l))


def forward(prompt_ids, resp_ids):
    ids = torch.tensor([prompt_ids + resp_ids])
    with torch.inference_mode():
        out = lm(input_ids=ids, output_hidden_states=True)
    return out.hidden_states, len(prompt_ids)


def problem_of(bench, pid):
    return (MATH if bench == "math500" else GSM)[pid]


def reach_index(text, gold, problem, offsets):
    """Token index where the gold number first appears in the response; None = not usable, -1 = never appears."""
    g = str(gold).replace(",", "").strip()
    try:
        gv = float(g)
    except ValueError:
        return None
    if abs(gv) < 10:
        return None
    variants = {g}
    if gv == int(gv):
        variants |= {str(int(gv)), f"{int(gv):,}"}
    pat = r"(?<![\d.,])(?:" + "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True)) + r")(?![\d]|[.,]\d)"
    if re.search(pat, problem):
        return None
    m = re.search(pat, text)
    if not m:
        return -1
    ch = m.start()
    for i, (a, b) in enumerate(offsets):
        if b > ch:
            return i
    return -1


# ------------------------------------------------------------------ choose traces
stats = json.load(open(f"{qw.ROOT}/released_stats.json"))
jobs = []
for bench in ["math500", "gsm8k"]:
    rows = qw.load_released("unedited", bench)
    st = {s["id"]: s for s in stats[f"{bench}_unedited"]}
    ok = [r for r in rows if st[r["id"]]["cut"] and st[r["id"]]["onset"] is not None and 300 <= st[r["id"]]["onset"] <= 1150]
    idx = rng.permutation(len(ok))[: args.n_base // 2]
    for i in idx:
        r = ok[i]
        o = st[r["id"]]["onset"]
        jobs.append(dict(set="base", bench=bench, row=r, onset=o, maxtok=min(o + 320, 1500)))
rows = qw.load_released("p50c", "math500")
for i in rng.permutation(len(rows))[: args.n_math]:
    jobs.append(dict(set="p50c_math", bench="math500", row=rows[i], onset=None, maxtok=768))
rows = qw.load_released("p50c", "gsm8k")
for i in rng.permutation(len(rows))[: args.n_gsm]:
    jobs.append(dict(set="p50c_gsm", bench="gsm8k", row=rows[i], onset=None, maxtok=1152))
print("jobs", len(jobs), {s: sum(j["set"] == s for j in jobs) for s in ["base", "p50c_math", "p50c_gsm"]}, flush=True)


def prep(job):
    r = job["row"]
    prob = problem_of(job["bench"], r["id"])
    p_ids = tok(qw.build_prompt(tok, prob["problem"]), add_special_tokens=False).input_ids
    enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
    return prob, p_ids, enc.input_ids, enc.offset_mapping


# ------------------------------------------------------------------ calibration of activity thresholds
t0 = time.time()
samples = [[] for _ in range(qw.N_LAYERS)]
cal = [j for s in ["base", "p50c_math", "p50c_gsm"] for j in [x for x in jobs if x["set"] == s][:5]]
for job in cal:
    prob, p_ids, r_ids, offs = prep(job)
    deltas = ASTRA if job["set"].startswith("p50c") else {}
    with qw.SwapWeights(model, deltas):
        hs, p0 = forward(p_ids, r_ids[:512])
    for l in range(qw.N_LAYERS):
        c = (H[l][p0:].float().abs() * WN[l]).reshape(-1)
        samples[l].append(c[torch.randint(0, c.numel(), (40000,))])
TAU = torch.stack([torch.quantile(torch.cat(s), args.q) for s in samples])
print("calibrated thresholds in", round(time.time() - t0, 1), "s; tau[0,8,15,19,23] =", TAU[[0, 8, 15, 19, 23]].numpy().round(4), flush=True)
torch.save(TAU, f"{args.out}/tau.pt")

# ------------------------------------------------------------------ accumulators (two halves for held-out use)
NC = 3  # base: 0 productive, 1 loop-during, 2 loop-onset ; p50c_gsm: 0 pre-reach, 1 post-reach
acc = {s: dict(sum_h=torch.zeros(2, NC, qw.N_LAYERS, qw.D_MLP, dtype=torch.float64),
               sum_act=torch.zeros(2, NC, qw.N_LAYERS, qw.D_MLP, dtype=torch.float64),
               sum_res=torch.zeros(2, NC, qw.N_LAYERS + 1, qw.D_MODEL, dtype=torch.float64),
               n=torch.zeros(2, NC, dtype=torch.float64)) for s in ["base", "p50c_gsm"]}
CKPT = [128, 256, 384, 512, 768, 1024]
SUB = 4
meta = []
for ji, job in enumerate(jobs):
    t1 = time.time()
    prob, p_ids, r_ids_full, offs = prep(job)
    r_ids = r_ids_full[: job["maxtok"]]
    T = len(r_ids)
    deltas = ASTRA if job["set"].startswith("p50c") else {}
    with qw.SwapWeights(model, deltas):
        hs, p0 = forward(p_ids, r_ids)
    dup = qw.dup_mask(np.array(r_ids_full))[:T]
    # ---- labels
    cls = np.full(T, -1)
    reach = None
    if job["set"] == "base":
        o = job["onset"]
        cls[: max(o - qw.GAP, 0)] = 0
        t = np.arange(T)
        cls[(t >= o + 64) & dup] = 1
        cls[max(o - 16, 0): min(o + 48, T)] = 2
    elif job["set"] == "p50c_gsm":
        reach = reach_index(job["row"]["text"], prob["answer"], prob["problem"], offs)
        if reach is not None and reach >= 0:
            t = np.arange(T)
            cls[t < reach] = 0
            cls[t > reach + 16] = 1
        elif reach == -1:
            cls[:] = 0
    half = ji % 2
    # ---- component-level per-token measures
    inst = np.zeros((T, qw.N_LAYERS), np.uint16)
    newc = np.zeros((T, qw.N_LAYERS), np.uint16)
    pr = np.zeros((T, qw.N_LAYERS), np.float32)
    union_ck = np.zeros((len(CKPT), qw.N_LAYERS), np.int32)
    usepr_ck = np.zeros((len(CKPT), qw.N_LAYERS), np.float32)
    A_sub = []
    cls_t = torch.from_numpy(cls)
    for l in range(qw.N_LAYERS):
        h = H[l][p0:].float()
        c = h.abs() * WN[l]
        act = c > TAU[l]
        cum = act.cumsum(0)
        inst[:, l] = act.sum(1).numpy()
        newc[:, l] = (act & (cum == 1)).sum(1).numpy()
        pr[:, l] = (c.sum(1) ** 2 / (c ** 2).sum(1)).numpy()
        ccum = c.cumsum(0)
        for k, L in enumerate(CKPT):
            if L <= T:
                union_ck[k, l] = int((cum[L - 1] > 0).sum())
                u = ccum[L - 1]
                usepr_ck[k, l] = float(u.sum() ** 2 / (u ** 2).sum())
        A_sub.append(act[:, ::SUB])
        if job["set"] in acc:
            a = acc[job["set"]]
            for k in range(NC):
                m = cls_t == k
                if m.any():
                    a["sum_h"][half, k, l] += h[m].sum(0).double()
                    a["sum_act"][half, k, l] += act[m].sum(0).double()
    if job["set"] in acc:
        a = acc[job["set"]]
        for k in range(NC):
            m = cls_t == k
            if m.any():
                a["n"][half, k] += float(m.sum())
                for li in range(qw.N_LAYERS + 1):
                    a["sum_res"][half, k, li] += hs[li][0, p0:][m].float().sum(0).double()
    # ---- return to earlier component combinations (Jaccard on a 1/4 neuron subsample)
    A = torch.cat(A_sub, 1).float()
    inter = A @ A.T
    sz = A.sum(1)
    J = inter / (sz[:, None] + sz[None, :] - inter).clamp_min(1)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=-16)
    Jm = J.masked_fill(~mask, -1.0)
    ret_max, ret_arg = Jm.max(1)
    ret_lag = (torch.arange(T) - ret_arg).numpy()
    ret_cnt = np.stack([((Jm > th).sum(1)).numpy() for th in (0.3, 0.4, 0.5, 0.6)], 1).astype(np.uint16)
    # mean Jaccard with the 64 preceding tokens (local continuity baseline)
    del inter, J, Jm, A
    # ---- activation-level baselines
    X15 = hs[16][0, p0:].float()
    X12 = hs[13][0, p0:].float()
    X19 = hs[20][0, p0:].float()
    Xc = X15 - X15.mean(0, keepdim=True)
    Xn = Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-6)
    C = (Xn @ Xn.T).masked_fill(~mask, -1.0)
    res_max = C.max(1).values.numpy()
    del C
    specdim = np.zeros((len(CKPT), 2), np.float32)
    for k, L in enumerate(CKPT):
        if L <= T:
            for m_i, X in enumerate((X12, X15)):
                s = torch.linalg.svdvals(X[:L] - X[:L].mean(0, keepdim=True))
                lam = s ** 2
                specdim[k, m_i] = float(lam.sum() ** 2 / (lam ** 2).sum())
    proj = np.stack([(X15 @ U_ASTRA[15]).numpy(), (X19 @ U_ASTRA[19]).numpy()], 1).astype(np.float32)
    out = dict(inst=inst, newc=newc, pr=pr, union_ck=union_ck, usepr_ck=usepr_ck, ret_max=ret_max.numpy().astype(np.float32),
               ret_lag=ret_lag.astype(np.int32), ret_cnt=ret_cnt, res_max=res_max.astype(np.float32), specdim=specdim,
               proj=proj, dup=dup, cls=cls.astype(np.int8), ids=np.array(r_ids, np.int32),
               mean_res15=X15.mean(0).numpy().astype(np.float32))
    if job["set"] == "base":
        out["res15"] = X15.numpy().astype(np.float16)
        out["res19"] = X19.numpy().astype(np.float16)
        out["h15"] = H[15][p0:].float().numpy().astype(np.float16)
        out["h19"] = H[19][p0:].float().numpy().astype(np.float16)
    np.savez(f"{args.out}/trace_{ji:04d}.npz", **out)
    r = job["row"]
    meta.append(dict(ji=ji, set=job["set"], bench=job["bench"], id=r["id"], T=T, n_full=len(r_ids_full), onset=job["onset"],
                     reach=reach, correct_first=r["correct_first"], correct_all=r["correct_all"], half=half,
                     level=prob.get("level"), subject=prob.get("subject"), closed=("</think>" in r["text"]),
                     cut=len(r_ids_full) >= 32000))
    if ji % 10 == 0 or ji == len(jobs) - 1:
        json.dump(meta, open(f"{args.out}/meta.json", "w"))
        torch.save(acc, f"{args.out}/acc.pt")
        print(f"[{ji + 1}/{len(jobs)}] {job['set']} T={T} {time.time() - t1:.1f}s | elapsed {time.time() - t0:.0f}s", flush=True)
json.dump(meta, open(f"{args.out}/meta.json", "w"))
torch.save(acc, f"{args.out}/acc.pt")
print("DONE", round(time.time() - t0, 1), "s", flush=True)
