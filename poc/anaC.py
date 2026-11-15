"""Summarize the fork-inside-a-loop experiment (C) together with the answer-change (E) and side-effect (T) measures."""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/claude/poc")
import qw


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


R = json.load(open("/home/claude/poc/outC/results.json"))
E = json.load(open("/home/claude/poc/results_E.json"))
try:
    T = json.load(open("/home/claude/poc/results_T.json"))
except Exception:
    T = None
conds = []
for r in R:
    if r["cond"] not in conds:
        conds.append(r["cond"])
n_pref = len({r["prefix"] for r in R})
out = {}
print(f"forks inside a tight loop: {n_pref} loops (GSM8K {len({r['prefix'] for r in R if r['bench'] == 'gsm8k'})}, MATH-500 {len({r['prefix'] for r in R if r['bench'] == 'math500'})})")
print(f"{'condition':22s} {'left loop':>10s} {'95% CI':>14s} {'mean dup last128':>17s} {'closed thinking':>16s} {'boxed gold (gsm)':>17s} | {'short answers changed':>22s} {'short correct':>14s} | {'KL productive':>14s} {'dlogp loop':>11s}")
for c in conds:
    rr = [r for r in R if r["cond"] == c]
    esc = [r["dup_last"] is not None and r["dup_last"] < 0.2 for r in rr]
    k, n = int(np.sum(esc)), len(esc)
    lo, hi = wilson(k, n)
    closed = np.mean([r["closed_think"] for r in rr])
    g = [r for r in rr if r["bench"] == "gsm8k"]
    box = [qw.numeric_correct(r["text"], r["gold"]) for r in g]
    e = [x for x in E if x["cond"] == c]
    ch = 1 - np.mean([x["same_as_base"] for x in e])
    ec = np.mean([x["correct"] for x in e])
    kl = np.mean(T[c]["kl_prod"]) if T and T[c]["kl_prod"] else float("nan")
    dl = np.mean(T[c]["dlogp_loop"]) if T and T[c]["dlogp_loop"] else float("nan")
    out[c] = dict(escaped=k, n=n, lo=lo, hi=hi, dup_last=float(np.mean([r["dup_last"] for r in rr])), closed=float(closed),
                  boxed_correct=int(np.sum(box)), n_gsm=len(g), short_changed=float(ch), short_correct=float(ec), kl_prod=float(kl), dlogp_loop=float(dl))
    print(f"{c:22s} {k:>4d}/{n:<5d} [{lo:.2f}, {hi:.2f}] {np.mean([r['dup_last'] for r in rr]):17.2f} {closed:16.2f} {int(np.sum(box)):>8d}/{len(g):<8d} | {ch:22.2f} {ec:14.2f} | {kl:14.4f} {dl:11.4f}")
json.dump(out, open("/home/claude/poc/results_C_summary.json", "w"))
