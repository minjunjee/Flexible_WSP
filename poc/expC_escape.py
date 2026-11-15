"""Experiment C: fork inside a tight verbatim loop. The context is kept; only the weights differ between branches.

Context = problem prompt + 1,000 response tokens taken from inside a tight loop of a released base-model trace
(traces not used for choosing neurons or directions). Each branch then samples 256 new tokens.
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/claude/poc")
import qw
from conds import all_conditions
from gen import Editor, generate

N_PREFIX = int(sys.argv[1]) if len(sys.argv) > 1 else 12
ALT = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [11, 23]
NEW, CTX = 256, 1000
OUT = "/home/claude/poc/outC"
import os
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
tok, model = qw.load_model()
ed = Editor(model)
conds, sets, S = all_conditions(model, ALT)
names = list(conds)
print("conditions:", names, flush=True)
MATH = json.load(open(f"{qw.ROOT}/data/math500.json"))
GSM = json.load(open(f"{qw.ROOT}/data/gsm8k_test.json"))
used = {m["id"] for m in json.load(open("/home/claude/poc/outA2/meta.json"))}
rng = np.random.default_rng(7)


def tight_start(d16, w=512, frac=0.9):
    c = np.concatenate([[0], np.cumsum(d16)])
    dens = (c[w:] - c[:-w]) / w
    ok = np.flatnonzero(dens >= frac)
    return int(ok[0]) if len(ok) else None


cands = []
for bench in ["gsm8k", "math500"]:
    for r in qw.load_released("unedited", bench):
        if r["id"] in used:
            continue
        ids = np.array(tok(r["text"], add_special_tokens=False).input_ids)
        if len(ids) < 32000:
            continue
        ts = tight_start(qw.dup_mask(ids))
        if ts is not None and ts + 300 + CTX + NEW < len(ids):
            cands.append((bench, r, ids, ts))
pick = rng.permutation(len(cands))[:N_PREFIX]
results = []
t0 = time.time()
for pi, ci in enumerate(pick):
    bench, r, ids, ts = cands[ci]
    prob = (MATH if bench == "math500" else GSM)[r["id"]]
    p_ids = tok(qw.build_prompt(tok, prob["problem"]), add_special_tokens=False).input_ids
    seg = [int(x) for x in ids[ts + 300: ts + 300 + CTX]]
    B = len(names)
    ed.set(B, [conds[n] for n in names])
    res = generate(model, p_ids + seg, B, [1000 * pi + b for b in range(B)], NEW, history=seg, stop_loop=False)
    ed.clear()
    for n, x in zip(names, res):
        d = x["dup"]
        txt = tok.decode(x["tokens"])
        results.append(dict(prefix=pi, id=r["id"], bench=bench, cond=n, dup_all=float(d.mean()), dup_last=float(d[-128:].mean()) if len(d) >= 128 else None,
                            n=len(x["tokens"]), end=x["end"], closed_think="</think>" in txt, boxed=qw.last_boxed(txt), gold=prob["answer"],
                            uniq=len(set(x["tokens"])) / max(len(x["tokens"]), 1), text=txt))
    json.dump(results, open(f"{OUT}/results.json", "w"))
    row = {n: next(x["dup_last"] for x in results if x["prefix"] == pi and x["cond"] == n) for n in names}
    print(f"[{pi + 1}/{len(pick)}] {r['id']} elapsed {time.time() - t0:.0f}s | last-128 duplicate share: " + " ".join(f"{n}={v:.2f}" for n, v in row.items()), flush=True)
print("DONE", flush=True)
