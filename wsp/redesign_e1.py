"""E1 and the measurements of E2 from the redesign sheet (2026-11-16): prefix forcing against conditional edits.

For every instruction, on the same 200 test questions (the first 40 of each of the five types):
  C1 prompt      instruction in the prompt; its answer is the "reference answer"
  C2 none        no instruction, no edit
  C3 prefix-4    no instruction; the first 4 tokens of the reference answer are forced, the model continues
  C4 prefix-16   the same with the first 16 tokens
  C5 edit        no instruction; the conditional edits built earlier (results/exp5/<model>/<task>/edits.npz, all rounds)
  C6 edit + prefix-4
E2 (per question, from C1 and C2): d1 = first position where the two answers differ / length of the reference answer;
nD = number of reference positions where the uninstructed model, fed the reference answer, would greedily write a
different token; dL = relative position of the last such position.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python -m wsp.redesign_e1 --tasks json_answer,lowercase
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from wsp.edit import load_stacked
from wsp.instructions import TASKS
from wsp.lm import LM, load_questions
from wsp.score import is_correct

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True)
ap.add_argument("--model", default="qwen35-0.8b")
ap.add_argument("--per_type", type=int, default=40)
ap.add_argument("--max_new", type=int, default=96)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()
t0 = time.time()
lm = LM(args.model)
test, seen = [], {}
for q in load_questions():
    if q["split"] == "test" and seen.get(q["type"], 0) < args.per_type:
        seen[q["type"]] = seen.get(q["type"], 0) + 1
        test.append(q)
strip = lambda toks: [t for t in toks if t not in lm.end_ids]
os.makedirs("results", exist_ok=True)
for task in args.tasks.split(","):
    INSTR, check = TASKS[task]["text"], TASKS[task]["check"]
    P0 = [lm.prompt_ids(q["question"]) for q in test]
    P1 = [lm.prompt_ids(q["question"], INSTR) for q in test]
    path = f"results/exp5/{args.model}/{task}/edits.npz"
    rounds = load_stacked(path) if os.path.exists(path) else []
    out = {}
    lm.restore()
    ref = [strip(t) for t in lm.generate(P1, args.max_new, batch=args.batch)]
    out["C1 prompt"] = (ref, [[] for _ in ref])
    plain = [strip(t) for t in lm.generate(P0, args.max_new, batch=args.batch)]
    out["C2 none"] = (plain, [[] for _ in ref])
    for name, k in (("C3 prefix-4", 4), ("C4 prefix-16", 16)):
        forced = [r[:k] for r in ref]
        gen = lm.generate([p + f for p, f in zip(P0, forced)], args.max_new, batch=args.batch)
        out[name] = ([f + strip(g) for f, g in zip(forced, gen)], forced)
    # E2 measurements on the original model
    meas = []
    for p, y, o in zip(P0, ref, plain):
        d1 = next((i for i, (a, b) in enumerate(zip(y, o)) if a != b), min(len(y), len(o)))
        if y:
            top = lm.logprobs(p + y)[len(p) - 1: len(p) - 1 + len(y)].argmax(-1).tolist()
            div = [j for j, (a, b) in enumerate(zip(top, y)) if a != b]
        else:
            div = []
        meas.append(dict(ref_len=len(y), d1=d1 / max(len(y), 1), nD=len(div), dL=(div[-1] / max(len(y) - 1, 1)) if div else 0.0,
                         nD_share=len(div) / max(len(y), 1)))
    for e in rounds:
        lm.apply(e)
    out["C5 edit"] = ([strip(t) for t in lm.generate(P0, args.max_new, batch=args.batch)], [[] for _ in ref])
    forced = [r[:4] for r in ref]
    gen = lm.generate([p + f for p, f in zip(P0, forced)], args.max_new, batch=args.batch)
    out["C6 edit + prefix-4"] = ([f + strip(g) for f, g in zip(forced, gen)], forced)
    lm.restore()
    with open(f"results/E1_{task}.jsonl", "w") as f:
        for cond, (answers, forced) in out.items():
            for q, toks, fo, m, r in zip(test, answers, forced, meas, ref):
                txt = lm.tok.decode(toks).strip()
                row = dict(task=task, condition=cond, id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], text=txt, tokens=toks,
                           follows=bool(check(txt)), correct=is_correct(txt, q["gold"]), n_edit_groups=len(rounds),
                           prefix_covers_whole_reference=bool(fo) and len(fo) >= len(r))
                if cond == "C1 prompt":
                    row.update(m)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fl = {c: np.mean([bool(check(lm.tok.decode(t).strip())) for t in a]) for c, (a, _) in out.items()}
    print(f"[{time.time() - t0:5.0f}s] {task} ({len(rounds)} edit groups): " + ", ".join(f"{c} {100 * v:.0f}%" for c, v in fl.items()), flush=True)
print("DONE", flush=True)
