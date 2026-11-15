"""The all-position steering baseline again, with its layer and scale chosen for "follows the instruction AND the
answer is correct" on the tuning questions.

run_suite.py chose them for "follows the instruction" alone. For several instructions that picked a setting that
follows the instruction while destroying the answer (uppercase: 100% follow, 0.6% correct). This is the fairer choice;
both are reported.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python -m wsp.suite_fair_baseline --model qwen35-0.8b
"""
import argparse
import json
import os
import time

import numpy as np

from wsp import edit
from wsp.instructions import TASKS
from wsp.lm import LM, load_questions
from wsp.score import boot_ci, is_correct

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen35-0.8b")
ap.add_argument("--tasks", default=",".join(TASKS))
ap.add_argument("--max_new", type=int, default=96)
ap.add_argument("--tune_per_type", type=int, default=10)
args = ap.parse_args()
t0 = time.time()
lm = LM(args.model)
QS = load_questions()
build = [q for q in QS if q["split"] == "build"]
test = [q for q in QS if q["split"] == "test"]
tune, seen = [], {}
for q in QS:
    if q["split"] == "tune" and seen.get(q["type"], 0) < args.tune_per_type:
        seen[q["type"]] = seen.get(q["type"], 0) + 1
        tune.append(q)
P0 = [lm.prompt_ids(q["question"]) for q in test]
grid = [(l, s) for l in range(1, lm.L, 2) for s in (0.5, 1.0, 2.0, 4.0)]
for task in args.tasks.split(","):
    out = f"results/exp5/{args.model}/{task}"
    if not os.path.isdir(out):
        continue
    INSTR, check = TASKS[task]["text"], TASKS[task]["check"]
    d = edit.start_effect(lm, [q["question"] for q in build], INSTR).mean(0)
    acc = np.zeros((len(grid), 2))
    for q in tune:
        for gi, toks in enumerate(lm.generate_rows(lm.prompt_ids(q["question"]), [[(l, "all", s * d[l])] for l, s in grid], args.max_new)):
            txt = lm.decode(toks)
            f = bool(check(txt))
            acc[gi] += (f, f and is_correct(txt, q["gold"]))
    acc /= len(tune)
    gi = max(range(len(grid)), key=lambda i: (acc[i, 1], acc[i, 0], -grid[i][1]))
    l, s = grid[gi]
    rows = []
    for q, toks in zip(test, lm.generate(P0, args.max_new, adds=[(l, "all", s * d[l])])):
        txt = lm.decode(toks)
        rows.append(dict(id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], text=txt, follows=bool(check(txt)), correct=is_correct(txt, q["gold"])))
    with open(f"{out}/outputs_hook_every_position_one_layer_chosen_for_follows_and_correct.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    S = dict(layer=l, scale=s, n_tune=len(tune), tune_follows=float(acc[gi, 0]), tune_follows_and_correct=float(acc[gi, 1]),
             grid=[dict(layer=a, scale=b, follows=float(acc[i, 0]), follows_and_correct=float(acc[i, 1])) for i, (a, b) in enumerate(grid)],
             follows=boot_ci([r["follows"] for r in rows]), correct=boot_ci([r["correct"] for r in rows]),
             follows_and_correct=boot_ci([r["follows"] and r["correct"] for r in rows]))
    json.dump(S, open(f"{out}/baseline_chosen_for_follows_and_correct.json", "w"), indent=1)
    print(f"[{time.time() - t0:5.0f}s] {task}: layer {l}, scale {s}: follows {100 * S['follows'][0]:.1f}%, correct {100 * S['correct'][0]:.1f}%, "
          f"both {100 * S['follows_and_correct'][0]:.1f}%", flush=True)
print("DONE", flush=True)
