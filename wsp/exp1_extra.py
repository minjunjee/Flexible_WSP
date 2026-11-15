"""Experiment 1, extra comparisons that the main run does not have.

  1. The per-layer differences added at EVERY position of EVERY layer (hook), with one scale chosen on the tuning
     questions. With the main run this completes the 2 x 2 of {every position, answer start only} x {one layer, all
     layers}, so that "where" and "how many layers" are not mixed up.
  2. No edit and no instruction, but the first answer token is forced to '{"'. If this alone gives valid JSON, then
     what carries the format after the first token is the token itself, not a state left behind by the instruction.
  3. The same with the first four tokens forced to '{"answer": "'.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m wsp.exp1_extra --out results/exp1/extra
"""
import argparse
import json
import os
import time

import numpy as np

from wsp import edit
from wsp.lm import LM, load_questions
from wsp.score import boot_ci, score_json

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/exp1/extra")
ap.add_argument("--max_new", type=int, default=40)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
INSTR = 'Respond only with a JSON object of the form {"answer": "..."} and nothing else.'
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)
lm = LM("qwen35-0.8b")
QS = load_questions()
build, tune, test = ([q for q in QS if q["split"] == s] for s in ("build", "tune", "test"))
d = edit.start_effect(lm, [q["question"] for q in build], INSTR).mean(0)
u = edit.writes(d, list(range(lm.L)))

# ---- scale for "every position, every layer", chosen on the tuning questions
SCALES = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
acc = np.zeros((len(SCALES), 2))
for q in tune:
    outs = lm.generate_rows(lm.prompt_ids(q["question"]), [[(l, "all", s * u[l]) for l in range(lm.L)] for s in SCALES], args.max_new)
    for i, toks in enumerate(outs):
        sc = score_json(lm.decode(toks), q["gold"])
        acc[i] += (sc["valid"], sc["valid"] and sc["correct"])
acc /= len(tune)
bi = max(range(len(SCALES)), key=lambda i: (acc[i, 0], acc[i, 1]))
log("every position, every layer; valid JSON on the tuning questions by scale:", {s: round(float(a), 2) for s, a in zip(SCALES, acc[:, 0])})

P0 = [lm.prompt_ids(q["question"]) for q in test]
OPEN = lm.tok('{"', add_special_tokens=False).input_ids
PREFIX = lm.tok('{"answer": "', add_special_tokens=False).input_ids
conds = {
    f"differences added at every position of every layer, scale {SCALES[bi]} (hook)": dict(adds=[(l, "all", SCALES[bi] * u[l]) for l in range(lm.L)], forced=[]),
    "no edit, first answer token forced to '{\"'": dict(adds=[], forced=OPEN),
    "no edit, first four answer tokens forced to '{\"answer\": \"'": dict(adds=[], forced=PREFIX),
}
S = dict(tune_valid_by_scale={str(s): float(a) for s, a in zip(SCALES, acc[:, 0])}, forced_token_ids=dict(open=OPEN, prefix=PREFIX), conditions={})
for ci, (name, c) in enumerate(conds.items()):
    outs = lm.generate([p + c["forced"] for p in P0], args.max_new - len(c["forced"]), adds=c["adds"])
    rows = []
    for q, toks in zip(test, outs):
        txt = lm.decode(c["forced"] + toks)
        rows.append(dict(id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], text=txt, **score_json(txt, q["gold"])))
    with open(f"{args.out}/outputs_{ci}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    S["conditions"][name] = dict(file=f"outputs_{ci}.jsonl", n=len(rows), **{k: boot_ci([r[k] for r in rows]) for k in ("valid", "key_answer", "correct", "clean_end")},
                                 valid_by_type={t: float(np.mean([r["valid"] for r in rows if r["type"] == t])) for t in dict.fromkeys(r["type"] for r in rows)})
    s = S["conditions"][name]
    log(f"{name}: valid JSON {100 * s['valid'][0]:.1f}%, key 'answer' {100 * s['key_answer'][0]:.1f}%, correct {100 * s['correct'][0]:.1f}%, by type {s['valid_by_type']}")
json.dump(S, open(f"{args.out}/summary.json", "w"), indent=1, ensure_ascii=False)
log("DONE")
