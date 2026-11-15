"""Stack conditional edits for one instruction along the model's own answers (wsp/onpolicy.py).

After every edit the test questions are answered with no hook and no instruction.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m wsp.run_onpolicy --out results/exp1/stack_json
"""
import argparse
import collections
import json
import os
import time

import numpy as np

from wsp import edit, onpolicy
from wsp.lm import LM, load_questions
from wsp.score import boot_ci, score_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen35-0.8b")
ap.add_argument("--out", required=True)
ap.add_argument("--instruction", default='Respond only with a JSON object of the form {"answer": "..."} and nothing else.')
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--layers", default="all layers")
ap.add_argument("--max_new", type=int, default=40)
ap.add_argument("--kl_min", type=float, default=1.0)
ap.add_argument("--min_group", type=float, default=0.1, help="stop when the best group has fewer positions than this share of the build questions")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

lm = LM(args.model)
QS = load_questions()
build = [q for q in QS if q["split"] == "build"]
test = [q for q in QS if q["split"] == "test"]
S = edit.layer_sets(lm.L)[args.layers]
Q = [dict(ids0=lm.prompt_ids(q["question"]), ids1=lm.prompt_ids(q["question"], args.instruction)) for q in build]
P0 = [lm.prompt_ids(q["question"]) for q in test]
P1 = [lm.prompt_ids(q["question"], args.instruction) for q in test]


def evaluate(prompts, tag):
    outs = lm.generate(prompts, args.max_new)
    rows = []
    for q, toks in zip(test, outs):
        txt = lm.decode(toks)
        rows.append(dict(id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], text=txt,
                         first_tokens=[lm.tok.decode([t]) for t in toks[:3]], ended=bool(toks and toks[-1] in lm.end_ids),
                         **score_json(txt, q["gold"])))
    with open(f"{args.out}/outputs_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = {k: boot_ci([r[k] for r in rows]) for k in ("valid", "key_answer", "correct")}
    s["key_answer_and_correct"] = boot_ci([r["key_answer"] and r["correct"] for r in rows])
    s["exactly_instructed_form"] = boot_ci([r["key_answer"] and r["clean_end"] and r["ended"] for r in rows])
    s["key_answer_by_type"] = {t: float(np.mean([r["key_answer"] for r in rows if r["type"] == t])) for t in dict.fromkeys(r["type"] for r in rows)}
    s["most_common_first_tokens"] = collections.Counter(" ".join(repr(t) for t in r["first_tokens"][:2]) for r in rows).most_common(5)
    log(f"{tag}: valid JSON {100 * s['valid'][0]:.1f}%, key 'answer' {100 * s['key_answer'][0]:.1f}%, correct {100 * s['correct'][0]:.1f}%; "
        f"first tokens {s['most_common_first_tokens'][:3]}")
    return s


R = dict(model=args.model, instruction=args.instruction, layers=args.layers, n_build=len(build), n_test=len(test), max_new=args.max_new,
         kl_min=args.kl_min, eval={}, rounds=[])
R["eval"]["instruction in the prompt"] = evaluate(P1, "instruction_in_the_prompt")
R["eval"]["no edit"] = evaluate(P0, "no_edit")
applied, all_edits = [], {}
for r in range(1, args.rounds + 1):
    info = onpolicy.measure(lm, Q, args.max_new, applied)
    prof = onpolicy.profile(info, args.kl_min)
    pts = onpolicy.next_points(info, args.kl_min)
    groups = onpolicy.candidate_groups(lm, info, pts)
    rec = dict(round=r, own_answers=prof, questions_with_an_untargeted_divergence=len(pts),
               candidate_groups=[{k: v for k, v in g.items() if k != "members"} for g in groups[:8]])
    log(f"round {r}: on its own answers the model disagrees with the instructed model at {prof['diverging']} of {prof['positions']} positions "
        f"(KL > {args.kl_min}); {len(pts)} of {len(Q)} questions have a divergence that no edit targets yet")
    for g in groups[:4]:
        log(f"   group ({g['kind']}): {g['size']} positions, median position {g['median_position']}, mean KL {g['kl_mean']:.2f}, "
            f"shared fraction {g['shared_fraction']:.2f}, instructed wants {g['instructed_next_tokens'][:3]}, current writes {g['current_next_tokens'][:3]}")
    if not groups or groups[0]["size"] < args.min_group * len(Q):
        rec["stopped"] = "no group is large enough"
        R["rounds"].append(rec)
        break
    edits, stats, members = onpolicy.build_edit(lm, Q, info, groups[0]["members"], S, args.max_new, args.kl_min, pts)
    lm.apply(edits)
    applied.append(edits)
    rec["positions_in_the_group"] = dict(chosen=groups[0]["size"], after_adding_positions_where_the_edit_fires_anyway=len(members))
    for e in edits:
        all_edits[f"round{r}_u_L{e['layer']}"] = e["u"].numpy()
        all_edits[f"round{r}_v_L{e['layer']}"] = e["v"].numpy()
    ok = [s for s in stats if not np.isnan(s["margin"])]
    rec["fit"] = dict(target_mean=float(np.mean([s["target_mean"] for s in ok])), target_min=float(min(s["target_min"] for s in ok)),
                      other_max_abs=float(max(s["other_max_abs"] for s in ok))) if ok else None
    rec["changed_weights"] = lm.changed_weights()
    rec["eval"] = evaluate(P0, f"after_{r}_edits")
    R["rounds"].append(rec)
    json.dump(R, open(f"{args.out}/summary.json", "w"), indent=1, ensure_ascii=False)
info = onpolicy.measure(lm, Q, args.max_new, applied)
R["own_answers_after_last_round"] = onpolicy.profile(info, args.kl_min)
np.savez_compressed(f"{args.out}/edits.npz", **all_edits)
json.dump(R, open(f"{args.out}/summary.json", "w"), indent=1, ensure_ascii=False)
log("final:", R["own_answers_after_last_round"])
log("DONE")
