"""Stack conditional edits for one instruction (PLAN.md, "갈라지는 자리를 찾아 편집을 쌓는 법").

Each round: measure KL(instructed || current model without instruction) at every aligned answer position on the build
questions, group the diverging positions, build one conditional edit for the group with the largest summed KL, add it
to the weights, and measure free generation on the test questions (no hook, no instruction).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m wsp.run_stack --out results/exp1/stack_json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from wsp import edit, stack
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
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

lm = LM(args.model)
QS = load_questions()
build = [q for q in QS if q["split"] == "build"]
test = [q for q in QS if q["split"] == "test"]
S = edit.layer_sets(lm.L)[args.layers]
P0 = [lm.prompt_ids(q["question"]) for q in test]
P1 = [lm.prompt_ids(q["question"], args.instruction) for q in test]


def evaluate(prompts, tag):
    outs = lm.generate(prompts, args.max_new)
    rows = []
    for q, toks in zip(test, outs):
        txt = lm.decode(toks)
        rows.append(dict(id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], text=txt,
                         ended=bool(toks and toks[-1] in lm.end_ids), **score_json(txt, q["gold"])))
    with open(f"{args.out}/outputs_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = {k: boot_ci([r[k] for r in rows]) for k in ("valid", "key_answer", "correct")}
    s["valid_and_key_and_correct"] = boot_ci([r["key_answer"] and r["correct"] for r in rows])
    s["key_answer_by_type"] = {t: float(np.mean([r["key_answer"] for r in rows if r["type"] == t])) for t in dict.fromkeys(r["type"] for r in rows)}
    log(f"{tag}: valid JSON {100 * s['valid'][0]:.1f}%, key 'answer' {100 * s['key_answer'][0]:.1f}%, correct {100 * s['correct'][0]:.1f}%")
    return s


cache = stack.instructed_cache(lm, [q["question"] for q in build], args.instruction, args.max_new)
R = dict(model=args.model, instruction=args.instruction, layers=args.layers, n_build=len(build), n_test=len(test), max_new=args.max_new,
         kl_min=args.kl_min, eval={}, rounds=[])
R["eval"]["instruction in the prompt"] = evaluate(P1, "instruction_in_the_prompt")
R["eval"]["no edit"] = evaluate(P0, "no_edit")
all_edits = {}
for r in range(1, args.rounds + 1):
    div = stack.divergence(lm, cache)
    klpos = stack.kl_by_position(cache, div)
    ch = stack.choose_cluster(lm, cache, div, args.kl_min)
    rec = dict(round=r, kl_by_position=klpos, n_positions=ch["n_positions"], n_diverging=ch["n_diverging"],
               clusters=[{k: v for k, v in c.items() if k != "members"} for c in ch["clusters"][:6]])
    log(f"round {r}: {ch['n_diverging']} of {ch['n_positions']} aligned positions have KL > {args.kl_min}; "
        f"{len(ch['clusters'])} groups; mean KL by position " + ", ".join(f"{k} {v['mean_kl']:.2f}" for k, v in klpos.items() if v["mean_kl"] is not None))
    if not ch["members"]:
        R["rounds"].append(rec)
        break
    c0 = ch["clusters"][0]
    log(f"   chosen group: {c0['size']} positions in {c0['questions']} questions, median position {c0['median_position']}, "
        f"mean KL {c0['kl_mean']:.2f}, shared fraction {c0['shared_fraction']:.2f}, instructed next tokens {c0['instructed_next_tokens'][:4]}")
    for c in ch["clusters"][1:5]:
        log(f"   other group: {c['size']} positions in {c['questions']} questions, median position {c['median_position']}, "
            f"mean KL {c['kl_mean']:.2f}, instructed next tokens {c['instructed_next_tokens'][:4]}")
    edits, stats = stack.build_edit(lm, cache, div, ch["members"], S, args.max_new, ch["diverging"])
    lm.apply(edits)
    for e in edits:
        all_edits[f"round{r}_u_L{e['layer']}"] = e["u"].numpy()
        all_edits[f"round{r}_v_L{e['layer']}"] = e["v"].numpy()
    rec["fit"] = dict(target_mean=float(np.mean([s["target_mean"] for s in stats])), target_min=float(min(s["target_min"] for s in stats)),
                      other_max_abs=float(max(s["other_max_abs"] for s in stats)))
    rec["changed_weights"] = lm.changed_weights()
    rec["eval"] = evaluate(P0, f"after_{r}_edits")
    R["rounds"].append(rec)
    json.dump(R, open(f"{args.out}/summary.json", "w"), indent=1, ensure_ascii=False)
div = stack.divergence(lm, cache)
R["kl_by_position_after_last_round"] = stack.kl_by_position(cache, div)
np.savez_compressed(f"{args.out}/edits.npz", **all_edits)
json.dump(R, open(f"{args.out}/summary.json", "w"), indent=1, ensure_ascii=False)
log("DONE")
