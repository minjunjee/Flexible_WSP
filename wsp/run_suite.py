"""One instruction, from measurement to stacked edits (experiment 5 in PLAN.md, with the on-policy procedure).

For the instruction it records, in this order:
  1. how often the model follows it when it is in the prompt, and when it is not (test questions)
  2. what can be measured BEFORE any edit is built: on the model's own answers to the build questions, where it parts
     ways with the instructed model, how those positions group, and how much of the hidden-state difference is shared
  3. two inference-time baselines: the answer-start difference added at every position of one layer (layer and scale
     chosen on tuning questions), and the per-layer differences added at the answer start only
  4. conditional edits stacked along the model's own answers (wsp/onpolicy.py), measured after every edit with no hook

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m wsp.run_suite --task lowercase --out results/exp5/qwen35-0.8b/lowercase
"""
import argparse
import json
import os
import time

import numpy as np

from wsp import edit, onpolicy
from wsp.instructions import TASKS
from wsp.lm import LM, load_questions
from wsp.score import boot_ci, is_correct

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=list(TASKS))
ap.add_argument("--model", default="qwen35-0.8b")
ap.add_argument("--out", required=True)
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--max_new", type=int, default=96)
ap.add_argument("--kl_min", type=float, default=1.0)
ap.add_argument("--min_group", type=float, default=0.1)
ap.add_argument("--tune_per_type", type=int, default=10)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T = TASKS[args.task]
INSTR, check = T["text"], T["check"]
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

lm = LM(args.model)
QS = load_questions()
build = [q for q in QS if q["split"] == "build"]
test = [q for q in QS if q["split"] == "test"]
tune, seen = [], {}
for q in QS:
    if q["split"] == "tune" and seen.get(q["type"], 0) < args.tune_per_type:
        seen[q["type"]] = seen.get(q["type"], 0) + 1
        tune.append(q)
S = list(range(lm.L))
Q = [dict(ids0=lm.prompt_ids(q["question"]), ids1=lm.prompt_ids(q["question"], INSTR)) for q in build]
P0 = [lm.prompt_ids(q["question"]) for q in test]
P1 = [lm.prompt_ids(q["question"], INSTR) for q in test]
R = dict(task=args.task, instruction=INSTR, source=T["source"], model=args.model, n_build=len(build), n_tune=len(tune), n_test=len(test),
         max_new=args.max_new, kl_min=args.kl_min, conditions={}, rounds=[])


def evaluate(name, prompts, tag, adds=()):
    outs = lm.generate(prompts, args.max_new, adds=adds)
    rows = []
    for q, toks in zip(test, outs):
        txt = lm.decode(toks)
        rows.append(dict(id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], text=txt, ended=bool(toks and toks[-1] in lm.end_ids),
                         follows=bool(check(txt)), correct=is_correct(txt, q["gold"]), tokens=len(toks)))
    with open(f"{args.out}/outputs_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = dict(file=f"outputs_{tag}.jsonl", follows=boot_ci([r["follows"] for r in rows]), correct=boot_ci([r["correct"] for r in rows]),
             follows_and_correct=boot_ci([r["follows"] and r["correct"] for r in rows]), ended=float(np.mean([r["ended"] for r in rows])),
             median_tokens=float(np.median([r["tokens"] for r in rows])),
             follows_by_type={t: float(np.mean([r["follows"] for r in rows if r["type"] == t])) for t in dict.fromkeys(r["type"] for r in rows)})
    R["conditions"][name] = s
    log(f"{name}: follows {100 * s['follows'][0]:.1f}%, correct {100 * s['correct'][0]:.1f}%, ended {100 * s['ended']:.0f}%, median tokens {s['median_tokens']:.0f}")
    json.dump(R, open(f"{args.out}/summary.json", "w"), indent=1, ensure_ascii=False)
    return s


# ---- 1. with and without the instruction
evaluate("instruction in the prompt", P1, "instruction_in_the_prompt")
evaluate("no edit", P0, "no_edit")

# ---- 2. measured before any edit
info = onpolicy.measure(lm, Q, args.max_new)
pts = onpolicy.next_points(info, args.kl_min)
groups = onpolicy.candidate_groups(lm, info, pts)
diffs = edit.start_effect(lm, [q["question"] for q in build], INSTR)                 # [Q, L, d]
d = diffs.mean(0)
first_pos = [j for _, j in pts]
R["before_any_edit"] = dict(
    own_answers=onpolicy.profile(info, args.kl_min),
    questions_with_a_divergence=len(pts), first_divergence_at_answer_start=float(np.mean([j == 0 for j in first_pos])) if pts else None,
    median_first_divergence_position=float(np.median(first_pos)) if pts else None,
    largest_group_share=groups[0]["size"] / len(Q) if groups else 0.0, largest_group_shared_fraction=groups[0]["shared_fraction"] if groups else None,
    n_token_groups=sum(g["kind"].startswith("same next token") for g in groups), n_direction_groups=sum(g["kind"].startswith("similar direction") for g in groups),
    shared_fraction_at_answer_start_last_layer=edit.shared_fraction(diffs[:, -1]), shared_fraction_at_answer_start_middle_layer=edit.shared_fraction(diffs[:, lm.L // 2]),
    candidate_groups=[{k: v for k, v in g.items() if k != "members"} for g in groups[:6]])
b = R["before_any_edit"]
log(f"before any edit: diverging positions {b['own_answers']['diverging']} of {b['own_answers']['positions']} ({100 * b['own_answers']['share']:.1f}%); "
    f"first divergence at the answer start in {100 * (b['first_divergence_at_answer_start'] or 0):.0f}% of questions; "
    f"largest group covers {100 * b['largest_group_share']:.0f}% of questions, shared fraction {b['largest_group_shared_fraction']}; "
    f"token groups {b['n_token_groups']}, direction groups {b['n_direction_groups']}; shared fraction at the answer start {b['shared_fraction_at_answer_start_last_layer']:.2f}")

# ---- 3. inference-time baselines
grid = [(l, s) for l in range(1, lm.L, 2) for s in (1.0, 2.0, 4.0)]
acc = np.zeros((len(grid), 2))
for q in tune:
    outs = lm.generate_rows(lm.prompt_ids(q["question"]), [[(l, "all", s * d[l])] for l, s in grid], args.max_new)
    for gi, toks in enumerate(outs):
        txt = lm.decode(toks)
        f = bool(check(txt))
        acc[gi] += (f, f and is_correct(txt, q["gold"]))
acc /= len(tune)
gi = max(range(len(grid)), key=lambda i: (acc[i, 0], acc[i, 1], -grid[i][1]))
R["all_position_baseline_choice"] = dict(layer=grid[gi][0], scale=grid[gi][1], tune_follows=float(acc[gi, 0]), tune_follows_and_correct=float(acc[gi, 1]), n_tune=len(tune))
evaluate(f"difference added at every position of layer {grid[gi][0]}, scale {grid[gi][1]} (hook)", P0, "hook_every_position_one_layer",
         adds=[(grid[gi][0], "all", grid[gi][1] * d[grid[gi][0]])])
evaluate("differences added at the answer start only, all layers (hook)", P0, "hook_answer_start_all_layers", adds=[(l, "start", u) for l, u in edit.writes(d, S).items()])

# ---- 4. conditional edits stacked along the model's own answers
applied, all_edits = [], {}
for r in range(1, args.rounds + 1):
    if r > 1:
        info = onpolicy.measure(lm, Q, args.max_new, applied)
        pts = onpolicy.next_points(info, args.kl_min)
        groups = onpolicy.candidate_groups(lm, info, pts)
    prof = onpolicy.profile(info, args.kl_min)
    rec = dict(round=r, own_answers=prof, questions_with_an_untargeted_divergence=len(pts),
               candidate_groups=[{k: v for k, v in g.items() if k != "members"} for g in groups[:6]])
    for g in groups[:3]:
        log(f"   round {r} group ({g['kind']}): {g['size']} positions, median position {g['median_position']}, mean KL {g['kl_mean']:.2f}, "
            f"shared fraction {g['shared_fraction']:.2f}, instructed wants {g['instructed_next_tokens'][:3]}, current writes {g['current_next_tokens'][:3]}")
    if not groups or groups[0]["size"] < args.min_group * len(Q):
        rec["stopped"] = "no group is large enough"
        R["rounds"].append(rec)
        log(f"round {r}: stopped, no group is large enough ({len(pts)} questions still have an untargeted divergence)")
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
    rec["chosen_group"] = {k: v for k, v in groups[0].items() if k != "members"}
    rec["changed_weights"] = lm.changed_weights()
    R["rounds"].append(rec)
    evaluate(f"conditional edits at {r} position group{'s' if r > 1 else ''} (weights only)", P0, f"after_{r}_edits")
info = onpolicy.measure(lm, Q, args.max_new, applied)
R["own_answers_after_last_round"] = onpolicy.profile(info, args.kl_min)
if all_edits:
    np.savez_compressed(f"{args.out}/edits.npz", **all_edits)
json.dump(R, open(f"{args.out}/summary.json", "w"), indent=1, ensure_ascii=False)
log("DONE")
