"""Experiment 1, steps 3 and 5 (PLAN.md): the JSON instruction on the shared question set.

Edits are built from the 64 build questions, the two hook baselines choose their layer and scale on the 100 tuning
questions, and every condition is measured on the 500 test questions with greedy decoding. Conditions marked
"weights only" run with no hook: down_proj.weight is changed in place and restored afterwards.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m wsp.exp1_json --out results/exp1/main
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch

from wsp import edit
from wsp.lm import LM, load_questions
from wsp.score import boot_ci, score_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen35-0.8b")
ap.add_argument("--out", default="results/exp1/main")
ap.add_argument("--max_new", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(f"{args.out}/outputs", exist_ok=True)
os.makedirs(f"{args.out}/edits", exist_ok=True)
torch.manual_seed(args.seed)
INSTR = 'Respond only with a JSON object of the form {"answer": "..."} and nothing else.'
INSTR_LOWER = "Respond only in lowercase letters."
SCALES = (0.5, 1.0, 2.0, 4.0)

t0 = time.time()
lm = LM(args.model)
QS = load_questions()
build = [q for q in QS if q["split"] == "build"]
tune = [q for q in QS if q["split"] == "tune"]
test = [q for q in QS if q["split"] == "test"]
SETS = edit.layer_sets(lm.L)
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)
strip = lambda toks: [t for t in toks if t not in lm.end_ids]

# ------------------------------------------------------------------ build: what the instruction changes at the answer start
ids0 = [lm.prompt_ids(q["question"]) for q in build]
ids1 = [lm.prompt_ids(q["question"], INSTR) for q in build]
y_plain = [strip(t) for t in lm.generate(ids0, args.max_new)]
y_instr = [strip(t) for t in lm.generate(ids1, args.max_new)]
items = [dict(ids0=a, y_plain=b, y_instr=c) for a, b, c in zip(ids0, y_plain, y_instr)]
comply = np.mean([score_json(lm.tok.decode(y), q["gold"])["valid"] for y, q in zip(y_instr, build)])
diffs = edit.start_effect(lm, [q["question"] for q in build], INSTR)                 # [Q, L, d_model]
d = diffs.mean(0)
d_lower = edit.start_effect(lm, [q["question"] for q in build], INSTR_LOWER).mean(0)
shared = [edit.shared_fraction(diffs[:, l]) for l in range(lm.L)]
log(f"build questions {len(build)}; instructed model gives valid JSON on {comply:.2f}; "
    f"shared fraction of the difference at the last layer {shared[-1]:.2f}, at the middle layer {shared[lm.L // 2]:.2f}")

# ------------------------------------------------------------------ tune the two hook baselines (layer x scale)
grid = [(l, s) for l in range(lm.L) for s in SCALES]
tune_scores = {"all": np.zeros((len(grid), 2)), "start": np.zeros((len(grid), 2))}
for q in tune:
    p = lm.prompt_ids(q["question"])
    for where in ("all", "start"):
        outs = lm.generate_rows(p, [[(l, where, s * d[l])] for l, s in grid], args.max_new)
        for gi, toks in enumerate(outs):
            sc = score_json(lm.decode(toks), q["gold"])
            tune_scores[where][gi] += (sc["valid"], sc["valid"] and sc["correct"])
best = {}
for where, m in tune_scores.items():
    m /= len(tune)
    gi = max(range(len(grid)), key=lambda i: (m[i, 0], m[i, 1], -grid[i][1]))
    best[where] = dict(layer=grid[gi][0], scale=grid[gi][1], tune_valid=float(m[gi, 0]), tune_valid_and_correct=float(m[gi, 1]))
log("hook baselines chosen on the tuning questions:", best)

# ------------------------------------------------------------------ conditional edits for several layer sets
WEIGHT_SETS = ["all layers", "every 4th layer", "later half", "last quarter", "every 8th layer", "last 2 layers", "last layer"]
EDITS, fit_stats = {}, {}
for name in WEIGHT_SETS:
    EDITS[name], fit_stats[name] = edit.build_start_edit(lm, items, d, SETS[name])
    edit.save_edits(f"{args.out}/edits/{name.replace(' ', '_')}.npz", EDITS[name])
    st = fit_stats[name]
    log(f"read vectors for '{name}' ({len(st)} layers), held-out build questions: value at the answer start mean "
        f"{np.mean([s['target_mean'] for s in st]):.2f}, lowest {min(s['target_min'] for s in st):.2f}; "
        f"largest |value| elsewhere {max(s['other_max_abs'] for s in st):.2f}")

# controls on the every-4th-layer set
g = torch.Generator().manual_seed(args.seed)
base = EDITS["every 4th layer"]
ctrl_random_u, ctrl_shuffled_v, ctrl_lower_u = [], [], []
u_lower = edit.writes(d_lower, SETS["every 4th layer"])
for e in base:
    r = torch.randn(e["u"].shape, generator=g)
    ctrl_random_u.append(dict(layer=e["layer"], u=r / r.norm() * e["u"].norm(), v=e["v"]))
    ctrl_shuffled_v.append(dict(layer=e["layer"], u=e["u"], v=e["v"][torch.randperm(len(e["v"]), generator=g)]))
    ctrl_lower_u.append(dict(layer=e["layer"], u=u_lower[e["layer"]].float().cpu(), v=e["v"]))

# ------------------------------------------------------------------ conditions
C = {}                                                   # name -> dict(kind, ...)
C["no edit"] = dict(kind="plain")
C["instruction in the prompt"] = dict(kind="instruction")
b = best["all"]
C[f"difference added at every position of layer {b['layer']}, scale {b['scale']} (hook)"] = dict(kind="hook", adds=[(b["layer"], "all", b["scale"] * d[b["layer"]])])
b = best["start"]
C[f"difference added at the answer start only, layer {b['layer']}, scale {b['scale']} (hook)"] = dict(kind="hook", adds=[(b["layer"], "start", b["scale"] * d[b["layer"]])])
for name in ["all layers", "every 4th layer", "later half", "earlier half", "last quarter", "last 2 layers", "last layer"]:
    C[f"differences added at the answer start only, {name} (hook)"] = dict(kind="hook", adds=[(l, "start", u) for l, u in edit.writes(d, SETS[name]).items()])
for name in WEIGHT_SETS:
    C[f"conditional edits, {name} ({len(SETS[name])} edits, weights only)"] = dict(kind="weights", edits=EDITS[name], set=name)
C["control: every 4th layer, u replaced by random vectors of the same norm (weights only)"] = dict(kind="weights", edits=ctrl_random_u)
C["control: every 4th layer, entries of v shuffled (weights only)"] = dict(kind="weights", edits=ctrl_shuffled_v)
C["control: every 4th layer, u taken from the lowercase instruction (weights only)"] = dict(kind="weights", edits=ctrl_lower_u)

# ------------------------------------------------------------------ test
P0 = [lm.prompt_ids(q["question"]) for q in test]
P1 = [lm.prompt_ids(q["question"], INSTR) for q in test]
summary, gate_eval = {}, {}
for ci, (name, c) in enumerate(C.items()):
    changed = 0
    if c["kind"] == "weights":
        lm.apply(c["edits"])
        changed = lm.changed_weights()
    outs = lm.generate(P1 if c["kind"] == "instruction" else P0, args.max_new, adds=c.get("adds", ()))
    if c.get("set"):
        # values of the read vectors inside the edited model, on the text it wrote itself
        ge = dict(start=[], elsewhere_max=[])
        for p, toks in zip(P0, outs):
            _, hh = lm.run(p + strip(toks))
            vals = torch.stack([hh[e["layer"]] @ e["v"].to(hh.device) for e in c["edits"]])     # [edits, T]
            ge["start"].append(vals[:, len(p) - 1].tolist())
            ge["elsewhere_max"].append(float(torch.cat([vals[:, :len(p) - 1], vals[:, len(p):]], 1).abs().max()))
        gate_eval[name] = ge
    lm.restore()
    rows = []
    for q, toks in zip(test, outs):
        txt = lm.decode(toks)
        rows.append(dict(id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], text=txt,
                         ended=bool(toks and toks[-1] in lm.end_ids), **score_json(txt, q["gold"])))
    slug = f"{ci:02d}_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:80]
    with open(f"{args.out}/outputs/{slug}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = dict(kind=c["kind"], n=len(rows), changed_weights=changed, file=f"outputs/{slug}.jsonl")
    for k in ("valid", "key_answer", "correct", "clean_end"):
        x = [(r[k] and r["ended"]) if k == "clean_end" else r[k] for r in rows]
        s[k] = boot_ci(x)
        s[k + "_by_type"] = {t: float(np.mean([v for v, r in zip(x, rows) if r["type"] == t])) for t in dict.fromkeys(r["type"] for r in rows)}
    s["valid_and_correct"] = boot_ci([r["valid"] and r["correct"] for r in rows])
    summary[name] = s
    log(f"{name}: valid JSON {100 * s['valid'][0]:.1f}%, key 'answer' {100 * s['key_answer'][0]:.1f}%, correct {100 * s['correct'][0]:.1f}%")

json.dump(dict(model=args.model, instruction=INSTR, max_new=args.max_new, seed=args.seed, n_build=len(build), n_tune=len(tune),
               n_test=len(test), instructed_valid_on_build=float(comply), shared_fraction_by_layer=shared,
               hook_baselines=best, tune_grid=[dict(layer=l, scale=s, all_valid=float(tune_scores["all"][i, 0]),
                                                    start_valid=float(tune_scores["start"][i, 0])) for i, (l, s) in enumerate(grid)],
               fit_stats=fit_stats, gate_eval=gate_eval, conditions=summary), open(f"{args.out}/summary.json", "w"), indent=1)

pc = lambda t: f"{100 * t[0]:.1f} [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"
with open(f"{args.out}/table.md", "w") as f:
    f.write(f"질문 {len(test)}개, greedy. 값은 %이고 대괄호는 질문을 다시 뽑는 bootstrap으로 구한 95% 구간이다.\n\n")
    f.write("| 조건 | 유효한 JSON | key 이름이 answer | 정답 문자열이 들어 있음 | JSON 뒤에 다른 글이 없음 | 바뀐 가중치 수 |\n| --- | --- | --- | --- | --- | --- |\n")
    for name, s in summary.items():
        f.write(f"| {name} | {pc(s['valid'])} | {pc(s['key_answer'])} | {pc(s['correct'])} | {pc(s['clean_end'])} | {s['changed_weights']:,} |\n")
    f.write("\n유형별 유효한 JSON 비율(%)\n\n| 조건 | " + " | ".join(dict.fromkeys(q["type"] for q in test)) + " |\n| --- |" + " --- |" * 5 + "\n")
    for name, s in summary.items():
        f.write(f"| {name} | " + " | ".join(f"{100 * v:.0f}" for v in s["valid_by_type"].values()) + " |\n")
log("DONE")
