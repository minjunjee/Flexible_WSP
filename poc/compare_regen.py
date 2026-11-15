"""Compare the result files of the original CPU run with the ones regenerated on the server.

Three result files exist in both versions: the original in this folder, the regenerated one in regen_check/.
results_D.json was not in the bundle; its regenerated rows are compared with the original results_D_rescored.json,
which holds per-condition rates that were scored again by a script we do not have. expD_install.py leaves the text
"<|im_end|>" in an answer and then counts it as two words, so an empty answer passes its own "at most three words"
test. The rescored file reports 0 words for such answers, so here the end tokens are removed before counting. That
rule is a guess at the original rescoring, not a copy of it.

    python compare_regen.py            # prints a summary, writes regen_check/comparison.json
"""
import collections
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(o, bool) or o is None or isinstance(o, str):
        yield path, o
    else:
        yield path, float(o)


OUT = {}
for name in ("results_loc.json", "results_anaA_check.json", "results_C_summary.json"):
    a = dict(leaves(json.load(open(f"{HERE}/{name}"))))
    p = f"{HERE}/regen_check/{name}"
    if not os.path.exists(p):
        print(f"{name}: no regenerated file")
        continue
    b = dict(leaves(json.load(open(p))))
    common = [k for k in a if k in b]
    num = [k for k in common if isinstance(a[k], float) and isinstance(b[k], float) and not (np.isnan(a[k]) or np.isnan(b[k]))]
    d = np.array([abs(a[k] - b[k]) for k in num])
    worst = sorted(num, key=lambda k: -abs(a[k] - b[k]))[:5]
    OUT[name] = dict(values_original=len(a), values_regenerated=len(b), only_in_original=len(set(a) - set(b)),
                     only_in_regenerated=len(set(b) - set(a)), numeric_compared=len(num), identical=int((d == 0).sum()),
                     within_0p01=int((d <= 0.01).sum()), within_0p05=int((d <= 0.05).sum()),
                     median_abs_diff=float(np.median(d)) if len(d) else None, max_abs_diff=float(d.max()) if len(d) else None,
                     largest=[dict(path=k, original=a[k], regenerated=b[k]) for k in worst])
    print(f"{name}: {len(num)} numbers compared, identical {int((d == 0).sum())}, within 0.01 {int((d <= 0.01).sum())}, "
          f"within 0.05 {int((d <= 0.05).sum())}, median |diff| {np.median(d):.4f}, max |diff| {d.max():.4f}")
    for k in worst:
        print(f"    {k}: original {a[k]:.4f}  regenerated {b[k]:.4f}")

# ---- expD: regenerated rows vs. the original rescored summary
O = json.load(open(f"{HERE}/results_D_rescored.json"))
N = json.load(open(f"{HERE}/results_D.json"))
OUT["results_D"] = {}


def words_of(text):
    return re.findall(r"[A-Za-z0-9]+", text.replace("<|im_end|>", "").replace("<|endoftext|>", ""))


def fmt_of(task, row):
    return row["fmt"] if task == "json" else 0 < len(words_of(row["text"])) <= 3


for task in O:
    agg = collections.defaultdict(list)
    for r in N[task]["rows"]:
        agg[(r["set"], r["cond"])].append(r)
    cells = []
    for sname, conds in O[task].items():
        for c, o in conds.items():
            rr = agg.get((sname, c))
            if rr:
                cells.append(dict(set=sname, cond=c, n=len(rr), n_original=o["n"],
                                  fmt_original=o["fmt"], fmt_regenerated=float(np.mean([fmt_of(task, x) for x in rr])),
                                  fmt_regenerated_as_scored_by_script=float(np.mean([x["fmt"] for x in rr])),
                                  ans_original=o["ans"], ans_regenerated=float(np.mean([x["correct"] for x in rr])),
                                  words_original=o["words"],
                                  words_regenerated=float(np.median([len(words_of(x["text"])) for x in rr]))))
    f = np.array([abs(c["fmt_original"] - c["fmt_regenerated"]) for c in cells])
    a = np.array([abs(c["ans_original"] - c["ans_regenerated"]) for c in cells])
    w = np.array([abs(c["words_original"] - c["words_regenerated"]) for c in cells])
    OUT["results_D"][task] = dict(cells=len(cells), fmt_identical=int((f < 1e-9).sum()), fmt_within_0p1=int((f <= 0.1 + 1e-9).sum()),
                                  ans_identical=int((a < 1e-9).sum()), ans_within_0p1=int((a <= 0.1 + 1e-9).sum()),
                                  median_words_within_1=int((w <= 1).sum()),
                                  fmt_max_abs_diff=float(f.max()), ans_max_abs_diff=float(a.max()), table=cells)
    print(f"results_D [{task}]: {len(cells)} (question set, condition) cells; format rate identical in {int((f < 1e-9).sum())}, "
          f"within 0.1 in {int((f <= 0.1 + 1e-9).sum())}; answer rate identical in {int((a < 1e-9).sum())}, "
          f"within 0.1 in {int((a <= 0.1 + 1e-9).sum())}; median word count within 1 word in {int((w <= 1).sum())}")
os.makedirs(f"{HERE}/regen_check", exist_ok=True)
json.dump(OUT, open(f"{HERE}/regen_check/comparison.json", "w"), indent=1)
