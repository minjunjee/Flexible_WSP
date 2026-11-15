"""Summarise E1: results/E1_summary.csv and a markdown table (results/E1_table.md).

Rates come with 95% Wilson intervals. For the two prefix conditions the rate is also given on the questions where the
forced prefix is shorter than the reference answer: when the prefix covers the whole reference answer, forcing it is
the same as writing the instructed answer for the model.
"""
import csv
import glob
import json
import math

from wsp.redesign_e2 import KIND   # noqa: F401  (also runs the E2 analysis)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


CONDS = ["C1 prompt", "C2 none", "C3 prefix-4", "C4 prefix-16", "C5 edit", "C6 edit + prefix-4"]
out, table = [], {}
for f in sorted(glob.glob("results/E1_*.jsonl")):
    R = [json.loads(l) for l in open(f)]
    task = R[0]["task"]
    for c in CONDS:
        rr = [r for r in R if r["condition"] == c]
        row = dict(task=task, kind=KIND[task], condition=c, n=len(rr), edit_groups=rr[0]["n_edit_groups"])
        for name, fn in (("follows", lambda r: r["follows"]), ("correct", lambda r: r["correct"]), ("follows_and_correct", lambda r: r["follows"] and r["correct"])):
            p, lo, hi = wilson(sum(fn(r) for r in rr), len(rr))
            row.update({name: p, name + "_lo": lo, name + "_hi": hi})
        part = [r for r in rr if not r["prefix_covers_whole_reference"]]
        row["n_prefix_shorter_than_reference"] = len(part) if c.startswith(("C3", "C4")) else ""
        row["follows_where_prefix_is_shorter"] = (sum(r["follows"] for r in part) / len(part) if part else float("nan")) if c.startswith(("C3", "C4")) else ""
        out.append(row)
        table[(task, c)] = row
with open("results/E1_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
order = ["prefix", "whole answer", "end or count", "other"]
tasks = sorted({r["task"] for r in out}, key=lambda t: (order.index(KIND[t]), t))
pc = lambda r, k="follows": f"{100 * r[k]:.0f} [{100 * r[k + '_lo']:.0f}, {100 * r[k + '_hi']:.0f}]"
with open("results/E1_table.md", "w") as f:
    f.write("지시문을 지킨 답의 %와 95% Wilson 구간. 질문 200개.\n\n| 지시문 | 종류 | C1 지시문을 넣음 | C2 아무것도 없음 | C3 첫 4 token 강제 | C4 첫 16 token 강제 | C5 조건부 편집 | C6 편집과 첫 4 token 강제 |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for t in tasks:
        f.write(f"| {t} | {KIND[t]} | " + " | ".join(pc(table[(t, c)]) for c in CONDS) + " |\n")
    f.write("\n정답 문자열이 든 답의 %\n\n| 지시문 | C1 | C2 | C3 | C4 | C5 | C6 |\n| --- | --- | --- | --- | --- | --- | --- |\n")
    for t in tasks:
        f.write(f"| {t} | " + " | ".join(f"{100 * table[(t, c)]['correct']:.0f}" for c in CONDS) + " |\n")
    f.write("\n지시문도 지키고 정답도 든 답의 %\n\n| 지시문 | C1 | C2 | C3 | C4 | C5 | C6 |\n| --- | --- | --- | --- | --- | --- | --- |\n")
    for t in tasks:
        f.write(f"| {t} | " + " | ".join(f"{100 * table[(t, c)]['follows_and_correct']:.0f}" for c in CONDS) + " |\n")
    f.write("\n강제한 앞머리가 참조 답보다 짧은 문항만 따로 본 준수율. 앞머리가 참조 답 전체를 덮으면 지시문 있는 모델의 답을 대신 써 준 것과 같다.\n\n"
            "| 지시문 | C3: 해당 문항 수 | C3: 준수 % | C4: 해당 문항 수 | C4: 준수 % | C5 조건부 편집 % |\n| --- | --- | --- | --- | --- | --- |\n")
    for t in tasks:
        a, b = table[(t, "C3 prefix-4")], table[(t, "C4 prefix-16")]
        g = lambda r: "-" if r["n_prefix_shorter_than_reference"] == 0 else f"{100 * r['follows_where_prefix_is_shorter']:.0f}"
        f.write(f"| {t} | {a['n_prefix_shorter_than_reference']} | {g(a)} | {b['n_prefix_shorter_than_reference']} | {g(b)} | {100 * table[(t, 'C5 edit')]['follows']:.0f} |\n")
print(open("results/E1_table.md").read())
