"""Print one table for all instructions of a model from results/exp5/<model>/<task>/summary.json.

    python -m wsp.report_suite qwen35-0.8b [--md results/exp5/qwen35-0.8b/table.md]
"""
import glob
import json
import sys

model = sys.argv[1]
md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
rows = []
for f in sorted(glob.glob(f"results/exp5/{model}/*/summary.json")):
    R = json.load(open(f))
    C = R["conditions"]
    pick = lambda pre: next((v for k, v in C.items() if k.startswith(pre)), None)
    ed = [(k, v) for k, v in C.items() if k.startswith("conditional edits")]
    best = max(ed, key=lambda kv: kv[1]["follows"][0]) if ed else None
    b = R.get("before_any_edit", {})
    rows.append(dict(task=R["task"], done="own_answers_after_last_round" in R, prompt=pick("instruction in"), none=pick("no edit"),
                     allpos=pick("difference added at every"), start=pick("differences added at the answer start"),
                     edits=[v for _, v in ed], best=best[1] if best else None, n_best=(ed.index(best) + 1) if best else 0, before=b,
                     allpos_choice=R.get("all_position_baseline_choice")))
pc = lambda x, k="follows": "-" if x is None else f"{100 * x[k][0]:.1f}"
ci = lambda x, k="follows": "-" if x is None else f"{100 * x[k][0]:.1f} [{100 * x[k][1]:.1f}, {100 * x[k][2]:.1f}]"
print(f"{'task':14s} {'prompt':>6s} {'none':>5s} {'allpos':>6s} {'start':>6s} | {'after 1/2/3/4 edits':28s} | {'correct: prompt/none/allpos/best':32s} | div% 1st@start group shared")
for r in rows:
    b = r["before"]
    print(f"{r['task']:14s} {pc(r['prompt']):>6s} {pc(r['none']):>5s} {pc(r['allpos']):>6s} {pc(r['start']):>6s} | "
          f"{' '.join(pc(e) for e in r['edits']):28s} | {pc(r['prompt'], 'correct')}/{pc(r['none'], 'correct')}/{pc(r['allpos'], 'correct')}/{pc(r['best'], 'correct'):12s} | "
          + (f"{100 * b['own_answers']['share']:.0f}% {100 * (b['first_divergence_at_answer_start'] or 0):.0f}% {100 * b['largest_group_share']:.0f}% {b['largest_group_shared_fraction'] or 0:.2f}" if b else "")
          + ("" if r["done"] else "  [running]"))
if md:
    with open(md, "w") as f:
        f.write("시험 질문 500개, greedy. 값은 지시문을 지킨 답의 %이고 대괄호는 질문을 다시 뽑는 bootstrap으로 구한 95% 구간이다.\n\n")
        f.write("| 지시문 | 지시문을 prompt에 넣음 | 편집 없음 | 차이를 한 layer의 모든 자리에 더함 | 차이를 답 시작 자리에만 더함 | 조건부 편집, 가장 좋은 단계 | 그때 쌓은 편집 수 |\n| --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(f"| {r['task']} | {ci(r['prompt'])} | {ci(r['none'])} | {ci(r['allpos'])} | {ci(r['start'])} | {ci(r['best'])} | {r['n_best'] or '-'} |\n")
        f.write("\n편집을 쌓을 때마다 지시문을 지킨 %와 정답 문자열이 든 %\n\n| 지시문 | 1개 | 2개 | 3개 | 4개 | 지시문을 넣었을 때의 정답 | 편집 없을 때의 정답 |\n| --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            cells = [f"{pc(e)} / {pc(e, 'correct')}" for e in r["edits"]] + ["-"] * (4 - len(r["edits"]))
            f.write(f"| {r['task']} | " + " | ".join(cells) + f" | {pc(r['prompt'], 'correct')} | {pc(r['none'], 'correct')} |\n")
        f.write("\n편집을 만들기 전에 잰 값. 편집용 질문 64개에 모델이 지시문 없이 쓴 답에서 쟀다.\n\n| 지시문 | 갈라진 자리의 비율 | 첫 갈라지는 자리가 답 시작 자리인 질문 | 가장 큰 묶음에 든 질문 | 그 묶음에서 질문이 바뀌어도 같은 부분의 비율 |\n| --- | --- | --- | --- | --- |\n")
        for r in rows:
            b = r["before"]
            if b:
                f.write(f"| {r['task']} | {100 * b['own_answers']['share']:.1f}% | {100 * (b['first_divergence_at_answer_start'] or 0):.0f}% | {100 * b['largest_group_share']:.0f}% | {b['largest_group_shared_fraction'] or 0:.2f} |\n")
