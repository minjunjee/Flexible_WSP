"""Assemble results/REPORT.md for the redesign sheet from the result files. Run again whenever a pending run finishes.

    python -m wsp.redesign_report
"""
import csv
import glob
import json
import os
import re

rd = lambda p: list(csv.DictReader(open(p)))
E1 = {(r["task"], r["condition"]): r for r in rd("results/E1_summary.csv")}
E2 = rd("results/E2_summary.csv")
SP = json.load(open("results/E2_spearman.json"))
E0 = rd("results/E0_summary.csv")
E3 = rd("results/E3_summary.csv")
LK = json.load(open("results/E4/E4a_leak_check.json"))
KO = {"json_answer": "JSON, key는 answer", "json_format": "JSON으로 감싸기", "title": "제목 붙이기", "start_phrase": "정해진 말로 시작",
      "lowercase": "소문자", "uppercase": "대문자", "french": "프랑스어", "no_comma": "쉼표 쓰지 않기", "end_phrase": "정해진 말로 끝내기",
      "postscript": "추신 붙이기", "three_bullets": "bullet 세 개", "quotation": "따옴표로 감싸기", "three_words": "세 단어 이하"}
KIND = {"prefix": "앞머리", "whole answer": "답 전체", "end or count": "끝·개수", "other": "기타"}
ORDER = ["json_answer", "json_format", "title", "start_phrase", "lowercase", "uppercase", "french", "no_comma", "end_phrase", "postscript", "three_bullets", "quotation", "three_words"]
CONDS = ["C1 prompt", "C2 none", "C3 prefix-4", "C4 prefix-16", "C5 edit"]
f1 = lambda r, k: f"{100 * float(r[k]):.0f}"
ci = lambda r, k="follows": f"{100 * float(r[k]):.0f} [{100 * float(r[k + '_lo']):.0f}, {100 * float(r[k + '_hi']):.0f}]"


def iwi(d):
    p = f"results/iwibench/{d}/summary.json" if not d.startswith("results/") else f"{d}/summary.json"
    if not os.path.exists(p):
        return None
    eps = json.load(open(p))["episodes"]
    return [(e["seed"], e["edited"]["target_successes"], e["edited"]["target_count"], e["edited"]["control_retention"], e["edited"]["reward"], e["edits"]) for e in eps]


def iwi_cell(d):
    x = iwi(d)
    if not x:
        return "도는 중", "도는 중"
    tot = f"{sum(a[1] for a in x)} / {sum(a[2] for a in x)}"
    per = ", ".join(f"{a[1]}/{a[2]} ({a[4]:.3f})" for a in x)
    return per, f"{tot}, 평균 점수 {sum(a[4] for a in x) / len(x):.3f}" + ("" if len(x) == 3 else f" (episode {len(x)}개만 끝남)")


L = []
w = L.append
w("# WSP 재설계 실험 보고 (지시서 E0~E4)\n")
w("2026-11-16 새벽(KST)에 돌렸다. 대상 모델은 Qwen3.5-0.8B, 생각 과정은 끔, 생성은 greedy에 새 token 96개까지다. 구간은 95% Wilson 구간이다. 표를 만드는 스크립트는 `wsp/redesign_report.py`이고, 도는 중인 실행이 끝나면 다시 돌려 채운다.\n")
w("## 판정 요약\n")
w("| 주장 | 지시서의 기준 | 나온 값 | 판정 |\n| --- | --- | --- | --- |")
lo, up, fr = (E1[(t, "C5 edit")] for t in ("lowercase", "uppercase", "french"))
d = {t: 100 * (float(E1[(t, "C5 edit")]["follows"]) - float(E1[(t, "C4 prefix-16")]["follows"])) for t in ("lowercase", "uppercase", "french")}
w(f"| A. 답 전체형 지시문에서 편집이 앞머리 강제를 넘는다 | 소문자, 대문자, 프랑스어 중 둘 이상에서 C5 − C4가 30%p 이상 | 소문자 {d['lowercase']:+.0f}%p, 대문자 {d['uppercase']:+.0f}%p, 프랑스어 {d['french']:+.0f}%p | 서지 않는다. 대문자 하나만 넘었다 |")
w(f"| B. 결정 자리를 재면 편집 성공을 미리 안다 | nD나 dL과 편집 준수율의 Spearman ρ가 −0.6 이하. 끝·개수형 셋이 dL 상위 셋 | nD ρ {SP['nD vs edit compliance']['rho']:.2f} (p {SP['nD vs edit compliance']['p']:.3f}), dL ρ {SP['dL vs edit compliance']['rho']:.2f} (p {SP['dL vs edit compliance']['p']:.3f}). dL 상위 셋은 추신, bullet 세 개, 정해진 말로 끝내기 | 반만 선다. 상위 셋은 맞았고 ρ는 기준에 조금 못 미친다 |")
e4 = iwi("results/E4/E4b_qwen35-0.8b")
w(f"| C. IWIBench 결과가 코드를 안 보고 쓴 문형으로도 유지된다 | episode 1 점수가 0.76 이상 | episode 1에서 {e4[0][1]}/{e4[0][2]}, 점수 {e4[0][4]:.3f} | 서지 않는다. 80개 문형의 0.095보다는 높다 |\n" if e4 else "| C | | 도는 중 | |\n")

w("## 표 1. E1 앞머리 강제 대조군\n")
w("질문은 시험 질문 500개 가운데 유형마다 앞 40개씩 200개다. C1은 지시문을 prompt에 넣은 것, C2는 아무것도 없는 것, C3과 C4는 C1이 쓴 답의 첫 4 token과 첫 16 token을 강제로 넣고 이어 쓰게 한 것, C5는 지시문 없이 조건부 편집만 넣은 것이다. 값은 지시문을 지킨 답의 %다.\n")
w("| 지시문 | 종류 | C1 | C2 | C3 | C4 | C5 | 편집 수 |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
for t in ORDER:
    w(f"| {KO[t]} | {KIND[E1[(t, 'C1 prompt')]['kind']]} | " + " | ".join(ci(E1[(t, c)]) for c in CONDS) + f" | {E1[(t, 'C5 edit')]['edit_groups']} |")
w("\n정답 문자열이 든 답의 %\n\n| 지시문 | C1 | C2 | C3 | C4 | C5 |\n| --- | --- | --- | --- | --- | --- |")
for t in ORDER:
    w(f"| {KO[t]} | " + " | ".join(f1(E1[(t, c)], "correct") for c in CONDS) + " |")
w("\n지시문도 지키고 정답도 든 답의 %\n\n| 지시문 | C1 | C2 | C3 | C4 | C5 | C6 편집과 첫 4 token |\n| --- | --- | --- | --- | --- | --- | --- |")
for t in ORDER:
    w(f"| {KO[t]} | " + " | ".join(f1(E1[(t, c)], "follows_and_correct") for c in CONDS + ["C6 edit + prefix-4"]) + " |")
w("\n강제한 앞머리가 참조 답 전체를 덮는 문항이 많다. 그런 문항에서는 C1의 답을 대신 써 준 것과 같아서 지키는 것이 당연하다. 앞머리가 참조 답보다 짧은 문항만 보면 이렇다.\n")
w("| 지시문 | C3에서 해당 문항 수 | C3 준수 % | C4에서 해당 문항 수 | C4 준수 % | C5 준수 % (200문항) |\n| --- | --- | --- | --- | --- | --- |")
for t in ORDER:
    a, b = E1[(t, "C3 prefix-4")], E1[(t, "C4 prefix-16")]
    g = lambda r: "-" if r["n_prefix_shorter_than_reference"] in ("0", "") else f"{100 * float(r['follows_where_prefix_is_shorter']):.0f}"
    w(f"| {KO[t]} | {a['n_prefix_shorter_than_reference']} | {g(a)} | {b['n_prefix_shorter_than_reference']} | {g(b)} | {f1(E1[(t, 'C5 edit')], 'follows')} |")
w("")
TEXT_A = open("results/REPORT_text_A.md").read() if os.path.exists("results/REPORT_text_A.md") else ""
w(TEXT_A)

w("## 표 2. E2 결정 자리 측정\n")
w("d1은 C1의 답과 C2의 답이 처음 달라지는 token 위치를 C1 답의 길이로 나눈 값이다. nD는 C1의 답을 지시문 없는 모델에 넣었을 때, 그 모델이 고를 token이 C1의 token과 다른 자리의 수다. dL은 그런 자리 가운데 마지막 자리의 상대 위치다. 셋 다 질문 200개의 평균이고 편집을 만들기 전의 원래 모델에서 잰다. nD는 지시서와 달리 200개 전부에서 쟀다.\n")
w("| 지시문 | 종류 | d1 | nD | dL | 참조 답의 길이 | 편집 준수율 % | 편집 수 |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
for r in sorted(E2, key=lambda r: ORDER.index(r["task"])):
    w(f"| {KO[r['task']]} | {KIND[r['kind']]} | {float(r['d1']):.2f} | {float(r['nD']):.1f} | {float(r['dL']):.2f} | {float(r['reference_length']):.1f} | {100 * float(r['edit_follows']):.1f} | {r['edit_groups']} |")
w("\n| 견준 것 | Spearman ρ | p | n |\n| --- | --- | --- | --- |")
for k in ("d1", "nD", "dL"):
    for tgt, name in (("edit compliance", "편집 준수율"), ("number of edits", "편집 수")):
        s = SP[f"{k} vs {tgt}"]
        w(f"| {k}와 {name} | {s['rho']:.2f} | {s['p']:.3f} | {s['n']} |")
w("\n![nD와 편집 준수율](E2_scatter.png)\n")
w(open("results/REPORT_text_B.md").read() if os.path.exists("results/REPORT_text_B.md") else "")

w("## 표 3. E4 누수 없는 문형으로 IWIBench\n")
w("과제는 IWIBench의 구조화된 응답이고 문항과 채점기는 그 저장소의 것이다. 대상 모델은 Qwen3.5-0.8B다. 4-gram 겹침은 문형의 4-gram 가운데 채점 문항 192개에도 나오는 것의 비율이다.\n")
w("| 편집을 만든 문형 | episode 1 | 세 episode | 4-gram 겹침 |\n| --- | --- | --- | --- |")
ov = LK["share_of_4grams_also_in_the_benchmark_targets"]
for name, d, key in (("우리가 쓴 80개", "qwen35-0.8b", "first small set (80 prompts)"), ("코드를 본 뒤 쓴 376개", "qwen35-0.8b_wide", "written after reading the code"),
                     ("코드를 본 뒤 쓴 376개에서 채점 문항의 낱말을 뺀 것", "qwen35-0.8b_wide_heldout_wording", "after reading the code, benchmark words left out"),
                     ("코드를 안 보고 쓴 376개 (E4a)", "results/E4/E4b_qwen35-0.8b", "written without seeing the code")):
    x = iwi(d)
    first = f"{x[0][1]}/{x[0][2]}, 점수 {x[0][4]:.3f}" if x else "도는 중"
    w(f"| {name} | {first} | {iwi_cell(d)[1]} | {100 * ov[key]:.2f}% |")
w("")
w(open("results/REPORT_text_C.md").read() if os.path.exists("results/REPORT_text_C.md") else "")

w("## E0. 편집을 빼면 원래 가중치로 돌아가는가\n")
w("JSON 지시문의 편집 둘을 `down_proj` 행렬 24개에 더한 뒤 같은 순서로 뺐다. 원소는 모두 88,080,384개다.\n")
w("| 수 형식 | 원래와 다른 원소 수 | 가장 큰 차이 | 일반 글 10,240 token에서 원래 모델과의 평균 KL |\n| --- | --- | --- | --- |")
for r in E0:
    w(f"| {r['format']} | {int(r['weights_different_after_add_then_subtract']):,} ({100 * float(r['share']):.1f}%) | {float(r['largest_absolute_difference']):.1e} | {float(r['mean_kl_to_original_on_10240_tokens']):.1e} |")
w("")
w(open("results/REPORT_text_E0.md").read() if os.path.exists("results/REPORT_text_E0.md") else "")

w("## E3. LoRA 비교\n")
w("| 지시문 | 방법 | 준수율 % | 정답 % | 둘 다 % | 일반 글에서의 평균 KL | 만드는 데 든 시간 |\n| --- | --- | --- | --- | --- | --- | --- |")
for r in E3:
    name = "LoRA rank 4, " + ("지시서 설정: lr 1e-4, 1 epoch, 38 step" if "sheet" in r["setting"] else "우리가 더한 설정: lr 1e-3, 5 epoch, 190 step")
    w(f"| {KO[r['task']]} | {name} | {ci(r)} | {100 * float(r['correct']):.1f} | {100 * float(r['follows_and_correct']):.1f} | {float(r['mean_kl_to_original_on_10240_tokens']):.4f} | 학습 {r['train_seconds']}초 |")
    if "longer" in r["setting"] or "added" in r["setting"]:
        e = E1[(r["task"], "C5 edit")]
        w(f"| {KO[r['task']]} | 조건부 편집 {e['edit_groups']}개, 훈련 없음 | {ci(e)} | {f1(e, 'correct')} | {f1(e, 'follows_and_correct')} | {'0.0201 (10만 token에서 잰 값)' if r['task'] == 'json_answer' else '재지 않음'} | forward pass만, 편집 하나에 1분쯤 |")
w("")
w(open("results/REPORT_text_E3.md").read() if os.path.exists("results/REPORT_text_E3.md") else "")
# ---------------------------------------------------------------- earlier runs, latest numbers
w("## 먼저 돌리던 실행의 최신 숫자\n")
w("### 모든 자리에 더하는 steering을 다시 조정한 것\n")
w("layer와 배율을 조정용 질문 40개에서 \"지시문도 지키고 정답도 든 비율\"로 골랐다. 처음에는 지시문을 지킨 비율만 보고 골라서, 형식만 맞고 답이 망가진 설정이 뽑힌 지시문이 있었다. 시험 질문 500개에서 잰 값이고, 칸마다 지킨 % / 정답 % / 둘 다 %다.\n")
w("| 지시문 | 지시문을 넣음 | 모든 자리에 더함, 다시 조정 | 조건부 편집, 쌓은 것 전부 |\n| --- | --- | --- | --- |")
tri = lambda x: f"{100 * x['follows'][0]:.0f} / {100 * x['correct'][0]:.0f} / {100 * x['follows_and_correct'][0]:.0f}"
for t in ORDER:
    sp, bp = f"results/exp5/qwen35-0.8b/{t}/summary.json", f"results/exp5/qwen35-0.8b/{t}/baseline_chosen_for_follows_and_correct.json"
    if not (os.path.exists(sp) and os.path.exists(bp)):
        continue
    C = json.load(open(sp))["conditions"]
    ed = [v for k, v in C.items() if k.startswith("conditional edits")]
    w(f"| {KO[t]} | {tri(C['instruction in the prompt'])} | {tri(json.load(open(bp)))} | {tri(ed[-1]) if ed else '편집 없음'} |")
w("\n### IWIBench의 다른 실행\n")
w("| 모델 | 편집을 만든 문형 | episode별 목표 문항과 점수 | 합계 |\n| --- | --- | --- | --- |")
for model, name, d in (("Qwen3.5-0.8B", "코드를 본 뒤 쓴 376개", "qwen35-0.8b_wide"), ("Qwen3.5-0.8B", "같은 문형에서 채점 문항의 낱말을 뺀 것", "qwen35-0.8b_wide_heldout_wording"),
                       ("Qwen3.5-0.8B", "낱말을 뺀 문형, 벤치의 test split episode", "qwen35-0.8b_wide_test_split"),
                       ("Qwen3.5-4B", "코드를 본 뒤 쓴 376개, episode 1만", "qwen35-4b_wide"),
                       ("Gemma 3 4B", "코드를 본 뒤 쓴 376개", "gemma3-4b_wide"), ("Gemma 3 4B", "코드를 안 보고 쓴 376개", "gemma3-4b_prompts_written_without_seeing_the_code")):
    per, tot = iwi_cell(d)
    w(f"| {model} | {name} | {per} | {tot} |")
w("")
w(open("results/REPORT_text_rest.md").read() if os.path.exists("results/REPORT_text_rest.md") else "")
open("results/REPORT.md", "w").write("\n".join(L) + "\n")
print("wrote results/REPORT.md", len("\n".join(L)))
