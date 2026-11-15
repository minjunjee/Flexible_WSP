"""Tables for the follow-up runs of 2026-11-16 (after 06:10 KST): the LoRA references.

    python -m wsp.followup_report        # writes results/followup/tables.md
"""
import csv
import glob
import json
import os

from wsp.iwibench_breakdown import breakdown

L = []
w = L.append


def episodes(folder):
    files = sorted(glob.glob(f"{folder}/summary.json") + glob.glob(f"{folder}/seed*/summary.json"))
    eps = [e for f in files for e in json.load(open(f))["episodes"] if "edited" in e]
    return sorted(eps, key=lambda e: e["seed"])


def cell(folder):
    eps = episodes(folder)
    if not eps:
        return "없음", "없음", "없음"
    per = ", ".join(f"{e['edited']['target_successes']}/{e['edited']['target_count']}" for e in eps)
    tot = f"{sum(e['edited']['target_successes'] for e in eps)} / {sum(e['edited']['target_count'] for e in eps)}"
    return f"{tot} ({per})", f"{sum(e['edited']['reward'] for e in eps) / len(eps):.3f}", ", ".join(f"{e['edited']['control_retention']:.2f}" for e in eps)


w("# 후속 실험: LoRA 기준선 (2026-11-16 06:10 KST 이후)\n")
w("표를 만드는 스크립트는 `wsp/followup_report.py`다. LoRA는 모두 rank 4이고 편집이 바꾸는 행렬과 같은 `down_proj`에 건다. 가중치에 합치지 않고 hook으로 적용해서 쟀다.\n")

w("## 표 1. 편집과 같은 질문 64개로 학습한 LoRA\n")
w("Qwen3.5-0.8B, 시험 질문 200개. E3에서는 LoRA를 300쌍으로 학습했고 편집은 질문 64개로 만들었다. 여기서는 LoRA도 그 64개만 쓴다. 입력은 지시문 없는 질문이고 목표는 지시문 있는 모델의 답이다. 코드는 `wsp/followup_lora_same_data.py`다.\n")
w("| 지시문 | 방법 | 학습 step | 지시문을 지킴 | 정답 | 일반 글에서의 KL | 학습 시간 |\n| --- | --- | --- | --- | --- | --- | --- |")
KO = {"json_answer": "JSON, key는 answer", "lowercase": "소문자"}
EDIT = {"json_answer": ("조건부 편집 둘", "98 [94, 99]", "74", "0.020"), "lowercase": ("조건부 편집 셋", "86 [81, 90]", "52", "재지 않음")}
if os.path.exists("results/followup/lora_same_64_questions.csv"):
    rows = list(csv.DictReader(open("results/followup/lora_same_64_questions.csv")))
    for task in ("json_answer", "lowercase"):
        for r in [r for r in rows if r["task"] == task]:
            w(f"| {KO[task]} | LoRA, 64쌍, {r['setting']} | {r['steps']} | {100 * float(r['follows']):.0f} [{100 * float(r['follows_lo']):.0f}, {100 * float(r['follows_hi']):.0f}] | "
              f"{100 * float(r['correct']):.1f} | {float(r['mean_kl_to_original_on_10240_tokens']):.4f} | {r['train_seconds']}초 |")
        e = EDIT[task]
        w(f"| {KO[task]} | {e[0]}, 질문 64개 | 학습 없음 | {e[1]} | {e[2]} | {e[3]} | |")
w("")

w("## 표 2. IWIBench의 구조화된 응답 과제에서 같은 문장으로 학습한 LoRA\n")
w("편집을 만들 때 쓴 문장 376개를 그대로 쓴다. 벤치 코드를 보지 않은 다른 모델이 쓴 문장이다. 이름을 뽑는 문장에서는 지시문 있는 모델의 답을, 나머지 문장에서는 원래 모델의 답을 목표로 학습한다. 편집이 따르는 대상과 같다. 벤치는 editor에게 gradient를 주지 않으므로 LoRA는 벤치의 규칙 밖에 있는 참고용이다. 코드는 `wsp/iwibench_structured.py --lora`다.\n")
w("| 모델 | split | 방법 | 목표 문항 | 평균 점수 | 통제 문항 유지 |\n| --- | --- | --- | --- | --- | --- |")
ROWS = [
    ("Gemma 3 4B", "development", "조건부 편집, 겨냥하는 자리를 제한한 절차", "results/iwibench/gemma3-4b_prompts_written_without_seeing_the_code"),
    ("Gemma 3 4B", "development", "LoRA, lr 1e-4, 1 epoch (47 step)", "results/iwibench_lora/gemma3-4b_development_lr1e-4_epochs1"),
    ("Gemma 3 4B", "development", "LoRA, lr 1e-3, 3 epoch (141 step)", "results/iwibench_lora/gemma3-4b_development_lr1e-3_epochs3"),
    ("Gemma 3 4B", "test", "조건부 편집", "results/iwibench/gemma3-4b_episodes_first_run_after_freezing_the_procedure"),
    ("Gemma 3 4B", "test", "LoRA, lr 1e-4, 1 epoch (47 step)", "results/iwibench_lora/gemma3-4b_test_lr1e-4_epochs1"),
    ("Gemma 3 4B", "test", "LoRA, lr 1e-3, 3 epoch (141 step)", "results/iwibench_lora/gemma3-4b_test_lr1e-3_epochs3"),
    ("Qwen3.5-0.8B", "development", "조건부 편집", "results/E4/E4b_qwen35-0.8b"),
    ("Qwen3.5-0.8B", "development", "LoRA, lr 1e-4, 1 epoch (47 step)", "results/iwibench_lora/qwen35-0.8b_development_lr1e-4_epochs1"),
    ("Qwen3.5-0.8B", "development", "LoRA, lr 1e-3, 3 epoch (141 step)", "results/iwibench_lora/qwen35-0.8b_development_lr1e-3_epochs3"),
    ("Qwen3.5-0.8B", "test", "LoRA, lr 1e-4, 1 epoch (47 step)", "results/iwibench_lora/qwen35-0.8b_test_lr1e-4_epochs1"),
    ("Qwen3.5-0.8B", "test", "LoRA, lr 1e-3, 3 epoch (141 step)", "results/iwibench_lora/qwen35-0.8b_test_lr1e-3_epochs3"),
]
for model, split, method, folder in ROWS:
    t, m, r = cell(folder)
    w(f"| {model} | {split} | {method} | {t} | {m} | {r} |")
w("")
w("LoRA의 학습 시간은 episode마다 Gemma 3 4B에서 47 step에 10초쯤, 141 step에 50초에서 100초다. 편집은 Gemma 3 4B에서 episode 하나에 20분에서 30분이 걸렸다. 편집 하나를 만들 때 layer 34개마다 10,240차원의 선형 방정식을 풀기 때문이다.\n")

w("### 산문으로 답하라는 문항\n")
w("벤치의 점수는 원래 모델이 맞힌 통제 문항만 분모에 넣는다. 원래 Gemma 3 4B는 산문으로 답하라는 문항 8개의 채점 기준을 하나도 통과하지 못하므로 그 문항은 점수에 들어가지 않는다. 그 문항에 JSON으로 답한 수를 따로 센다.\n")
w("| 실행 | episode별, 8개 중 JSON으로 답한 수 |\n| --- | --- |")
for model, split, method, folder in ROWS:
    b = breakdown(folder)
    if b:
        w(f"| {model}, {split}, {method} | {', '.join(str(x['prose_requests_answered_with_json']) for x in b)} |")
w("")
w("## 표 3. test split에서 틀린 문항에서 편집은 켜졌는가\n")
w("Gemma 3 4B, test split의 첫 episode(key는 `subject`). 편집된 모델로 목표 문항 64개에 다시 답하게 하고, 답의 자리마다 각 편집의 read vector 값을 쟀다. 자리 0은 답 시작 자리이고 자리 1은 key 이름을 쓰는 자리다. 코드는 `wsp/followup_gate_diagnostic.py`다.\n")
if os.path.exists("results/followup/gates_test_seed0.json"):
    import numpy as np
    G = json.load(open("results/followup/gates_test_seed0.json"))
    t = [o for o in G if o["kind"] == "target"]
    grp = {"key를 `subject`로 쓴 문항": [o for o in t if o["text"].startswith('{"subject"')], "key를 다른 낱말로 쓴 문항": [o for o in t if not o["text"].startswith('{"subject"')],
           "일반 질문": [o for o in G if o["kind"] == "general"], "산문으로 답하라는 문항": [o for o in G if o["kind"] == "override"]}
    w("| 문항 | 수 | 첫 편집의 read vector, 자리 0 | 둘째 편집의 read vector, 자리 1 |\n| --- | --- | --- | --- |")
    for name, g in grp.items():
        v1 = np.mean([o["gates_first_positions"][0][0] for o in g])
        v2 = np.mean([o["gates_first_positions"][1][1] for o in g if len(o["gates_first_positions"][1]) > 1])
        w(f"| {name} | {len(g)} | {v1:.2f} | {v2:.2f} |")
    w("")
w("## 표 4. key 이름을 쓰는 편집을 키우면 고쳐지는가\n")
w("같은 실행의 편집을 다시 넣되 둘째 편집에만 배수를 곱하고 episode를 다시 채점했다. 채점 문항으로 배수를 들여다본 것이므로 실패의 원인을 보는 진단이고, 배수를 고르는 방법이 아니다. 코드는 `wsp/followup_scale_key_edit.py`다.\n")
w("| episode | 배수 | 목표 문항 | 통제 문항 유지 | 첫 목표 문항의 답 |\n| --- | --- | --- | --- | --- |")
for name, f in (("test split, seed 0, key는 `subject`", "results/followup/scale_key_edit_test_seed0.json"), ("test split, seed 6, key는 `profile_name`", "results/followup/scale_key_edit_test_seed6.json")):
    if os.path.exists(f):
        for k, v in json.load(open(f))["factors"].items():
            w(f"| {name} | {k} | {v['targets']}/{v['n_targets']} | {v['retention']:.2f} | `{v['examples'][1][:44]}` |")
w("")
w("## 표 5. key 이름을 쓰는 자리에서 편집이 logit을 얼마나 옮기는가\n")
w("같은 episode에서 편집된 모델이 key를 `resident`로 쓴 문항 16개와 `subject`로 쓴 문항 16개. `{\"` 바로 뒤의 자리에서 `subject`의 logit에서 `resident`의 logit을 뺀 값이다. 0보다 커야 `subject`가 이긴다. 배수는 둘째 편집에만 곱했다. 코드는 `wsp/followup_write_ceiling.py`다.\n")
if os.path.exists("results/followup/write_ceiling_test_seed0.json"):
    C = json.load(open("results/followup/write_ceiling_test_seed0.json"))
    names = list(C)
    w("| 둘째 편집의 배수 | key를 `resident`로 쓴 문항 | key를 `subject`로 쓴 문항 | MLP들이 residual에 더한 값의 길이 |\n| --- | --- | --- | --- |")
    for f in C[names[0]]:
        w(f"| {f} | {C[names[0]][f]['mean_logit_gap_asked_minus_other']:+.1f} | {C[names[1]][f]['mean_logit_gap_asked_minus_other']:+.1f} | {C[names[0]][f]['mean_length_of_what_the_mlps_add']:,.0f} |")
w("")
if os.path.exists("results/followup/tables_text.md"):
    w(open("results/followup/tables_text.md").read())
os.makedirs("results/followup", exist_ok=True)
open("results/followup/tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
