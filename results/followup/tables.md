# 후속 실험: LoRA 기준선 (2026-11-16 06:10 KST 이후)

표를 만드는 스크립트는 `wsp/followup_report.py`다. LoRA는 모두 rank 4이고 편집이 바꾸는 행렬과 같은 `down_proj`에 건다. 가중치에 합치지 않고 hook으로 적용해서 쟀다.

## 표 1. 편집과 같은 질문 64개로 학습한 LoRA

Qwen3.5-0.8B, 시험 질문 200개. E3에서는 LoRA를 300쌍으로 학습했고 편집은 질문 64개로 만들었다. 여기서는 LoRA도 그 64개만 쓴다. 입력은 지시문 없는 질문이고 목표는 지시문 있는 모델의 답이다. 코드는 `wsp/followup_lora_same_data.py`다.

| 지시문 | 방법 | 학습 step | 지시문을 지킴 | 정답 | 일반 글에서의 KL | 학습 시간 |
| --- | --- | --- | --- | --- | --- | --- |
| JSON, key는 answer | LoRA, 64쌍, lr 1e-4, 1 epoch | 8 | 0 [0, 2] | 78.5 | 0.0000 | 0.6초 |
| JSON, key는 answer | LoRA, 64쌍, lr 1e-4, 5 epochs | 40 | 100 [98, 100] | 73.5 | 0.0021 | 4.1초 |
| JSON, key는 answer | LoRA, 64쌍, lr 1e-3, 5 epochs | 40 | 98 [95, 99] | 73.0 | 0.0710 | 9.6초 |
| JSON, key는 answer | 조건부 편집 둘, 질문 64개 | 학습 없음 | 98 [94, 99] | 74 | 0.020 | |
| 소문자 | LoRA, 64쌍, lr 1e-4, 1 epoch | 8 | 18 [14, 24] | 77.0 | 0.0000 | 2.2초 |
| 소문자 | LoRA, 64쌍, lr 1e-4, 5 epochs | 40 | 96 [92, 98] | 52.0 | 0.0061 | 10.9초 |
| 소문자 | LoRA, 64쌍, lr 1e-3, 5 epochs | 40 | 98 [94, 99] | 58.0 | 0.1754 | 11.3초 |
| 소문자 | 조건부 편집 셋, 질문 64개 | 학습 없음 | 86 [81, 90] | 52 | 재지 않음 | |

## 표 2. IWIBench의 구조화된 응답 과제에서 같은 문장으로 학습한 LoRA

편집을 만들 때 쓴 문장 376개를 그대로 쓴다. 벤치 코드를 보지 않은 다른 모델이 쓴 문장이다. 이름을 뽑는 문장에서는 지시문 있는 모델의 답을, 나머지 문장에서는 원래 모델의 답을 목표로 학습한다. 편집이 따르는 대상과 같다. 벤치는 editor에게 gradient를 주지 않으므로 LoRA는 벤치의 규칙 밖에 있는 참고용이다. 코드는 `wsp/iwibench_structured.py --lora`다.

| 모델 | split | 방법 | 목표 문항 | 평균 점수 | 통제 문항 유지 |
| --- | --- | --- | --- | --- | --- |
| Gemma 3 4B | development | 조건부 편집, 겨냥하는 자리를 제한한 절차 | 192 / 192 (64/64, 64/64, 64/64) | 0.958 | 0.93, 0.98, 0.97 |
| Gemma 3 4B | development | LoRA, lr 1e-4, 1 epoch (47 step) | 157 / 192 (55/64, 57/64, 45/64) | 0.818 | 1.00, 1.00, 1.00 |
| Gemma 3 4B | development | LoRA, lr 1e-3, 3 epoch (141 step) | 192 / 192 (64/64, 64/64, 64/64) | 0.993 | 0.98, 1.00, 1.00 |
| Gemma 3 4B | test | 조건부 편집 | 74 / 192 (29/64, 0/64, 45/64) | 0.328 | 0.69, 0.48, 0.96 |
| Gemma 3 4B | test | LoRA, lr 1e-4, 1 epoch (47 step) | 192 / 192 (64/64, 64/64, 64/64) | 1.000 | 1.00, 1.00, 1.00 |
| Gemma 3 4B | test | LoRA, lr 1e-3, 3 epoch (141 step) | 192 / 192 (64/64, 64/64, 64/64) | 1.000 | 1.00, 1.00, 1.00 |
| Qwen3.5-0.8B | development | 조건부 편집 | 100 / 192 (34/64, 4/64, 62/64) | 0.494 | 0.95, 0.93, 0.95 |
| Qwen3.5-0.8B | development | LoRA, lr 1e-4, 1 epoch (47 step) | 0 / 192 (0/64, 0/64, 0/64) | 0.000 | 0.93, 0.98, 0.98 |
| Qwen3.5-0.8B | development | LoRA, lr 1e-3, 3 epoch (141 step) | 192 / 192 (64/64, 64/64, 64/64) | 0.814 | 0.75, 0.73, 0.97 |
| Qwen3.5-0.8B | test | LoRA, lr 1e-4, 1 epoch (47 step) | 192 / 192 (64/64, 64/64, 64/64) | 0.982 | 0.98, 0.98, 0.98 |
| Qwen3.5-0.8B | test | LoRA, lr 1e-3, 3 epoch (141 step) | 192 / 192 (64/64, 64/64, 64/64) | 0.856 | 0.68, 0.98, 0.91 |

LoRA의 학습 시간은 episode마다 Gemma 3 4B에서 47 step에 10초쯤, 141 step에 50초에서 100초다. 편집은 Gemma 3 4B에서 episode 하나에 20분에서 30분이 걸렸다. 편집 하나를 만들 때 layer 34개마다 10,240차원의 선형 방정식을 풀기 때문이다.

### 산문으로 답하라는 문항

벤치의 점수는 원래 모델이 맞힌 통제 문항만 분모에 넣는다. 원래 Gemma 3 4B는 산문으로 답하라는 문항 8개의 채점 기준을 하나도 통과하지 못하므로 그 문항은 점수에 들어가지 않는다. 그 문항에 JSON으로 답한 수를 따로 센다.

| 실행 | episode별, 8개 중 JSON으로 답한 수 |
| --- | --- |
| Gemma 3 4B, development, 조건부 편집, 겨냥하는 자리를 제한한 절차 | 0, 6, 0 |
| Gemma 3 4B, development, LoRA, lr 1e-4, 1 epoch (47 step) | 0, 0, 0 |
| Gemma 3 4B, development, LoRA, lr 1e-3, 3 epoch (141 step) | 0, 0, 0 |
| Gemma 3 4B, test, 조건부 편집 | 0, 0, 0 |
| Gemma 3 4B, test, LoRA, lr 1e-4, 1 epoch (47 step) | 0, 0, 0 |
| Gemma 3 4B, test, LoRA, lr 1e-3, 3 epoch (141 step) | 0, 0, 0 |
| Qwen3.5-0.8B, development, 조건부 편집 | 0, 0, 0 |
| Qwen3.5-0.8B, development, LoRA, lr 1e-4, 1 epoch (47 step) | 0, 0, 0 |
| Qwen3.5-0.8B, development, LoRA, lr 1e-3, 3 epoch (141 step) | 0, 0, 0 |
| Qwen3.5-0.8B, test, LoRA, lr 1e-4, 1 epoch (47 step) | 0, 0, 0 |
| Qwen3.5-0.8B, test, LoRA, lr 1e-3, 3 epoch (141 step) | 0, 0, 0 |

## 표 3. test split에서 틀린 문항에서 편집은 켜졌는가

Gemma 3 4B, test split의 첫 episode(key는 `subject`). 편집된 모델로 목표 문항 64개에 다시 답하게 하고, 답의 자리마다 각 편집의 read vector 값을 쟀다. 자리 0은 답 시작 자리이고 자리 1은 key 이름을 쓰는 자리다. 코드는 `wsp/followup_gate_diagnostic.py`다.

| 문항 | 수 | 첫 편집의 read vector, 자리 0 | 둘째 편집의 read vector, 자리 1 |
| --- | --- | --- | --- |
| key를 `subject`로 쓴 문항 | 38 | 1.03 | 0.96 |
| key를 다른 낱말로 쓴 문항 | 26 | 1.03 | 0.96 |
| 일반 질문 | 64 | 0.50 | 0.27 |
| 산문으로 답하라는 문항 | 8 | 0.30 | 0.06 |

## 표 4. key 이름을 쓰는 편집을 키우면 고쳐지는가

같은 실행의 편집을 다시 넣되 둘째 편집에만 배수를 곱하고 episode를 다시 채점했다. 채점 문항으로 배수를 들여다본 것이므로 실패의 원인을 보는 진단이고, 배수를 고르는 방법이 아니다. 코드는 `wsp/followup_scale_key_edit.py`다.

| episode | 배수 | 목표 문항 | 통제 문항 유지 | 첫 목표 문항의 답 |
| --- | --- | --- | --- | --- |
| test split, seed 0, key는 `subject` | 1.0 | 29/64 | 0.69 | `{"resident": "Greta Rossi"}` |
| test split, seed 0, key는 `subject` | 1.5 | 0/64 | 0.39 | `{"resident": Greta Rossi"}` |
| test split, seed 0, key는 `subject` | 2.0 | 0/64 | 0.14 | `{"resident":subject_":"Greta Rossi"}` |
| test split, seed 0, key는 `subject` | 3.0 | 0/64 | 0.00 | `subjectsubjectsubjectsubjectsubjectsubjectsu` |
| test split, seed 6, key는 `profile_name` | 1.0 | 45/64 | 0.96 | `{"profile_name": "Amira Novak"}` |
| test split, seed 6, key는 `profile_name` | 1.5 | 20/64 | 0.55 | `{"profile_name":Amira Novak"}` |
| test split, seed 6, key는 `profile_name` | 2.0 | 0/64 | 0.22 | `{"profile_name":Amira Novak"}` |
| test split, seed 6, key는 `profile_name` | 3.0 | 0/64 | 0.00 | ````Amprofilenamenamenamenamenamenamenamename` |

## 표 5. key 이름을 쓰는 자리에서 편집이 logit을 얼마나 옮기는가

같은 episode에서 편집된 모델이 key를 `resident`로 쓴 문항 16개와 `subject`로 쓴 문항 16개. `{"` 바로 뒤의 자리에서 `subject`의 logit에서 `resident`의 logit을 뺀 값이다. 0보다 커야 `subject`가 이긴다. 배수는 둘째 편집에만 곱했다. 코드는 `wsp/followup_write_ceiling.py`다.

| 둘째 편집의 배수 | key를 `resident`로 쓴 문항 | key를 `subject`로 쓴 문항 | MLP들이 residual에 더한 값의 길이 |
| --- | --- | --- | --- |
| 0.0 | -8.4 | -8.3 | 15,274 |
| 0.5 | -1.6 | -0.7 | 11,803 |
| 1.0 | -0.9 | +1.1 | 11,134 |
| 1.5 | -4.7 | -1.7 | 11,354 |
| 2.0 | -2.6 | -1.0 | 12,066 |
| 3.0 | +6.3 | +6.6 | 13,044 |

## 읽는 법

**같은 문장으로 학습한 LoRA는 본 적 없는 episode에서도 다 풀었다.** Gemma 3 4B의 test split에서 LoRA는 두 설정 모두 192문제를 다 풀고 통제 문항도 전부 지켰다. 편집은 같은 episode 셋에서 74문제였다. 그러니 test split에서 편집이 무너진 것을 문장 376개가 그 양식을 덮지 못한 탓으로 돌릴 수 없다. 같은 문장에서 LoRA는 일반화했고 편집은 하지 못했다.

**LoRA는 설정에 따라 결과가 갈린다.** development split에서 47 step LoRA는 Gemma 3 4B에서 157문제, Qwen3.5-0.8B에서 0문제다. 141 step으로 늘리면 두 모델 모두 192문제다. 다만 Qwen3.5-0.8B의 141 step LoRA는 통제 문항 유지가 0.68에서 0.75까지 떨어진 episode가 셋 있다. 편집은 Gemma 3 4B의 development split에서 192문제이고 통제 문항 유지가 0.93에서 0.98이다.

**데이터 양을 맞춰도 LoRA가 높다.** 편집과 같은 질문 64개만 쓴 LoRA는 40 step이면 JSON 100%, 소문자 96%다. 편집은 98%와 86%다. 8 step으로는 0%와 18%라서, 지시서의 설정인 lr 1e-4에 1 epoch을 64쌍에 그대로 쓰면 배우지 못한다.

**계산 시간도 LoRA가 짧다.** Gemma 3 4B에서 LoRA 학습은 10초에서 100초이고 편집은 episode 하나에 20분에서 30분이다.

그래서 이 결과가 말해 주는 것은 이렇다. gradient를 쓸 수 있으면 이 과제에는 LoRA가 맞다. 조건부 편집이 뜻을 갖는 것은 IWIBench처럼 gradient와 optimiser를 주지 않는 설정 안에서이고, 거기서도 본 적 없는 기록 양식과 token 여러 개짜리 key에서는 무너진다.

**편집이 test split에서 일반화하지 못한 까닭.** 표 3에서 표 5까지가 그 진단이다.

- 틀린 문항에서도 편집은 켜졌다. key를 `resident`로 쓴 26문항에서 둘째 편집의 read vector는 key를 쓰는 자리에서 평균 0.96이다. 맞은 38문항과 같다. 그러니 실패는 read vector 쪽이 아니다.
- 편집이 만드는 여유가 작다. key를 쓰는 자리에서 편집이 없으면 `subject`가 `resident`보다 8.4 nat 낮다. 편집을 넣으면 맞은 문항에서 +1.1, 틀린 문항에서 −0.9가 된다. 편집은 두 낱말을 거의 같은 높이까지만 끌어올리고, 어느 쪽이 이기는지는 문항마다 1 nat 안팎의 차이로 갈린다. test split의 기록에는 `Resident:`라는 항목 이름이 적혀 있어서 그 낱말이 강한 경쟁자가 된다.
- 편집을 키워서는 고쳐지지 않는다. 배수 1.5와 2에서는 그 차이가 오히려 나빠지고, 배수 3에서 +6.3으로 뒤집히지만 그때는 모델이 망가져서 목표 문항이 0/64다. Gemma 3는 MLP 출력을 정규화한 뒤 residual에 더하므로, 정규화 앞에 더하는 vector를 키워도 residual에 닿는 값의 길이는 늘지 않는다. 표 5의 마지막 열에서 그 길이는 배수와 상관없이 11,000에서 15,000이다. 편집은 MLP가 더하는 값에 얹히는 것이 아니라 그 값의 방향을 바꾼다. 그래서 배수가 "더 세게 미는" 손잡이로 작동하지 않는다.

같은 문장으로 학습한 LoRA는 이 episode를 다 풀었다. LoRA는 모든 자리의 MLP 출력을 문맥에 따라 다르게 고치고, 편집은 자리마다 정해진 vector 하나를 더한다는 것이 차이다.

