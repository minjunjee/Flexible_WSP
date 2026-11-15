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

## 읽는 법

**같은 문장으로 학습한 LoRA는 본 적 없는 episode에서도 다 풀었다.** Gemma 3 4B의 test split에서 LoRA는 두 설정 모두 192문제를 다 풀고 통제 문항도 전부 지켰다. 편집은 같은 episode 셋에서 74문제였다. 그러니 test split에서 편집이 무너진 것을 문장 376개가 그 양식을 덮지 못한 탓으로 돌릴 수 없다. 같은 문장에서 LoRA는 일반화했고 편집은 하지 못했다.

**LoRA는 설정에 따라 결과가 갈린다.** development split에서 47 step LoRA는 Gemma 3 4B에서 157문제, Qwen3.5-0.8B에서 0문제다. 141 step으로 늘리면 두 모델 모두 192문제다. 다만 Qwen3.5-0.8B의 141 step LoRA는 통제 문항 유지가 0.68에서 0.75까지 떨어진 episode가 셋 있다. 편집은 Gemma 3 4B의 development split에서 192문제이고 통제 문항 유지가 0.93에서 0.98이다.

**데이터 양을 맞춰도 LoRA가 높다.** 편집과 같은 질문 64개만 쓴 LoRA는 40 step이면 JSON 100%, 소문자 96%다. 편집은 98%와 86%다. 8 step으로는 0%와 18%라서, 지시서의 설정인 lr 1e-4에 1 epoch을 64쌍에 그대로 쓰면 배우지 못한다.

**계산 시간도 LoRA가 짧다.** Gemma 3 4B에서 LoRA 학습은 10초에서 100초이고 편집은 episode 하나에 20분에서 30분이다.

그래서 이 결과가 말해 주는 것은 이렇다. gradient를 쓸 수 있으면 이 과제에는 LoRA가 맞다. 조건부 편집이 뜻을 갖는 것은 IWIBench처럼 gradient와 optimiser를 주지 않는 설정 안에서이고, 거기서도 본 적 없는 기록 양식과 token 여러 개짜리 key에서는 무너진다.

