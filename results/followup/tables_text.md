## 읽는 법

**같은 문장으로 학습한 LoRA는 본 적 없는 episode에서도 다 풀었다.** Gemma 3 4B의 test split에서 LoRA는 두 설정 모두 192문제를 다 풀고 통제 문항도 전부 지켰다. 편집은 같은 episode 셋에서 74문제였다. 그러니 test split에서 편집이 무너진 것을 문장 376개가 그 양식을 덮지 못한 탓으로 돌릴 수 없다. 같은 문장에서 LoRA는 일반화했고 편집은 하지 못했다.

**LoRA는 설정에 따라 결과가 갈린다.** development split에서 47 step LoRA는 Gemma 3 4B에서 157문제, Qwen3.5-0.8B에서 0문제다. 141 step으로 늘리면 두 모델 모두 192문제다. 다만 Qwen3.5-0.8B의 141 step LoRA는 통제 문항 유지가 0.68에서 0.75까지 떨어진 episode가 셋 있다. 편집은 Gemma 3 4B의 development split에서 192문제이고 통제 문항 유지가 0.93에서 0.98이다.

**데이터 양을 맞춰도 LoRA가 높다.** 편집과 같은 질문 64개만 쓴 LoRA는 40 step이면 JSON 100%, 소문자 96%다. 편집은 98%와 86%다. 8 step으로는 0%와 18%라서, 지시서의 설정인 lr 1e-4에 1 epoch을 64쌍에 그대로 쓰면 배우지 못한다.

**계산 시간도 LoRA가 짧다.** Gemma 3 4B에서 LoRA 학습은 10초에서 100초이고 편집은 episode 하나에 20분에서 30분이다.

그래서 이 결과가 말해 주는 것은 이렇다. gradient를 쓸 수 있으면 이 과제에는 LoRA가 맞다. 조건부 편집이 뜻을 갖는 것은 IWIBench처럼 gradient와 optimiser를 주지 않는 설정 안에서이고, 거기서도 본 적 없는 기록 양식과 token 여러 개짜리 key에서는 무너진다.
