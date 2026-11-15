# 실험 4의 일부. 일반 글, 긴 문서, 되돌리기

2026-11-16 새벽(KST)에 Qwen3.5-0.8B와 JSON 편집 둘로 돌렸다. 코드는 `wsp/exp4_distance.py`, 결과는 `distance/summary.json`이다.

- **일반 글.** Wikipedia 글 100,352 token에서 편집의 v와 h의 곱이 0.5를 넘은 자리는 없다. 가장 큰 값이 0.11이다. 원래 모델과의 평균 KL은 0.020 nat이고 글에 매기는 loss는 token당 0.034 nat 늘었다.
- **긴 문서 뒤.** 질문 100개 앞에 상관없는 문서를 끼웠다. 편집의 유효한 JSON 비율은 문서가 없으면 97%, 1,000 token이면 83%, 4,000 token이면 85%, 16,000 token이면 93%다. 지시문을 넣은 쪽은 99%에서 100%다. `PLAN.md`의 판정 기준은 16,000 token에서 편집이 10%p 이상 높은 것이었고, 넘지 못했다.
- **되돌리기.** 편집을 다시 빼도 원소 720만 개가 다르게 남는다. `results/REPORT.md`의 E0에서 반올림 때문임을 확인했다.

`distance/log_first_run_out_of_memory.txt`와 `log_second_run_documents_mostly_empty.txt`는 버린 실행이다. 두 번째 실행은 문서 위치 계산이 틀려서 질문 대부분에 빈 문서가 들어갔다.

실험 4의 나머지인 만드는 비용 표와 지시문 둘 합치기는 하지 않았다. 훈련을 쓰는 방법과의 비교는 `results/REPORT.md`의 E3에 있다.
