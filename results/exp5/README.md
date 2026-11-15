# 지시문 13개에 같은 절차를 돌린다

2026-11-16 새벽(KST)에 돌렸다. `PLAN.md`의 실험 5에 해당하고, 편집을 쌓는 절차는 `wsp/onpolicy.py`다.

- 지시문과 통과 기준은 `wsp/instructions.py`에 있다. 열 개는 IFEval의 문구와 기준이고 세 개는 우리가 정했다.
- 폴더는 `<모델>/<지시문>/`이다. 그 안에 조건별 출력 원문 `outputs_*.jsonl`, 편집 `edits.npz`, 요약 `summary.json`이 있다.
- Qwen3.5-0.8B의 표는 `qwen35-0.8b/table.md`에 있다. 만드는 명령은 `python -m wsp.report_suite qwen35-0.8b --md results/exp5/qwen35-0.8b/table.md`다.
- 모든 자리에 더하는 steering을 "지시문도 지키고 정답도 든 비율"로 다시 조정한 결과는 지시문 폴더마다 `baseline_chosen_for_follows_and_correct.json`에 있다. 만든 스크립트는 `wsp/suite_fair_baseline.py`다.
- `first_attempt_lowercase/`는 쌓는 규칙을 고치기 전의 실행이다. 둘째 편집이 첫 편집과 같은 종류의 자리를 겨냥해서 준수율이 72%에서 8%로 떨어졌다.
- `qwen35-0.8b_8rounds/`는 정해진 말로 시작하기를 편집 여덟 개까지 쌓은 것이다. 57%에서 멈췄다.
- `qwen35-4b/`는 JSON 지시문 하나만 돌렸다. 편집 셋으로 91.0%다.

결과를 읽은 글은 `results/REPORT.md`와 `docs/draft.md`의 5.2절에 있다. 한 줄로 줄이면, 답의 앞머리로 정해지는 형식은 편집으로 지시문 수준이 나오고, 답의 끝에서 정해지는 형식은 나오지 않는다.

돌린 명령은 `scripts/run_suite_worker.sh <GPU> <모델> <지시문 ...>`이다.
