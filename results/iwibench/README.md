# IWIBench의 구조화된 응답 과제

2026-11-16 새벽(KST)에 돌렸다. 문항과 채점기는 `github.com/hcltech-robotics/iwibench`(commit `db9475f`)의 것이고, 우리 쪽 코드는 `wsp/iwibench_structured.py`다. 공개된 실험 설정과 같은 episode 셋(seed 0, 3, 6)을 쓴다.

폴더 이름은 모델과 편집을 만든 문장을 뜻한다.

| 폴더 | 편집을 만든 문장 |
| --- | --- |
| `qwen35-0.8b/` | 우리가 쓴 80개, 말투 여섯 가지 |
| `qwen35-0.8b_wide/` | 벤치의 문제 생성 코드를 읽은 뒤 쓴 376개 |
| `qwen35-0.8b_wide_heldout_wording/` | 같은 376개에서 채점 문항의 낱말 Extract, note, Which, profile을 뺀 것 |
| `qwen35-0.8b_wide_test_split/` | 낱말을 뺀 문형으로, 벤치의 test split episode에서 잰 것 |
| `qwen35-4b_wide/` | 코드를 읽은 뒤 쓴 376개, episode 하나 |
| `gemma3-4b_*/` | Gemma 3 4B. `gemma3-4b_first_attempt/NOTE.txt`에 첫 시도가 왜 실패했는지 적었다 |
| `../E4/E4b_qwen35-0.8b/` | 코드를 보지 않은 다른 모델이 쓴 376개 |

폴더마다 episode별 출력 원문 `outputs_seed*.jsonl`, 편집 `edits_seed*.npz`, 우리가 쓴 문장 `our_build_prompts_seed*.json`, 요약 `summary.json`이 있다. 숫자와 해석은 `results/REPORT.md`의 표 3과 `docs/draft.md`의 5.5절에 있다.

점수는 그쪽 채점 방식대로 목표 문항의 성공률에 통제 문항을 지킨 비율을 곱한 것이다. Qwen 모델은 그 벤치가 정한 대상이 아니라서, 그쪽의 calibration 검사를 통과하지 못한 episode가 있다. 그런 episode의 점수는 같은 식으로 우리가 계산했다.
