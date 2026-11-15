# Flexible_WSP

지시문을 훈련 없이 모델 가중치로 옮기는 방법의 PoC를 진행하는 repo다. minjunjee와 daehanwon이 함께 쓴다.

확인하려는 주장은 이것이다. 지시문은 답의 몇 군데 자리에서만 모델이 다음 token에 매기는 확률을 바꾼다. 그 자리에서만 작동하는 가중치 변경을 넣으면, 지시문을 prompt에 넣지 않아도 모델이 지시문을 따른다. 가중치 변경은 MLP의 출력 행렬 `down_proj`에 vector 두 개를 곱해 만든 행렬 `u v^T`를 더하는 rank-1 변경이다. `PLAN.md`에서는 이 변경을 조건부 편집이라고 부른다.

실험 계획, 조건부 편집을 만드는 절차, 실험별 판정 기준은 전부 `PLAN.md`에 있다. 실험을 돌리기 전에 그 문서를 처음부터 끝까지 읽는다.

## 지금 상태

2026-11-15 기준이다.

- 실험 0은 끝났다. Qwen3.5-0.8B를 CPU로 돌린 것이고 코드와 결과가 `poc/`에 있다. `PLAN.md`에 따르면 MLP 행렬 6개에 조건부 편집을 하나씩 넣었을 때, 지시문 없는 질문 30개 중 27개에서 유효한 JSON이 나왔다. 이 서버에서 다시 확인한 값은 아니다.
- 실험 1부터 14까지는 시작하지 않았다. 이 서버에서는 아직 어떤 코드도 돌리지 않았다.

## 폴더

| 경로 | 내용 |
| --- | --- |
| `PLAN.md` | 실험 0부터 14까지의 계획과 판정 기준 |
| `poc/` | 실험 0의 코드와 결과. CPU로 돌린 것을 고치지 않고 그대로 옮겼다. 파일별 설명은 `poc/README.md`에 있다 |
| `wsp/` | 실험 1부터 새로 짜는 코드를 둔다. 지금은 비어 있다 |
| `results/` | 실험마다 폴더를 하나씩 둔다. 폴더마다 `README.md`를 쓰고, `results/SUMMARY.md`에 실험마다 한 문단씩 쌓는다 |
| `requirements.txt` | `poc/` 스크립트가 쓰는 Python package 목록. 버전은 아직 고정하지 않았다 |

## `poc/`를 돌리기 전에

`poc/`의 스크립트는 이 서버에서 아직 돌려 보지 않았다. 지금 상태 그대로는 돌지 않는다. 이유는 세 가지다.

첫째, 스크립트 안에 `/home/claude/poc`라는 경로가 그대로 적혀 있다. `PLAN.md`의 실험 1에서 실제 경로로 바꾸기로 했다.

둘째, 스크립트가 읽는 모델과 자료가 repo에 없다. 아래 경로에 따로 받아야 한다. 받는 명령은 `poc/README.md`의 준비 절에 있다.

| 경로 | 무엇인가 | 누가 읽는가 |
| --- | --- | --- |
| `poc/models/qwen35_08b` | `Qwen/Qwen3.5-0.8B` 모델 | 거의 모든 스크립트. 저장된 결과만 읽는 `anaA.py`와 `anaC.py`는 읽지 않는다 |
| `poc/ext/lm_understand-logs` | Weight-Space Programming 글이 공개한 편집과 풀이 기록 | `qw.py`의 `load_released`, `astra_edit`, `astra_deltas` 함수. 이 함수를 부르는 스크립트는 `ana_stop_oracle.py`, `conds.py`, `export_edits.py`, `expA_record.py`, `expA2_tight.py`, `expC_escape.py`다 |
| `poc/ext/weight-space-lp` | `poc/README.md`에 선형계획법 재현과 steering vector가 들어 있다고 적힌 저장소 | `conds.py` |
| `poc/data/math500.json`, `poc/data/gsm8k_test.json` | MATH-500과 GSM8K 문제 | 이름이 `expA`, `expC`, `expT`로 시작하는 스크립트와 `anaA_check.py` |

실험 0의 표를 만든 `expF_gated.py`, `expG_divergence.py`, `expH_share.py`는 이 가운데 모델만 읽는다.

셋째, 스크립트와 `poc/README.md`가 가리키는 파일 일부가 받은 묶음에 들어 있지 않았다. `edits_poc.npz`, `outC/results.json`, `out*.txt`, `released_stats.json`, `results_anaA.json`, `results_D.json`, `results_E.json`, `results_T.json`, 그리고 중간 출력 폴더 `outA/`와 `outA2/`다. `PLAN.md`에 따르면 이 파일들을 읽고 쓰는 스크립트는 후보 1과 2를 처음 검증할 때 쓴 것이다. `expF_gated.py`, `expG_divergence.py`, `expH_share.py`는 이 파일들을 읽지 않는다.

## 서버에서 돌릴 때

minjunjee의 서버 기준이다.

- GPU는 다른 프로젝트와 같이 쓴다. 띄우기 전에 `nvidia-smi`로 비어 있는지 확인하고, 실행할 때 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<번호>`를 붙인다.
- 모델 weight, checkpoint, 큰 출력은 `/data/minjun_dev/Introduction_LLM/Flexible_WSP/`에 두고 repo 안에는 symlink만 건다. 홈 디렉터리가 있는 partition은 여유가 거의 없다.
- Hugging Face 모델과 dataset은 `HF_HOME=/data/minjun_dev/hf_home`을 지정해서 받는다.
- 모델 가중치 전체는 git에 올리지 않는다. 편집은 u와 v만 `npz` 파일로 저장해서 올린다.

## Branch

branch 이름은 `{minjunjee|daehanwon}/{branch name}` 형식으로 쓴다. `PLAN.md`에는 실험 결과를 `code/wsp-poc` branch에 push한다고 적혀 있다. 이 repo에서는 같은 용도로 `minjunjee/wsp-poc`를 쓴다.
