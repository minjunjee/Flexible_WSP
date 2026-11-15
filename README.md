# Flexible_WSP

지시문을 훈련 없이 모델 가중치로 옮기는 방법의 PoC를 진행하는 repo다. minjunjee와 daehanwon이 함께 쓴다.

확인하려는 주장은 이것이다. 지시문은 답의 몇 군데 자리에서만 모델이 다음 token에 매기는 확률을 바꾼다. 그 자리에서만 작동하는 가중치 변경을 넣으면, 지시문을 prompt에 넣지 않아도 모델이 지시문을 따른다. 가중치 변경은 MLP의 출력 행렬 `down_proj`에 vector 두 개를 곱해 만든 행렬 `u v^T`를 더하는 rank-1 변경이다. `PLAN.md`에서는 이 변경을 조건부 편집이라고 부른다.

실험 계획, 조건부 편집을 만드는 절차, 실험별 판정 기준은 전부 `PLAN.md`에 있다. 실험을 돌리기 전에 그 문서를 처음부터 끝까지 읽는다.

## 지금 상태

2026-11-15 기준이다.

- 실험 0은 끝났다. Qwen3.5-0.8B를 CPU로 돌린 것이고 코드와 결과가 `poc/`에 있다. `PLAN.md`에 따르면 MLP 행렬 6개에 조건부 편집을 하나씩 넣었을 때, 지시문 없는 질문 30개 중 27개에서 유효한 JSON이 나왔다. 이 서버에서 다시 확인한 값은 아니다.
- 받은 묶음에 없던 파일은 2026-11-15에 서버에서 `poc/` 스크립트를 다시 돌려 만들었다. 무엇을 만들었고 원래 결과와 얼마나 같은지는 `poc/README.md`의 마지막 절에 있다. 실험 0의 표를 만든 `expF_gated.py`, `expG_divergence.py`, `expH_share.py`는 이때 돌리지 않았다.
- 실험 1부터 14까지는 시작하지 않았다.

## 폴더

| 경로 | 내용 |
| --- | --- |
| `PLAN.md` | 실험 0부터 14까지의 계획과 판정 기준 |
| `poc/` | 실험 0의 코드와 결과. 파일별 설명은 `poc/README.md`에 있다. 받은 묶음에서 바꾼 것은 스크립트 안의 경로와 thread 수 설정뿐이다 |
| `docs/plan_related_work.md` | 원래 계획 문서에서 `PLAN.md`로 옮기지 않은 부분. 겹치는 최근 연구의 표와 출처 목록이 있다 |
| `scripts/setup_env.sh` | Python 환경, 모델, 외부 저장소를 `/data` 아래에 준비하고 repo 안에 symlink를 건다 |
| `wsp/` | 실험 1부터 새로 짜는 코드를 둔다. 지금은 비어 있다 |
| `results/` | 실험마다 폴더를 하나씩 둔다. 폴더마다 `README.md`를 쓰고, `results/SUMMARY.md`에 실험마다 한 문단씩 쌓는다 |
| `requirements.txt` | `poc/` 스크립트가 쓰는 Python package 목록 |
| `requirements.lock.txt` | 2026-11-15에 서버에 실제로 설치된 버전. Python 3.12.3, torch 2.15.0, transformers 5.23.0이다 |

## `poc/`를 돌리려면

```
scripts/setup_env.sh
cd poc && ../.venv/bin/python expF_gated.py
```

`scripts/setup_env.sh`는 세 가지를 `/data` 아래에 준비한다. Python 환경, `Qwen/Qwen3.5-0.8B` 모델, 외부 저장소 두 개다. 외부 저장소는 Weight-Space Programming 글이 공개한 기록인 `lm_understand-logs`와 `weight-space-lp`다. 그다음 repo 안에 `.venv`, `poc/models`, `poc/ext`, `poc/outA`, `poc/outA2`라는 symlink를 건다. 다른 계정에서는 `DATA_ROOT`와 `HF_HOME`을 바꿔서 부른다.

`poc/`의 스크립트는 모델을 CPU에서 돌린다. GPU로 옮기는 코드는 들어 있지 않다. thread 수는 `POC_THREADS`로 정하고 기본값은 2다.

실험 0의 표를 만든 `expF_gated.py`, `expG_divergence.py`, `expH_share.py`는 모델만 있으면 돈다. 나머지 스크립트는 문제 파일과 앞 단계의 출력이 더 필요하다. `poc/run_missing.sh`가 그것들을 순서대로 만든다.

## 서버에서 돌릴 때

minjunjee의 서버 기준이다.

- GPU는 다른 프로젝트와 같이 쓴다. 띄우기 전에 `nvidia-smi`로 비어 있는지 확인하고, 실행할 때 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<번호>`를 붙인다.
- 모델 weight, checkpoint, 큰 출력은 `/data/minjun_dev/Introduction_LLM/Flexible_WSP/`에 두고 repo 안에는 symlink만 건다. 홈 디렉터리가 있는 partition은 여유가 거의 없다.
- Hugging Face 모델과 dataset은 `HF_HOME=/data/minjun_dev/hf_home`을 지정해서 받는다.
- 모델 가중치 전체는 git에 올리지 않는다. 편집은 u와 v만 `npz` 파일로 저장해서 올린다.

## Branch

branch 이름은 `{minjunjee|daehanwon}/{branch name}` 형식으로 쓴다. `PLAN.md`에는 실험 결과를 `code/wsp-poc` branch에 push한다고 적혀 있다. 이 repo에서는 같은 용도로 `minjunjee/wsp-poc`를 쓴다.
