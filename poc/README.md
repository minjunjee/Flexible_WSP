> 서버에서 돌릴 실험 계획은 한 단계 위의 `PLAN.md`에 있다. 이 폴더는 그 계획의 실험 0에 해당하는 코드와 결과다.

# 후보 1, 2 PoC 코드

대상 모델은 `Qwen/Qwen3.5-0.8B`이고, CPU 2코어와 bfloat16으로 돌렸다. GPU에서도 그대로 돈다. 다만 스크립트에 모델을 GPU로 옮기는 코드는 없어서, GPU가 있는 서버에서도 CPU로 돈다.

## 준비

```
pip install torch transformers safetensors huggingface_hub scikit-learn pyarrow
git clone https://github.com/RikiyaT/lm_understand-logs ext/lm_understand-logs      # MIT 실험 기록, 편집 파일, 추론 기록
git clone https://github.com/nathanrs/weight-space-lp ext/weight-space-lp            # 선형계획법 재현, 조종 벡터
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/qwen35_08b
```

스크립트는 자기가 놓인 폴더를 기준으로 파일을 읽고 쓴다. 원래는 `/home/claude/poc`라는 경로가 그대로 적혀 있었고, 2026-11-15에 서버로 옮기면서 바꿨다. MATH-500과 GSM8K 문제는 `data/`에 json으로 넣는다. 형식은 `expA_record.py`가 읽는 대로 문제 id를 키로 하고 `problem`, `answer`, `level`을 값으로 둔다. `make_inputs.py`가 이 두 파일과 `released_stats.json`을 만든다.

## 파일

| 파일 | 하는 일 |
| --- | --- |
| `qw.py` | 모델 불러오기, 루프 판정, 공개 편집을 체크섬까지 맞춰 다시 만들기 |
| `gen.py` | 한 배치 안에 편집된 모델 여러 개를 넣고 생성. 편집은 실제 저장될 bfloat16 가중치와 같은 값으로 적용 |
| `expA_record.py` | 공개 추론 기록 350개를 다시 통과시켜 뉴런 단위 지표와 활성값 지표를 기록 |
| `expA2_tight.py` | 굳은 루프 구간과 정상 추론 구간의 뉴런 통계 |
| `anaA.py`, `anaA_check.py` | 측정 분석. `anaA_check.py`가 더 강한 기준선과 부트스트랩 구간을 쓴 최종본 |
| `anaLoc.py` | 루프 편집의 위치를 방향 보기와 뉴런 보기로 찾기 |
| `conds.py` | 비교한 편집 조건 정의 |
| `expC_escape.py` | 굳은 루프 안에서 가중치만 바꿔 이어 쓰게 하는 갈래 실험 |
| `expE_answer.py` | 편집이 짧은 답을 바꾸는지 |
| `expT_side.py` | 정상 추론 구간에서 다음 토큰 분포가 얼마나 달라지는지 |
| `expD_install.py` | 지시문 없이 JSON 형식과 짧은 답을 심는 편집 |
| `export_edits.py` | 편집 조건을 npz로 내보내거나, 조건 하나를 vLLM용 safetensors 체크포인트로 저장 |
| `anaC.py` | 갈래 실험, 짧은 답, 분포 변화를 한 표로 요약 |
| `edits_poc.npz` | 비교한 편집 조건 전부. 뉴런 번호, 방향, 열별 변화량 |
| `results_*.json`, `out*.txt`, `outC/results.json` | 이번 실행의 결과. `outC/results.json`에는 갈래마다 이어 쓴 글이 들어 있다 |

## 순서

```
python expA_record.py --n_base 100 --n_math 150 --n_gsm 100
python expA2_tight.py 40 30
python anaLoc.py && python anaA.py && python anaA_check.py
python expC_escape.py 12 21,23
python expE_answer.py 21,23
python expT_side.py 5 21,23
python expD_install.py
```

## 주의

- CPU에서 프로세스 두 개를 동시에 돌리면 둘 다 열 배 가까이 느려진다.
- bfloat16 추론은 배치 크기에 따라 로짓이 0.2에서 0.4 정도 달라진다. 편집 전후 비교는 반드시 같은 배치 안에서 한다.
- 긴 문맥을 여러 개 묶어 넣을 때 전 위치의 로짓을 만들면 메모리가 수 GB 필요하다. `gen.py`는 마지막 위치만 계산한다.
- 여기서 정답률은 재지 않았다. 정답률은 `export_edits.py`로 체크포인트를 만들어 `weight-space-lp`의 vLLM 평가 코드로 재면 된다.

## 후보 2 추가 실험: 답 시작 자리에서만 켜지는 편집

11월 15일 저녁에 돌렸다. 지시문 없이 JSON으로 답하게 만드는 편집을 훈련 없이 만든다. 답의 첫 token을 정하는 자리는 마지막 prompt token 자리다. 이 자리에서 지시문이 있을 때와 없을 때의 hidden state 차이를 layer마다 구한다. 그 자리에서만 그 차이가 더해지도록 MLP 출력 행렬 `down_proj`에 rank-1 변경 `u v^T`를 더한다. `v`는 그 자리의 MLP 활성값과 곱하면 1, 다른 자리와 곱하면 0에 가깝도록 ridge regression으로 구한다.

| 파일 | 하는 일 |
| --- | --- |
| `gen.py` | 한 배치에 여러 편집 모델을 넣고 생성한다. 이번에 `("sched", layer, {j: vec})` 연산을 추가했다. 답의 j번째 자리에만 vec을 더한다. j가 0이면 마지막 prompt token 자리다 |
| `expF_gated.py` | JSON 실험이다. 조건은 21개다. 편집 없음, 모든 자리에 더하기, 답 시작 자리에만 더하기, rank-1 가중치 변경, 지시문을 prompt에 넣기 |
| `expG_divergence.py` | 지시문 있는 모델의 답을 따라가며 자리마다 KL을 잰다. 짧게 답하기 지시문의 자유 생성 비교도 들어 있다 |
| `expH_share.py` | 한 자리에서 지시문이 만드는 hidden state 차이 가운데 질문이 바뀌어도 같은 부분의 비율을 잰다 |
| `ana_stop_oracle.py` | 공개된 풀이 기록에서 정답 숫자가 생각 과정에 나온 횟수와 최종 정답 수를 센다. 결과는 `results_oracle_stop.json`이다 |
| `results_F.json`, `results_G.json`, `results_H.json` | 이번 실행 결과다. 조건별 출력 전문이 들어 있다 |
| `logF.txt`, `logG.txt`, `logH.txt` | 실행 기록이다 |

```
python expF_gated.py          # CPU 2코어에서 약 10분
python expG_divergence.py     # 약 7분
python expH_share.py          # 약 2분
```

주요 결과다. 모델은 Qwen3.5-0.8B, 생성은 greedy, 편집을 만든 질문은 16개, 잰 질문은 30개다.

- 편집 없음: 유효한 JSON 0/30.
- 차이를 layer 17의 모든 자리에 더함: 유효한 JSON 6/30, 답이 맞은 것 2/30.
- layer 3, 7, 11, 15, 19, 23에 rank-1 가중치 변경 하나씩: 유효한 JSON 27/30, key 이름이 answer인 것 14/30, 답이 맞은 것 28/30.
- 지시문을 prompt에 넣음: 30/30, 30/30, 25/30.
- 짧게 답하기 지시문은 같은 방법으로 옮겨지지 않았다. 세 단어 이하 20/20, 답이 맞은 것 4/20이다. 지시문을 넣으면 15/20이다.

## 서버에서 다시 만든 파일

받은 묶음에는 위 표에 나오는 파일 가운데 일부가 들어 있지 않았다. 빠진 것은 전부 스크립트가 읽는 입력이거나 스크립트의 출력이다. 그래서 2026-11-15 22:50부터 23:06까지(KST) 서버에서 스크립트를 다시 돌려 만들었다. 이 절은 그때 무엇을 어떻게 만들었고, 원래 결과와 얼마나 같은지를 적는다.

다시 돌리는 명령은 아래와 같다. `run_missing.sh`는 위 순서 절의 명령을 같은 인자로 부른다. 처음 세 단계인 `make_inputs.py`, `expA_record.py`, `expD_install.py`는 같은 명령을 직접 쳐서 돌렸고, 나머지는 이 스크립트로 돌렸다.

```
scripts/setup_env.sh                                           # repo 맨 위에서. venv, 모델, 외부 저장소
cd poc && PY=../.venv/bin/python POC_THREADS=16 ./run_missing.sh
../.venv/bin/python compare_regen.py                           # 원래 결과와 견준다
```

원래 실행과 같게 둔 것은 셋이다. 모델을 CPU에서 돌렸고, 가중치는 bfloat16이고, 스크립트 인자는 순서 절과 같다. 다른 것도 셋이다. CPU가 AMD EPYC 9555이고, thread를 2개가 아니라 16개 썼고, package는 torch 2.15.0과 transformers 5.23.0이다. thread 수는 `POC_THREADS`로 정하고 기본값은 원래대로 2다. 원래 실행이 쓴 CPU와 package 버전은 모른다.

| 파일 | 만든 스크립트 | 덧붙일 말 |
| --- | --- | --- |
| `data/math500.json`, `data/gsm8k_test.json` | `make_inputs.py` | git에는 올리지 않는다. 공개된 풀이 기록 800개와 문제 id, 정답을 견줬고 전부 일치했다 |
| `released_stats.json` | `make_inputs.py` | 원래 파일을 본 적이 없어서 내용을 짐작해 다시 짰다. 아래에 따로 적는다 |
| `outA/`, `outA2/` | `expA_record.py`, `expA2_tight.py` | 각각 1.8 GB와 1.2 GB다. `/data`에 두고 symlink를 걸었다. git에는 올리지 않는다 |
| `results_anaA.json` | `anaA.py` | |
| `outC/results.json` | `expC_escape.py` | |
| `results_E.json`, `results_T.json` | `expE_answer.py`, `expT_side.py` | |
| `results_D.json` | `expD_install.py` | |
| `edits_poc.npz` | `export_edits.py` | array 84개 |
| `out*.txt` | 위 스크립트들이 화면에 찍은 글 | 원래 실행의 기록이 아니다 |
| `regen_check/` | `anaLoc.py`, `anaA_check.py`, `anaC.py`, `compare_regen.py` | 원래 결과가 이미 있는 세 파일을 다시 만든 것과, 둘을 견준 `comparison.json` |

`results_loc.json`, `results_anaA_check.json`, `results_C_summary.json`, `results_D_rescored.json`은 원래 실행의 결과 그대로다. 덮어쓰지 않았다.

### `released_stats.json`을 다시 짠 방법

`expA_record.py`는 이 파일을 읽어서, 편집 전 모델의 풀이 기록 가운데 어느 것을 쓸지 고른다. 길이 한도까지 간 기록이고 loop가 300번째에서 1,150번째 token 사이에 시작한 것만 쓴다. 이 파일은 묶음에 없었고 만드는 스크립트도 없었다.

`make_inputs.py`는 풀이 기록마다 네 값을 적는다. `n_tok`은 답의 token 수다. `closed`는 답에 `</think>`가 있는지다. `onset`은 loop가 시작하는 token 위치이고 `qw.loop_onset`으로 구한다. `cut`은 token 수가 32,000 이상인지다. `expA_record.py`가 자기 출력에 `cut`을 적을 때 쓰는 기준과 같게 정했다.

원래 파일이 `cut`을 어떻게 정했는지는 모른다. `weight-space-lp` 저장소는 `</think>`가 없는 기록을 cut이라고 부른다. 두 기준으로 각각 골라 봤는데 고른 기록이 같았다. MATH-500은 두 기준 모두 93개, GSM8K는 123개였고 한쪽에만 든 기록은 없었다.

### 원래 결과와 얼마나 같은가

다시 만든 파일은 원래 파일과 숫자 하나하나까지 같지는 않다. 편집을 넣지 않은 조건은 같았고, 편집을 넣은 조건 일부가 달랐다. 아래 숫자는 `compare_regen.py`가 찍은 것이고 `regen_check/comparison.json`에 전부 들어 있다.

| 원래 파일 | 견준 방법 | 결과 |
| --- | --- | --- |
| `results_loc.json` | 숫자 216개를 하나씩 | 205개는 차이가 0.01 이하다. 나머지 11개는 뉴런 수를 센 값이고 차이가 1이나 2다. 예를 들어 layer 22에서 고른 뉴런 수가 33개에서 35개가 됐다 |
| `results_anaA_check.json` | 숫자 126개를 하나씩 | 122개는 차이가 0.01 이하다. 가장 큰 차이는 0.034이고 bootstrap으로 구한 p값에서 났다 |
| `results_C_summary.json` | 조건 9개에서 loop 12개 중 벗어난 수 | 7개 조건은 같다. `astra_edit`는 9개에서 11개, `random_1024`는 1개에서 2개가 됐다 |
| `results_D_rescored.json` | 지시문, 질문 묶음, 조건으로 나눈 56칸씩 | JSON 지시문은 형식을 지킨 비율이 54칸에서 같고, 정답 비율은 50칸에서 차이가 0.1 이하다. 짧게 답하기는 형식을 지킨 비율이 28칸에서 같고, 정답 비율은 53칸에서 차이가 0.1 이하다 |

`results_loc.json`과 `results_anaA_check.json`이 거의 같게 나왔으므로, 다시 짠 `released_stats.json`으로 고른 풀이 기록은 원래와 같거나 거의 같다고 본다. 고른 기록이 달랐다면 이 통계들이 이만큼 맞기 어렵다. 다만 원래 고른 기록의 목록이 없어서 직접 확인하지는 못했다.

`expC_escape.py`는 token을 sampling으로 뽑는다. 그래서 `results_C_summary.json`은 숫자가 조금만 달라도 뽑히는 token이 달라진다.

`expD_install.py`와 `expE_answer.py`는 greedy로 생성하는데도 결과가 달랐다. `expD_install.py`에서 편집 없는 조건은 세 질문 묶음 모두 형식 비율과 정답 비율이 원래와 같았다. 다른 칸은 전부 편집을 넣은 조건이다. `expE_answer.py`의 결과는 `results_C_summary.json`에 든 값으로 견줬다. 가장 크게 다른 것은 `random_1024` 조건이고, 짧은 답이 바뀐 비율이 0.44에서 0.26이 됐다. 이 조건은 끌 뉴런을 무작위로 고른다. 후보를 정하는 통계가 조금 달라져서 고른 뉴런이 달라졌을 것이다. 나머지 조건은 이 비율의 차이가 0.04 이하다. 내 추측으로는 CPU와 package가 달라 bfloat16 계산값이 조금 다르고, 그 차이가 편집에 쓰는 통계와 다음 token 선택을 조금씩 바꾼다. 이 추측을 확인하는 실험은 하지 않았다.

짧게 답하기의 형식 비율은 견주는 기준부터 확실하지 않다. `results_D_rescored.json`은 `results_D.json`을 다시 채점한 파일인데 채점 스크립트가 묶음에 없다. `expD_install.py`는 답에 `<|im_end|>`라는 글자를 남겨 두고 이것을 두 단어로 센다. 그래서 빈 답이 세 단어 이하라는 기준을 통과한다. 원래의 다시 채점한 파일에서는 이런 답이 0단어로 적혀 있다. `compare_regen.py`는 `<|im_end|>`를 지우고 단어를 셌다. 이 규칙은 원래 채점을 짐작한 것이다.
