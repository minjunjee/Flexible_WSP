> 서버에서 돌릴 실험 계획은 한 단계 위의 `PLAN.md`에 있다. 이 폴더는 그 계획의 실험 0에 해당하는 코드와 결과다.

# 후보 1, 2 PoC 코드

대상 모델은 `Qwen/Qwen3.5-0.8B`이고, CPU 2코어와 bfloat16으로 돌렸다. GPU에서도 그대로 돈다.

## 준비

```
pip install torch transformers safetensors huggingface_hub scikit-learn pyarrow
git clone https://github.com/RikiyaT/lm_understand-logs ext/lm_understand-logs      # MIT 실험 기록, 편집 파일, 추론 기록
git clone https://github.com/nathanrs/weight-space-lp ext/weight-space-lp            # 선형계획법 재현, 조종 벡터
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/qwen35_08b
```

스크립트에는 `/home/claude/poc` 경로가 그대로 적혀 있다. 한 번에 바꾸려면 `sed -i 's#/home/claude/poc#내 경로#g' *.py`를 쓴다. MATH-500과 GSM8K 문제는 `data/`에 json으로 넣는다. 형식은 `expA_record.py`가 읽는 대로 문제 id를 키로 하고 `problem`, `answer`, `level`을 값으로 둔다.

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
