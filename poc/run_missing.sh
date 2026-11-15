#!/usr/bin/env bash
# 받은 묶음에 없던 입력과 출력 파일을 다시 만든다. 순서와 인자는 README.md의 "순서" 절과 같다.
# 모델은 CPU에서 bfloat16으로 돈다. 원래 실행과 같은 조건이다.
#
#   PY=/path/to/venv/bin/python POC_THREADS=16 ./run_missing.sh            # 전부
#   PY=... ./run_missing.sh A2 rest                                        # 일부 단계만
#
# 단계: inputs, A, D, A2, rest
#   inputs : data/, released_stats.json
#   A      : outA/, outA.txt
#   D      : results_D.json, outD.txt                (A와 무관하다)
#   A2     : outA2/, outA2.txt                       (A가 끝나야 한다)
#   rest   : results_anaA.json, outC/results.json, results_E.json, results_T.json, edits_poc.npz,
#            regen_check/ 아래의 비교용 파일          (A2가 끝나야 한다)
#
# results_loc.json, results_anaA_check.json, results_C_summary.json은 원래 실행의 결과가 이미 있다.
# 이 셋을 만드는 스크립트는 같은 이름으로 덮어쓰므로, 새로 나온 것은 regen_check/로 옮기고 원래 파일을 되돌려 놓는다.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export POC_THREADS=${POC_THREADS:-16}
STEPS=${*:-inputs A D A2 rest}

keep_original() {   # keep_original <결과 파일> <스크립트> <로그>
    local f=$1 script=$2 log=$3
    mv "$f" "$f.orig"
    if "$PY" "$script" > "$log" 2>&1; then
        mkdir -p regen_check && mv "$f" "regen_check/$f"
    fi
    mv "$f.orig" "$f"
}

for step in $STEPS; do
    case $step in
    inputs) "$PY" make_inputs.py > out_inputs.txt 2>&1 ;;
    A)  "$PY" expA_record.py --n_base 100 --n_math 150 --n_gsm 100 > outA.txt 2>&1 ;;
    D)  "$PY" expD_install.py > outD.txt 2>&1 ;;
    A2) "$PY" expA2_tight.py 40 30 > outA2.txt 2>&1 ;;
    rest)
        "$PY" expC_escape.py 12 21,23 > outC.txt 2>&1 &
        "$PY" expE_answer.py 21,23 > outE.txt 2>&1 &
        "$PY" expT_side.py 5 21,23 > outT.txt 2>&1 &
        "$PY" export_edits.py > out_export.txt 2>&1 &
        "$PY" anaA.py > out_anaA.txt 2>&1 &
        wait
        keep_original results_loc.json anaLoc.py out_anaLoc.txt
        keep_original results_anaA_check.json anaA_check.py out_anaA_check.txt
        keep_original results_C_summary.json anaC.py out_anaC.txt
        ;;
    *) echo "unknown step: $step" >&2; exit 1 ;;
    esac
    echo "step $step done: $(TZ=Asia/Seoul date '+%F %T KST')"
done
