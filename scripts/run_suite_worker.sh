#!/usr/bin/env bash
# 지시문 여러 개를 한 GPU에서 차례로 돌린다.
#   scripts/run_suite_worker.sh <GPU 번호> <모델 이름> <지시문 이름> [<지시문 이름> ...]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=$1; MODEL=$2; shift 2
for task in "$@"; do
    out=results/exp5/$MODEL/$task
    mkdir -p "$out"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m wsp.run_suite --model "$MODEL" --task "$task" --out "$out" > "$out/log.txt" 2>&1
    echo "$task exit $? $(TZ=Asia/Seoul date '+%T KST')"
done
