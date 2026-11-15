#!/usr/bin/env bash
# IWIBench의 구조화된 응답 과제를 Gemma 3 4B에서 episode 하나씩 따로 돌린다.
#   scripts/run_gemma_iwibench.sh <GPU 번호> <episode seed>
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/data/minjun_dev/hf_home} CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
s=$2
# 다른 폴더에 쓰려면 OUT_NAME을 준다. 절차를 고친 뒤의 실행을 앞선 실행과 따로 두려는 것이다.
B=results/iwibench/${OUT_NAME:-gemma3-4b_prompts_written_without_seeing_the_code}/seed$s
W=results/iwibench/gemma3-4b_wide/seed$s
mkdir -p "$B" "$W"
nohup .venv/bin/python -m wsp.iwibench_structured --model gemma3-4b --no_fences --seeds "$s" --out "$B" ${EXTRA:-} \
    --prompt_file results/E4/E4a_prompts_written_without_seeing_the_benchmark_code.json > "$B/log.txt" 2>&1 &
if [ "${3:-}" = "both" ]; then
    nohup .venv/bin/python -m wsp.iwibench_structured --model gemma3-4b --no_fences --seeds "$s" --out "$W" --prompts wide > "$W/log.txt" 2>&1 &
fi
