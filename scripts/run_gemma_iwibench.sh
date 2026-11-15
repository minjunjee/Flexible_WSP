#!/usr/bin/env bash
# IWIBench의 구조화된 응답 과제를 Gemma 3 4B에서 episode 하나씩 따로 돌린다.
#   scripts/run_gemma_iwibench.sh <GPU 번호> <episode seed>
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/data/minjun_dev/hf_home} CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1
s=$2
B=results/iwibench/gemma3-4b_prompts_written_without_seeing_the_code/seed$s
W=results/iwibench/gemma3-4b_wide/seed$s
mkdir -p "$B" "$W"
nohup .venv/bin/python -m wsp.iwibench_structured --model gemma3-4b --no_fences --seeds "$s" --out "$B" \
    --prompt_file results/E4/E4a_prompts_written_without_seeing_the_benchmark_code.json > "$B/log.txt" 2>&1 &
nohup .venv/bin/python -m wsp.iwibench_structured --model gemma3-4b --no_fences --seeds "$s" --out "$W" --prompts wide > "$W/log.txt" 2>&1 &
