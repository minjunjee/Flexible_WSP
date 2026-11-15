#!/usr/bin/env bash
# 절차를 고정한 뒤에 처음 돌리는 episode. 절차를 고치는 동안 본 적이 없는 episode에서 재려는 것이다.
#   scripts/run_gemma_iwibench_fresh.sh <GPU 번호> <episode seed> <development|test>
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/data/minjun_dev/hf_home} CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
s=$2
split=$3
B=results/iwibench/${OUT_NAME:-gemma3-4b_episodes_first_run_after_freezing_the_procedure}/seed${s}_${split}
mkdir -p "$B"
nohup .venv/bin/python -m wsp.iwibench_structured --model gemma3-4b --no_fences --seeds "$s" --split "$split" --out "$B" ${EXTRA:-} \
    --prompt_file results/E4/E4a_prompts_written_without_seeing_the_benchmark_code.json > "$B/log.txt" 2>&1 &
