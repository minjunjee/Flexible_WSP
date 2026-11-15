#!/usr/bin/env bash
# IWIBench의 구조화된 응답 과제에 LoRA 기준선을 돌린다. 편집을 만들 때와 같은 문장 376개로 학습한다.
# 벤치는 editor에게 gradient를 주지 않으므로 이 실행은 벤치의 규칙 밖에 있는 참고용이다.
#   scripts/run_iwibench_lora.sh <GPU 번호> <qwen35-0.8b|gemma3-4b> <development|test> <학습률,epoch 수>
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/data/minjun_dev/hf_home} CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
model=$2; split=$3; setting=$4
extra=""
if [ "$model" = "gemma3-4b" ]; then extra="--no_fences"; fi
B=results/iwibench_lora/${model}_${split}_lr${setting/,/_epochs}
mkdir -p "$B"
nohup .venv/bin/python -m wsp.iwibench_structured --model "$model" $extra --seeds 0,3,6 --split "$split" --out "$B" --lora "$setting" \
    --prompt_file results/E4/E4a_prompts_written_without_seeing_the_benchmark_code.json > "$B/log.txt" 2>&1 &
