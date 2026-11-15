#!/usr/bin/env bash
# Python 환경, 모델, 외부 저장소를 /data 아래에 준비하고 repo 안에 symlink를 건다.
# 여러 번 돌려도 된다. 이미 있는 것은 건너뛴다.
#
#   scripts/setup_env.sh
#   DATA_ROOT=/data/<계정>/Introduction_LLM/Flexible_WSP HF_HOME=/data/<계정>/hf_home scripts/setup_env.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-/data/minjun_dev/Introduction_LLM/Flexible_WSP}
export HF_HOME=${HF_HOME:-/data/minjun_dev/hf_home}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$(dirname "$HF_HOME")/uv_cache}

mkdir -p "$DATA_ROOT"/{models,ext,poc_out/outA,poc_out/outA2,logs}

# Python 환경. torch가 수 GB라서 venv도 /data에 둔다.
[ -d "$DATA_ROOT/venv" ] || uv venv "$DATA_ROOT/venv" --python 3.12
uv pip install --python "$DATA_ROOT/venv/bin/python" -r "$REPO/requirements.txt"

# 모델과 외부 저장소
"$DATA_ROOT/venv/bin/hf" download Qwen/Qwen3.5-0.8B --local-dir "$DATA_ROOT/models/qwen35_08b"
[ -d "$DATA_ROOT/ext/lm_understand-logs" ] || git clone -q https://github.com/RikiyaT/lm_understand-logs "$DATA_ROOT/ext/lm_understand-logs"
[ -d "$DATA_ROOT/ext/weight-space-lp" ] || git clone -q https://github.com/nathanrs/weight-space-lp "$DATA_ROOT/ext/weight-space-lp"

# repo 안의 symlink. 전부 .gitignore에 들어 있다.
ln -sfn "$DATA_ROOT/venv" "$REPO/.venv"
ln -sfn "$DATA_ROOT/models" "$REPO/poc/models"
ln -sfn "$DATA_ROOT/ext" "$REPO/poc/ext"
ln -sfn "$DATA_ROOT/poc_out/outA" "$REPO/poc/outA"
ln -sfn "$DATA_ROOT/poc_out/outA2" "$REPO/poc/outA2"

"$REPO/.venv/bin/python" -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
echo "setup done"
