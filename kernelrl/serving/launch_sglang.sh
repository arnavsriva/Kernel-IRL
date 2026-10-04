#!/usr/bin/env bash
# Launch Qwen3-4B on SGLang, pinned to the version in slime's image (slimerl/sglang:v0.5.15.post1-cu129).
# Flags verified against sglang 0.5.15.post1 server_args.py.
#
#   GPU=0 bash kernelrl/serving/launch_sglang.sh                 # radix cache on (default)
#   GPU=0 RADIX=off PORT=30001 bash kernelrl/serving/launch_sglang.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B}
REVISION=${REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}   # HF commit, 8.1 GB
PORT=${PORT:-30000}
GPU=${GPU:-0}
MEM_FRACTION=${MEM_FRACTION:-0.5}     # Phase 0 shares one GPU with the sandbox
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
RADIX=${RADIX:-on}                    # on | off  (Part E prefix-caching benchmark)
SEED=${SEED:-0}

python - <<'PY'
import importlib.metadata as md, sys
v = md.version("sglang")
sys.exit(0 if v == "0.5.15.post1" else f"sglang {v} installed; expected 0.5.15.post1 (see pyproject.toml)")
PY

EXTRA=()
if [[ "$RADIX" == "off" ]]; then EXTRA+=(--disable-radix-cache); fi

export CUDA_VISIBLE_DEVICES=$GPU
exec python -m sglang.launch_server \
  --model-path "$MODEL" --revision "$REVISION" \
  --host 127.0.0.1 --port "$PORT" \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CONTEXT_LENGTH" \
  --reasoning-parser qwen3 \
  --enable-metrics --enable-cache-report \
  --random-seed "$SEED" \
  "${EXTRA[@]}" "$@"
