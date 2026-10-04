#!/usr/bin/env bash
# Set up a Linux + NVIDIA box for Phase 0. Two options:
#
#   docker (recommended): the exact image slime builds on, so Phase 2 needs no migration.
#       bash scripts/setup_gpu_box.sh docker --yes-large-download
#   venv: pip-installed pins (lighter, but CUDA wheels must match the driver).
#       bash scripts/setup_gpu_box.sh venv
#
# The Docker image is larger than 20 GB, so it requires an explicit --yes-large-download
# (CLAUDE.md: ask before downloading anything larger than 20 GB).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-slimerl/sglang:v0.5.15.post1-cu129}
MODE=${1:-}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
df -h . | tail -1

case "$MODE" in
  docker)
    if [[ "${2:-}" != "--yes-large-download" ]]; then
      echo "Refusing to pull $IMAGE (>20 GB) without --yes-large-download"; exit 1
    fi
    docker pull "$IMAGE"
    docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' | tee docs/results/docker_image_digest.txt
    cat <<MSG
Start a dev container with:
  docker run --gpus all --ipc=host --network=host -it --name kernelrl \\
    -v "\$PWD":/workspace/KernelIRL -v \$HOME/.cache/huggingface:/root/.cache/huggingface \\
    -w /workspace/KernelIRL $IMAGE bash
then inside it:
  pip install --no-deps -e . && pip install pytest==8.4.2 PyYAML==6.0.2
  python -c "import torch, triton, sglang; print(torch.__version__, triton.__version__, sglang.__version__)"
MSG
    ;;
  venv)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv --python 3.12 .venv
    # torch 2.11.0+cu129 (slime's build); triton 3.6.0 comes with it.
    uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu129 -e ".[gpu,serving,dev]"
    .venv/bin/python -c "import torch, triton, sglang; print(torch.__version__, torch.version.cuda, triton.__version__, sglang.__version__); assert torch.cuda.is_available()"
    ;;
  *)
    echo "usage: $0 docker --yes-large-download | venv"; exit 2 ;;
esac
