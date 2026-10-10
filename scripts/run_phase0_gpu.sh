#!/usr/bin/env bash
# Run every GPU-dependent Phase 0 step in order on one GPU. Stops at the first failure.
#   bash scripts/run_phase0_gpu.sh            # all steps
#   STEPS="validate verdicts" bash scripts/run_phase0_gpu.sh
#   EVAL_ARGS="--k 2 --tasks l1_vector_add" STEPS=eval bash scripts/run_phase0_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPU=${GPU:-0}
STEPS=${STEPS:-"probe validate verdicts eval bench"}
RUN_ID=${RUN_ID:-qwen3-4b_nothink_k8_t2}
export CUDA_VISIBLE_DEVICES=$GPU

wait_server() {  # $1 = port
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null && return 0
    sleep 2
  done
  echo "server on port $1 did not come up"; return 1
}

start_server() {  # $1 = on|off, $2 = port
  RADIX=$1 PORT=$2 GPU=$GPU nohup bash kernelrl/serving/launch_sglang.sh >"runs/sglang_radix_$1.log" 2>&1 &
  echo $! >"runs/sglang_radix_$1.pid"
  wait_server "$2"
}

stop_server() { kill "$(cat "runs/sglang_radix_$1.pid")" 2>/dev/null || true; sleep 5; }

mkdir -p runs
for step in $STEPS; do
  echo "=================== $step ==================="
  case "$step" in
    probe) $PY scripts/gpu_probe.py --out docs/results/gpu_probe.json ;;
    diagnose) FIXTURES="$FIXTURES" $PY scripts/diagnose_fixture.py 2>&1 | tee docs/results/diagnose_fixture.log ;;
    validate) $PY -m kernelrl.tasks.validate --device cuda ;;
    verdicts) $PY -m pytest -v -rs --durations=15 ${PYTEST_ARGS:-tests/} 2>&1 | tee "docs/results/verdicts_pytest${PYTEST_ARGS:+_targeted}.log" ;;  # pipefail propagates pytest failures   # includes tests/gpu
    eval)
      start_server on 30000
      $PY -m kernelrl.eval.run_eval --run-id "$RUN_ID" --server-url http://127.0.0.1:30000/v1 ${EVAL_ARGS:-}
      stop_server on ;;
    bench)
      start_server on 30000
      $PY -m kernelrl.bench.prefix_cache run --label radix_on --server-url http://127.0.0.1:30000/v1 ${BENCH_ARGS:-}
      stop_server on
      start_server off 30001
      $PY -m kernelrl.bench.prefix_cache run --label radix_off --server-url http://127.0.0.1:30001/v1 ${BENCH_ARGS:-}
      stop_server off
      $PY -m kernelrl.bench.prefix_cache compare docs/results/prefix_cache_radix_on.json \
        docs/results/prefix_cache_radix_off.json ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
