# Progress

## 2026-10-04: Phase 0 starter code

### Environment
- Dev machine: Apple M5 Pro (no NVIDIA GPU). All CPU-side code and tests run locally with
  Python 3.12.14 and torch 2.11.0 (CPU).
- Pins come from slime's image `slimerl/sglang:v0.5.15.post1-cu129`: sglang 0.5.15.post1,
  torch 2.11.0+cu129, triton 3.6.0 (pinned by torch), openai 2.6.1, transformers 5.12.1, numpy
  1.26.4, Python 3.12. Megatron commit 1dcf0dafa884ad52ffb243625717a3471643e087 is noted for Phase 2.
- Model: Qwen/Qwen3-4B at HF revision 1cfa9a7208912126459214e8b04321603b3df60c (8.1 GB).
- Triton launch hooks (`triton.knobs.runtime.launch_enter_hook`) and SGLang flags and metric names
  were checked against the pinned versions' source, not guessed.

### Done
- **A. Tasks.** 30 tasks (12 L1, 12 L2, 6 L3) with JSON schema, reference, and separate hidden
  shapes. The validator passes on CPU; the largest matmul shapes are skipped locally.
  Provenance: original code, categories inspired by KernelBench (MIT).
- **B. Sandbox.** Static AST allowlist, isolated compile/check/bench stages, correctness with
  fresh seeds and allocator poisoning, 13 runtime and bench anti-hacking checks, paired benchmark
  with clean-process eager and torch.compile baselines, feedback renderer, and reward with config.
- **C. Fixtures.** 5 good, 4 buggy, 20 hacks, and 9 hostile fixtures. CPU tests cover the static
  verdict for every fixture, isolation, reward, tasks, decoupling, serving, and eval. GPU tests are
  written and skipped locally.
- **D. Serving and eval.** SGLang launch script, prompt template, robust extraction, async client,
  resumable two-turn eval, and JSON plus markdown report.
- **E. Prefix-cache benchmark.** Radix on/off runner and comparison report.
- Tests: `pytest` gives 184 passed and 27 skipped (GPU-only).

### Not done: needs a GPU box
Nothing GPU-side has been executed yet. On the box, run
`bash scripts/setup_gpu_box.sh docker --yes-large-download` and then
`bash scripts/run_phase0_gpu.sh`. In order, that will:
1. Validate tasks on CUDA. Expect tolerance adjustments for a few fp16/bf16 tasks, because GPU
   eager uses different accumulation than CPU.
2. Run `pytest` including `tests/gpu`. **If any hack fixture is judged correct, stop and report it
   before going further.** Likely first-run fixes:
   - Profiler kernel-event extraction (`checks._kernel_events`).
   - Triton kernel-name matching between the launch hook and CUPTI.
   - API-sentinel false positives from torch lazily replacing callables.
3. Run the baseline eval (pass@1/pass@8, status mix, speedups, hacks, 1-turn vs 2-turn) into
   `docs/results/`.
4. Run the prefix-cache benchmark and write the interpretation.

### Open issues and decisions
- Clock locking for benchmarks (`nvidia-smi -lgc`) needs root, so ask before changing system state.
- The Docker image is over 20 GB, so the setup script requires an explicit flag.
- Phase 0 shares one GPU between SGLang (mem-fraction 0.5) and the sandbox. The eval runs
  generation and evaluation strictly in sequence so timings never overlap with inference.
- Hidden shapes are fixed per task. Per-evaluation sampled hidden shapes are a later hardening step.

### Proposed Phase 1 interface (Ray reward workers)
`kernelrl.sandbox.evaluate.evaluate(task_id, source, ...) -> EvalResult` is already the unit of
work. It is torch-free in the parent, file-based, and takes a GPU id, so it can be wrapped
unchanged:

```python
@ray.remote(num_gpus=1, max_concurrency=1, max_restarts=-1)
class RewardWorker:
    def __init__(self, sandbox_overrides: dict | None = None): ...
    def evaluate(self, req: dict) -> dict:
        # req = {request_id, task_id, source, run_bench=True, seeds=None}
        # returns {request_id, result: EvalResult.to_dict(), reward, feedback, worker, gpu_uuid}
    def health(self) -> dict:          # runs the GPU probe; the pool restarts the actor if unhealthy

@ray.remote(num_cpus=1)
def static_prefilter(task_id, source) -> dict | None:   # CPU-only; skips GPUs for static rejects

class RewardPool:                        # driver side, used by slime's custom reward function
    def submit(self, reqs: list[dict]) -> list[ObjectRef]
    def results(self, refs, timeout_s) -> list[dict]     # infra_error -> retry once on another worker
```

- One kernel at a time per GPU (`max_concurrency=1`) keeps benchmarks clean. Ray sets
  `CUDA_VISIBLE_DEVICES` per actor.
- Baselines move from a per-host disk cache to a shared store, either a `BaselineStore` actor or a
  shared filesystem, keyed exactly as now.
- Metrics to study: evals/s per GPU, stage latency breakdown (`EvalResult.stage_seconds`), and
  queueing delay vs number of workers.
