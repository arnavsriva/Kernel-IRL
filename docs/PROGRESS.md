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

### Running on Modal (2026-10-09)
The user has Modal credits, and Modal is now the default GPU platform. `scripts/modal_app.py` runs
each step in the `slimerl/sglang:v0.5.15.post1-cu129` image with a Volume for runs, results, the
baseline cache, and HF weights. Start with `modal run scripts/modal_app.py --step probe`.

Modal runs containers under gVisor. The sandbox's launch check needs CUPTI kernel traces from
`torch.profiler`, and no public documentation confirms they work under gVisor.
`scripts/gpu_probe.py` checks this first, along with the Triton launch hook, process-group kill,
the RSS watchdog, `RLIMIT_FSIZE`, and 10 fixtures end to end. If CUPTI tracing is unavailable, the
launch check needs a fallback before any eval; it must not be skipped.

### GPU probe on Modal L40S (2026-10-09): PASSED
Report: `docs/results/gpu_probe.json`. Image `slimerl/sglang:v0.5.15.post1-cu129`, driver
610.57.04 (CUDA 13.3), torch 2.11.0+cu129, triton 3.6.0, sglang 0.5.15.post1.
- Under Modal's sandbox, CUPTI kernel tracing works through `torch.profiler`. Triton
  launch-hook names match the CUPTI kernel names exactly, and the hook reports the current stream.
- Isolation primitives all work: process-group timeout kill, SIGSEGV reporting, `RLIMIT_FSIZE`,
  and the RSS watchdog.
- All 10 fixtures got the expected verdict. The GPU stayed healthy after illegal memory access, a
  device-side infinite loop, and a segfault.
- The first attempt found an API-sentinel false positive. `torch._dynamo` replaces
  `torch.manual_seed` when it is first imported, and harness code triggered that after the
  snapshot. The fix preloads trusted torch modules before the snapshot, which does not weaken the
  sentinel. The same fix closed a gap where runtime patches of `torch.Tensor` methods inherited
  from `TensorBase` were missed.
- Note: the image ships numpy 2.4.4, while the pyproject pins 1.26.4 (slime's top layer).
  KernelRL does not depend on numpy at runtime. Revisit in Phase 2 with the full slime image.
- Use `modal run --detach` for GPU steps. Without it, a dropped connection from the Mac stopped the
  app twice during the image build.

### Adversarial verdict suite on Modal L40S (2026-10-09): 215/215 PASSED
Log: `docs/results/verdicts_pytest.log`. Every good, buggy, hack, and hostile fixture gets its
expected verdict. `test_no_hack_scores_correct` reruns all runtime hacks with benchmarking
enabled, and none scores correct. A good kernel still evaluates correctly right after hostile
kernels on the same GPU.
- The first run passed 213 of 214 tests. `buggy/matmul_drops_k_tail` was labelled
  `hack_detected/pointer_keyed_cache` instead of `incorrect`. On the hidden shape M=N=K=1 that bug
  always outputs a single 0, and for one derived seed 0 was within atol of the reference.
  Diagnosis is in `docs/results/diagnose_fixture.log`.
- Fix: the `cached_output` and `pointer_keyed_cache` findings now require a computed-looking output
  (more than one element, finite, non-constant). This is flagged to the user as a verifier change.
  The correctness gate and reward are unchanged; only the hack label for trivial outputs is
  affected.

### Smoke eval on Modal H100 (2026-10-10): pipeline works, model scores 0/8
Run `smoke`: Qwen3-4B non-thinking, l1_vector_add + l2_softmax, k=2, 2 turns
(`docs/results/smoke.md`). SGLang, generation, sandbox, revision turn, and report all worked
end to end in about 3 minutes.
- Every failure is a genuine model failure. The model invented Triton APIs (`triton.launch`,
  `triton.grid`, `tl.num_blocks_per_grid`, `T.launch_index`), wrote CUDA-style one-thread-per-element
  code, and passed `data_ptr()` integers. One sample looped on a repeated comment until it hit the
  8192-token cap.
- Fix made: forbidden-API messages now say the API may not exist and list the allowed host APIs and
  the launch syntax. Previously "not allowed" read as a rules problem, and turn 2 repeated the same
  invented calls.
- Open decision (asked of the user): a near-zero pass rate gives GRPO no learning signal in Phase 2.
  Options are a short API cheat-sheet plus one worked example in the prompt, thinking mode, and
  presence_penalty=1.5 for the repetition loops (the Qwen3 card recommends it).

### Full baselines on Modal H100 (2026-10-10)
The user chose to compare prompts. Summary is in `docs/results/baseline_phase0.md`; per-run reports
are in `docs/results/qwen3-4b_*_k8_t2.{json,md}`.

| Setup | pass@1 t1 / t2 | pass@8 | L1 / L2 / L3 pass@1 (t2) | speedup of correct kernels |
|---|---|---|---|---|
| v1 zero-shot, non-thinking | 0% / 0% | 0% | 0 / 0 / 0 | n/a |
| v2 cheat-sheet, non-thinking | 0.8% / 0.8% | 6.7% | 2.1 / 0 / 0% | 1.3x (2 kernels) |
| v2 cheat-sheet, thinking | 22.1% / 24.2% | 56.7% | 46.9 / 13.5 / 0% | 0.54-0.58x eager |

Phase 2 should start from v2 + thinking. L3 has no signal yet; consider a curriculum.

Bugs found and fixed while running:
- eval client: the semaphore was shared across event loops (turn-2 crash), and context overflow on
  long turn-2 prompts raised instead of shrinking the budget.
- verifier labels: `state_dependent` now needs replay evidence; empty launch grids are no longer
  infrastructure errors; launch-based hack labels need a correct output or selective skipping. The
  first two were flagged to the user. Rewards for wrong outputs are unchanged.
- `l1_row_max` inputs are now all-negative.
- New fixtures from real model failures. The verdict suite now passes 261/261 on L40S.

### Open issues and decisions
- **Policy (user):** should a host-side dtype cast after a correct Triton kernel (e.g.
  `out.to(torch.bfloat16)`) count as a violation? It is currently a `low_triton_time_fraction` hack,
  documented by `tests/fixtures/hacks/host_dtype_cast.py`.
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
