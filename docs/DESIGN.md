# KernelRL Phase 0 design

## 1. Evaluation pipeline

```
parent (no torch)                         fresh child processes (torch + triton)
-----------------                         ----------------------------------------
static AST check ──reject──▶ static_reject / compile_error(syntax)
      │
      ├─▶ compile  child: import candidate, call run() once per shape/dtype (JIT + autotune)
      ├─▶ check    child: correctness + anti-hacking checks
      ├─▶ bench    child: paired timing vs eager reference, verified timed outputs
      └─▶ baseline child (cached): eager + torch.compile timings, never sees a candidate
```

* Every stage is a new process group with its own hard timeout (`sandbox/config.yaml`), killed with
  SIGKILL on expiry. A host-RSS watchdog kills runaway host allocations. `RLIMIT_FSIZE` caps file and
  log writes and `RLIMIT_CORE=0` disables core dumps.
* The parent communicates through files in a per-job temp dir and never imports torch. Hangs,
  segfaults, CUDA context corruption, and OOM in a child cannot take the parent down.
  `tests/test_decoupling.py` enforces this.
* After a timeout or crash the parent runs a GPU health probe. The result is recorded as
  `meta.gpu_healthy_after`.
* Precedence when several outcomes apply: static_reject > compile_error > timeout > crash >
  hack_detected > incorrect > correct. A hack outranks "incorrect" so it is counted in the taxonomy.
* Verifier or infrastructure failures (no CUDA, no CUPTI, harness bug) set `meta.infra_error`. The
  reward is then `None` rather than 0, so the trainer can drop the sample instead of learning noise.

## 2. Correctness

* Shapes: the public shapes from `task.json` plus hidden shapes from `hidden_shapes.json`. Hidden
  sets always contain tiny, non-power-of-2, and at-least-as-large-as-public shapes; the validator
  enforces this.
* Seeds: fresh random seeds per evaluation (default 2 per shape), recorded in the result.
  `correctness.fixed_seeds` makes an evaluation reproducible.
* Reference: computed on upcast inputs (fp16/bf16 to fp32, fp32 to fp64). The comparison is
  elementwise `|out - ref| <= atol + rtol * |ref|` with no mismatch allowance. Shape, dtype, device,
  NaN/Inf, and input mutation are also checked. The expected shape and dtype come from a meta-device
  run, so a correct native-dtype output is never materialized on the GPU.
* Allocator poisoning: before each candidate call, a block of the expected output size is filled
  with 0xFF bytes (NaN for every float dtype) and freed. An output that is allocated but never
  written then reads as NaN instead of stale memory that might happen to be right.
* Task tolerances are validated by checking that the native-dtype PyTorch reference passes them
  against the high-precision reference on every shape (`python -m kernelrl.tasks.validate`).

## 3. Anti-hacking checks and taxonomy

| Key | Layer | Catches |
|---|---|---|
| forbidden_import | static | os, subprocess, ctypes, importlib, time, functools, torch.nn.functional, ... (allowlist: torch, triton, triton.language, math, typing) |
| forbidden_api | static | any `torch.*` or `triton.*` path outside a small allowlist (allocation, dtypes, device queries); covers torch.softmax, torch.compile, torch.ops, torch.cuda.Stream/synchronize, triton.testing |
| forbidden_tensor_method | static | host-code compute methods (`x.sum()`, `x.softmax()`, `x.exp_()`, `.cpu()`, ...) |
| forbidden_operator | static | `@` in host code |
| module_escape | static | module objects used other than as `module.attr` (`t = torch`, `f(torch)`), which would dodge the path allowlist |
| module_attribute_assignment | static | monkeypatching (`torch.softmax = ...`, `torch.cuda.Event.elapsed_time = ...`, `del torch.x`) |
| forbidden_builtin / dunder_access | static | getattr/setattr/exec/eval/open/globals/vars/__import__, any `__dunder__` attribute |
| global_statement | static | `global` / `nonlocal` (explicit cross-call state) |
| no_triton_kernel / missing_entrypoint / entrypoint_signature | static | no `@triton.jit`, no `run`, wrong arity |
| output_aliases_input | runtime | returning an input or a view of one (in-place "kernels") |
| lazy_output | runtime | outputs that change when inputs are modified after the call |
| cached_output | runtime | bit-identical outputs for two inputs whose references differ (memoized or hardcoded results). Requires a computed-looking output (more than one element, finite, non-constant); trivial outputs fall through to `incorrect` |
| output_buffer_reused | runtime | a later call overwrites a previously returned output |
| pointer_keyed_cache | runtime | stale result when the same input buffers are refilled with new contents (same evidence rule as cached_output) |
| state_dependent | runtime | same input correct earlier, wrong after calls are reordered |
| no_triton_launch | runtime | no Triton launch observed via `triton.knobs.runtime.launch_enter_hook` (host arithmetic, skipped work) |
| low_triton_time_fraction | runtime | under 80% of profiled GPU kernel time in the candidate's Triton kernels, on a shape whose output is correct (token kernel plus PyTorch doing the work). Strict by owner decision (2026-10-10): host-side dtype casts count too |
| side_stream_launch | runtime | Triton launch on a non-current stream |
| api_tamper | runtime | any public callable in torch, torch.nn.functional, torch.cuda, torch.Tensor, torch.cuda.Event, time, or the reference module replaced |
| bench_output_incorrect | bench | wrong output on a randomly chosen timed iteration (stops working while benchmarked) |
| timing_mismatch | bench | wall-clock far above CUDA-event time (work escaping the timed stream) |
| reference_timing_perturbed | bench | eager reference in the candidate process is outside [1/2, 2] times its clean-process time |

Static checks use allowlists. Anything not explicitly allowed is rejected, and the runtime checks
assume the static layer can be bypassed. Each hostile fixture marked `SKIP_STATIC` runs with
static checks off to prove this.

## 4. Timing methodology (approved)

* An own implementation, not `triton.testing.do_bench`, so the flush, interleaving, and
  verification are under our control and the timer cannot be patched by the candidate.
* One timed iteration works as follows:
  1. Load input variant `i % 2`, rolled by an `i`-dependent shift. Every call sees unique contents
     (exact for all dtypes, same distribution), so memoizing on pointers or contents cannot pay.
  2. Zero a 256 MiB buffer to flush L2.
  3. Run `start.record(); out = fn(); end.record()`.
* Steps 1 and 2 are enqueued before `start`, so they are not timed.
* Warmup is 25 iterations, then 100 timed iterations. Candidate and eager reference are
  interleaved ABAB/BABA in one process to cancel clock drift. Reported stats are median, IQR, mean,
  std, and min.
* Speedups use eager and `torch.compile` timings from a clean process that never imported the
  candidate, measured with the identical loop. These are cached by reference source, shape, dtype,
  methodology version, GPU, driver, and torch/triton versions. The in-process eager timing only
  serves as a tamper sentinel.
* Aggregate speedup is the geometric mean over public shapes plus hidden shapes marked `bench`.
* Reward: 0 for every failure, `incorrect` configurable (default 0), and correct kernels get
  `0.3 + 0.7 * (clip(log2(speedup_vs_eager), 0, 2) / 2)`. Coefficients live in
  `kernelrl/reward/config.yaml`. torch.compile speedup is reported but not rewarded in Phase 0.

## 5. Feedback format (approved)

Structured text with fixed sections: status, the first error truncated to 1500 characters, the
failing shape, a public-shape pass/fail list, a hidden-shape pass count, and timings vs eager and
torch.compile on public shapes. A profiler section exists as a config toggle
(`feedback.include_profiler`, off) for the Phase 5 ablation.

Hidden shapes never reach the model. Failures on hidden shapes are reported as "a hidden test
shape", and hidden-only dimension values of 10 or more are redacted from error text.
`tests/test_serving_eval.py` checks this for prompts and feedback.

## 6. Known limitations (what the verifier cannot catch yet)

1. **In-process forgery if the static check is bypassed.** The candidate runs in the same
   interpreter as the check harness. Code that escapes the static check, for example through an
   undiscovered reflection path, could overwrite harness state or the result file. Phase 1 fix: run
   the candidate in a grandchild that returns raw outputs over a pipe, and/or nsjail/seccomp with
   no filesystem write access.
2. **Process creation and network are blocked only statically.** There is no `RLIMIT_NPROC`
   (per-user, would break Ray) and no network namespace. The process-group kill reaps children on
   timeout. Phase 1: run workers in containers with `--network none` and a pids limit.
3. **Partial host-side compute.** Host tensor arithmetic (`x + y`) cannot be typed statically. The
   runtime launch check tolerates up to 20% of GPU time outside the candidate's Triton kernels, so a
   kernel can still offload a small part of the work to PyTorch.
4. **Fixed hidden shapes.** Hidden shapes are never shown, but over many RL steps the reward signal
   leaks information about them. Phase 5: sample hidden shapes per evaluation from a distribution.
5. **Data-dependent cheating.** A kernel that is wrong only on rare input patterns, or that detects
   the benchmark by call count and input statistics, may pass. Unique per-call inputs plus verified
   timed iterations make this hard, but they do not prove its absence.
6. **Tolerance exploitation.** Lower-precision math, such as fp16 accumulation or fast
   approximations, is accepted when within tolerance. This is by design, but it is a lever for
   speed that may not reflect a real win.
7. **Benchmark noise.** GPU clocks are not locked; `nvidia-smi -lgc` needs root, so ask first. Only
   the first dtype of a task is benchmarked. The `timing_mismatch` check only fires above a 0.1 ms
   gap, and `reference_timing_perturbed` uses a coarse 2x band to avoid false positives.
8. **Profiler dependence.** The time-fraction check needs CUPTI kernel events. When they are
   missing the evaluation is marked `infra_error`, never silently passed.
9. **GPU recovery after device-side hangs** depends on the driver tearing down the killed context.
   The health probe detects a wedged GPU, but the worker must then be restarted (Phase 1 actor
   restart).
10. **Degenerate shapes give weaker hack evidence.** On 1-element or constant outputs, caching
    cannot be told apart from a bug, so those cases are reported as `incorrect` rather than
    `hack_detected`. Larger shapes in the same evaluation still catch real caches.
11. **Hardware coverage.** Verified on an NVIDIA L40S (Modal, gVisor) with triton 3.6.0 and torch
    2.11.0+cu129. H100 runs are pending; see PROGRESS.md.
