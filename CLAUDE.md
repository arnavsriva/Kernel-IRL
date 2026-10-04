# KernelRL

## Project goal
Train an open LLM (Qwen3 4B → 8B → 14B) to write optimized Triton kernels via multi-turn RL.
Each turn: model writes a kernel → sandbox compiles, checks correctness, profiles → feedback returns
to the model for revision. Reward = correctness-gated measured speedup.

The project exists to build deep practical skill in four systems, so each must be used for real:
- SGLang: rollout/serving engine (multi-turn, prefix caching, weight updates)
- Ray: fleet of isolated reward workers (compile/test/benchmark kernels on dedicated GPUs)
- Megatron-LM: policy training (TP/PP/CP at 8B–14B)
- slime: RL framework tying them together (custom generate fn, custom reward, async training)

## Research contributions (keep these in mind when designing anything)
1. Reward-hacking defense: a hardened verifier + a taxonomy of hacks observed during training.
2. Systems study: sync vs async, colocated vs disaggregated, staleness tolerance, reward-worker scaling.
3. Profiler-in-the-loop feedback: does richer multi-turn feedback improve sample efficiency?

## Phases
- Phase 0: task set + sandbox/verifier + SGLang baseline (1 GPU)   ← CURRENT
- Phase 1: Ray reward-worker fleet
- Phase 2: slime integration, single-turn GRPO (4B)
- Phase 3: multi-turn + async training
- Phase 4: scale to 8B/14B with Megatron parallelism tuning
- Phase 5: reward-hacking study + systems ablations + write-up

## Engineering rules
- The verifier is the foundation. A noisy or exploitable reward ruins every later phase.
  Never weaken a verifier check to make a test pass. Flag it to me instead.
- Every kernel evaluation runs in a fresh subprocess with a hard timeout. Bad kernels hang,
  segfault, or corrupt the CUDA context; the parent process must survive all of these.
- Pin all dependency versions. Before choosing SGLang/torch/triton versions, check what
  slime's current official Docker image uses so Phase 2 doesn't force a migration.
- Every measurement is reproducible: fixed seeds where applicable, and results written as JSON
  with the git commit hash, GPU model, driver, and library versions.
- Keep modules decoupled: the sandbox must not import anything from serving or training code,
  so it can become a Ray actor in Phase 1 unchanged.
- Small, frequent commits with descriptive messages. Update docs/PROGRESS.md at the end of each session.
- Ask before downloading anything larger than 20 GB or installing system-level packages.
- When unsure about an API (SGLang flags, Triton features), check the installed version's
  docs/source rather than guessing.

## Repo layout
kernelrl/
  tasks/          # task definitions (JSON + reference PyTorch code)
  sandbox/        # compile / correctness / benchmark / anti-hacking checks
  reward/         # reward function built on sandbox results
  serving/        # SGLang launch scripts, prompt templates, code extraction
  eval/           # baseline evals, pass@k, reports
  bench/          # systems benchmarks (prefix caching, throughput)
tests/            # including adversarial "hacking kernel" fixtures
docs/             # PROGRESS.md, DESIGN.md, results
scripts/