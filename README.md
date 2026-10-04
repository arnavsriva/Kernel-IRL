# KernelRL

Multi-turn RL that teaches open LLMs (Qwen3 4B to 8B to 14B) to write fast, correct Triton kernels.
See `CLAUDE.md` for goals and phases, `docs/DESIGN.md` for the verifier, and `docs/PROGRESS.md`
for status.

## Layout
- `kernelrl/tasks`: 30 tasks (task.json, reference.py, hidden_shapes.json) and a validator.
- `kernelrl/sandbox`: static check, isolated compile/check/bench stages, baselines, and feedback.
  `evaluate()` is the entry point.
- `kernelrl/reward`: correctness-gated clipped log-speedup reward (`config.yaml`).
- `kernelrl/serving`: SGLang launch script, prompts, code extraction, and client.
- `kernelrl/eval`: pass@k baseline, 1-turn vs 2-turn comparison, and reports.
- `kernelrl/bench`: prefix-caching (radix cache) benchmark.
- `tests`: adversarial fixtures (good, buggy, hacks, hostile) and tests. `tests/gpu` needs CUDA.

## Local (CPU) development

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[cpu,dev]" openai==2.6.1
.venv/bin/python -m pytest -q
.venv/bin/python -m kernelrl.tasks.validate --device cpu --max-work 2e9
```

## GPU box (Linux and NVIDIA)

```bash
bash scripts/setup_gpu_box.sh docker --yes-large-download   # or: venv
bash scripts/run_phase0_gpu.sh                              # validate, verdicts, eval, bench
```
