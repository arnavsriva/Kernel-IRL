"""Child-process entry point: python -m kernelrl.sandbox.runner <job_dir> <stage>

Stages: compile | check | bench | baseline | probe. Always writes <stage>.result.json
(atomically) unless the process is killed; the parent treats a missing result as a crash.
"""
from __future__ import annotations

import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path


def _set_rlimits(limits: dict) -> None:
    fsize = int(limits["max_file_bytes"])
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _setup_torch():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the sandbox child")
    torch.cuda.set_device(0)
    return torch


def _import_candidate(path: Path):
    from kernelrl.sandbox.child_utils import StageFailure, format_exception

    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        raise StageFailure("compile_error", "import_error", format_exception(e, str(path))) from None
    fn = getattr(mod, "run", None)
    if not callable(fn):
        raise StageFailure("compile_error", "missing_entrypoint", "module does not define a callable run()")
    return fn


def _compile_stage(job, run_fn, ref_mod, progress, cand_path) -> dict:
    """First call on every shape/dtype: triggers Triton JIT + autotune for each specialization."""
    import torch

    from kernelrl.sandbox.child_utils import StageFailure, classify_exception, format_exception
    from kernelrl.sandbox.compare import DTYPES

    per_shape = []
    for spec in job["shapes"]:
        for dt in job["dtypes"]:
            inputs = ref_mod.make_inputs(spec["dims"], DTYPES[dt], "cuda", 0)
            t0 = time.perf_counter()
            try:
                run_fn(*inputs)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                status, reason = classify_exception(e)
                raise StageFailure(status, reason, format_exception(e, cand_path), spec) from None
            dt_s = time.perf_counter() - t0
            per_shape.append({"shape": spec["dims"], "dtype": dt, "seconds": round(dt_s, 4)})
            progress.log(event="compiled", shape=spec["dims"], dtype=dt, seconds=dt_s)
    return {"ok": True, "per_shape": per_shape}


def _probe() -> dict:
    import torch

    from kernelrl.common.env_fingerprint import runtime_fingerprint

    a = torch.randn(1024, 1024, device="cuda")
    v = float((a @ a).sum())
    torch.cuda.synchronize()
    return {"ok": v == v, "runtime": runtime_fingerprint()}


def main() -> None:
    job_dir, stage = Path(sys.argv[1]), sys.argv[2]
    job = json.loads((job_dir / "job.json").read_text())
    _set_rlimits(job["config"]["limits"])
    t0 = time.perf_counter()
    result: dict
    try:
        _setup_torch()
        from kernelrl.common.io import atomic_write_json  # noqa: F401  (import before candidate)
        from kernelrl.sandbox.child_utils import ApiSentinel, Progress, StageFailure

        try:
            if stage == "probe":
                result = _probe()
            else:
                from kernelrl.tasks.registry import load_task

                ref_mod = load_task(job["task_id"]).load_module()
                progress = Progress(job_dir / f"{stage}.progress.jsonl")
                if stage == "baseline":
                    from kernelrl.sandbox.benchmark import run_baseline_stage

                    result = run_baseline_stage(job, ref_mod, progress)
                else:
                    from kernelrl.sandbox import benchmark, checks  # import harness before candidate code

                    sentinel = ApiSentinel(extra_modules=(ref_mod,))
                    cand_path = str(job_dir / job["candidate_file"])
                    run_fn = _import_candidate(Path(cand_path))
                    if stage == "compile":
                        result = _compile_stage(job, run_fn, ref_mod, progress, cand_path)
                    elif stage == "check":
                        ctx = checks.Ctx(job, run_fn, ref_mod, progress, cand_path)
                        try:
                            result = checks.run_check_stage(ctx, sentinel)
                        except StageFailure as f:
                            result = f.to_result()
                            result["findings"] = ctx.findings  # keep hack evidence gathered before the failure
                            result["shape_results"] = ctx.shape_results
                    elif stage == "bench":
                        result = benchmark.run_bench_stage(job, run_fn, ref_mod, progress, sentinel, cand_path)
                    else:
                        raise ValueError(f"unknown stage {stage}")
        except StageFailure as f:
            result = f.to_result()
    except BaseException as e:  # noqa: BLE001
        # CUDA errors are asynchronous: a bad candidate kernel can surface in harness code
        # (clone/compare). Those are the candidate's fault; anything else is infrastructure.
        reason, infra = "harness_exception", True
        try:
            from kernelrl.sandbox.child_utils import classify_exception

            _, r = classify_exception(e)
            if r in ("cuda_error", "cuda_oom"):
                reason, infra = r, False
        except Exception:  # noqa: BLE001
            pass
        result = {"ok": False, "status": "crash", "reason": reason, "infra_error": infra,
                  "message": "".join(traceback.format_exception(e))[-4000:]}
    result["stage"] = stage
    result["seconds"] = round(time.perf_counter() - t0, 3)
    from kernelrl.common.io import atomic_write_json

    atomic_write_json(job_dir / f"{stage}.result.json", result)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip interpreter teardown (a corrupted CUDA context can hang it)


if __name__ == "__main__":
    main()
