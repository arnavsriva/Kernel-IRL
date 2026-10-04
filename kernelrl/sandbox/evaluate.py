"""Public entry point: evaluate(task_id, source) -> EvalResult.

Runs in the parent (no torch import). Pipeline:
  static (AST) -> compile (child) -> check (child) -> bench (child) [+ clean baselines]
Each child stage is a fresh process with its own hard timeout. This function is what
becomes a Ray actor method in Phase 1.
"""
from __future__ import annotations

import json
import math
import os
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path

from kernelrl.common.io import read_json, truncate
from kernelrl.common.schemas import EvalResult, Finding, ShapeResult, ShapeTiming, Status, TimingStats
from kernelrl.sandbox.baselines import _k, get_baselines
from kernelrl.sandbox.config import load_config
from kernelrl.sandbox.isolation import ChildOutcome, child_env, run_child
from kernelrl.sandbox.static_check import check_source, reference_arity
from kernelrl.tasks.registry import Task, load_task


def _specs(task: Task, shapes: list[dict] | None) -> list[dict]:
    if shapes is not None:
        return shapes
    return [{"dims": s.dims, "hidden": s.hidden, "bench": s.bench} for s in task.all_shapes()]


def _last_progress(job_dir: Path, stage: str) -> dict | None:
    p = job_dir / f"{stage}.progress.jsonl"
    try:
        lines = p.read_text().strip().splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError, json.JSONDecodeError):
        return None


def _run_stage(stage: str, job_dir: Path, cfg: dict, env: dict) -> tuple[dict, ChildOutcome]:
    oc = run_child([sys.executable, "-m", "kernelrl.sandbox.runner", str(job_dir), stage], cwd=job_dir, env=env,
                   timeout_s=cfg["timeouts_s"][stage], rss_limit_bytes=int(cfg["limits"]["host_rss_gb"] * 2**30),
                   poll_s=cfg["limits"]["watchdog_interval_s"], name=stage)
    rp = job_dir / f"{stage}.result.json"
    last = _last_progress(job_dir, stage)
    where = f" (last completed step: {last.get('event')} on {last.get('shape')})" if last else ""
    if oc.timed_out:
        return {"ok": False, "status": "timeout", "reason": f"{stage}_timeout",
                "message": f"{stage} stage exceeded the {cfg['timeouts_s'][stage]}s limit{where}",
                "last_progress": last}, oc
    if oc.rss_exceeded:
        return {"ok": False, "status": "crash", "reason": "host_memory_limit",
                "message": f"host memory exceeded {cfg['limits']['host_rss_gb']} GiB{where}"}, oc
    if rp.exists():
        return read_json(rp), oc
    sig = oc.signal_name
    reason = f"killed_by_{sig}" if sig else f"exit_code_{oc.returncode}"
    if sig == "SIGXFSZ":
        reason = "file_size_limit"
    return {"ok": False, "status": "crash", "reason": reason,
            "message": f"process died ({sig or f'exit code {oc.returncode}'}){where}\n{oc.tail(nbytes=1500)}",
            "last_progress": last}, oc


def _probe(job_dir: Path, cfg: dict, env: dict) -> bool:
    (job_dir / "probe").mkdir(exist_ok=True)
    (job_dir / "probe" / "job.json").write_text(json.dumps({"config": cfg}))
    res, _ = _run_stage("probe", job_dir / "probe", cfg, env)
    return bool(res.get("ok"))


def _fail(result: EvalResult, stage: str, r: dict, max_chars: int) -> EvalResult:
    result.status = Status(r["status"])
    result.reason = r.get("reason", "")
    result.message = truncate(r.get("message", ""), max_chars)
    result.stage = stage
    if r.get("shape") is not None:
        result.meta["failure_shape"] = r["shape"]
        result.meta["failure_hidden"] = bool(r.get("hidden"))
    if r.get("infra_error"):
        result.meta["infra_error"] = True
    return result


def _geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def evaluate(task_id: str, source: str, *, config: dict | None = None, overrides: dict | None = None,
             gpu: str | int | None = None, shapes: list[dict] | None = None, run_bench: bool = True,
             skip_static: bool = False, keep_dir: bool = False, workdir: str | Path | None = None) -> EvalResult:
    """Evaluate one candidate. `skip_static` exists only so tests can prove the runtime
    layers hold even when the static check is bypassed; never use it in training."""
    cfg = config or load_config(overrides=overrides)
    task = load_task(task_id)
    max_chars = cfg["feedback"]["max_error_chars"]
    cc = cfg["correctness"]
    seeds = list(cc["fixed_seeds"]) if cc.get("fixed_seeds") else [secrets.randbelow(2**31 - 1)
                                                                   for _ in range(cc["num_seeds"])]
    order_seed = seeds[0] ^ 0x5EED
    result = EvalResult(task_id=task_id, status=Status.CORRECT, seeds=seeds)
    result.meta["order_seed"] = order_seed

    # ---- 1. static ----------------------------------------------------------------
    t0 = time.perf_counter()
    st = check_source(source, expected_arity=reference_arity(task.reference_source()),
                      allowed_imports=task.allowed_imports, max_bytes=cfg["limits"]["max_source_bytes"])
    result.stage_seconds["static"] = round(time.perf_counter() - t0, 4)
    if not st.ok:
        if st.syntax_error:
            return _fail(result, "static", {"status": "compile_error", "reason": "syntax_error",
                                            "message": st.message}, max_chars)
        result.findings = st.findings
        if not skip_static:
            return _fail(result, "static", {"status": "static_reject", "reason": st.reason,
                                            "message": st.message}, max_chars)
        result.meta["static_bypassed"] = st.reason

    # ---- job directory ------------------------------------------------------------
    job_dir = Path(tempfile.mkdtemp(prefix=f"kernelrl_{task_id}_", dir=workdir))
    try:
        specs = _specs(task, shapes)
        (job_dir / "candidate.py").write_text(source)
        job = {"task_id": task_id, "candidate_file": "candidate.py", "config": cfg, "shapes": specs,
               "dtypes": list(task.dtypes), "tolerance": task.tolerance, "seeds": seeds, "order_seed": order_seed}
        (job_dir / "job.json").write_text(json.dumps(job))
        (job_dir / "tmp").mkdir()
        extra = {"TRITON_CACHE_DIR": str(job_dir / "triton_cache"), "TMPDIR": str(job_dir / "tmp")}
        if gpu is not None:
            extra["CUDA_VISIBLE_DEVICES"] = str(gpu)
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            extra["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        env = child_env(cfg["child_env_passthrough"], extra)

        def stage(name: str) -> dict | None:
            r, oc = _run_stage(name, job_dir, cfg, env)
            result.stage_seconds[name] = round(oc.wall_s, 3)
            if not r.get("ok"):
                if r.get("status") in ("timeout", "crash") and cfg.get("probe_after_failure"):
                    result.meta["gpu_healthy_after"] = _probe(job_dir, cfg, env)
                return r
            return None

        # ---- 2. compile ------------------------------------------------------------
        if (r := stage("compile")) is not None:
            return _fail(result, "compile", r, max_chars)

        # ---- 3. check --------------------------------------------------------------
        r_check, oc = _run_stage("check", job_dir, cfg, env)
        result.stage_seconds["check"] = round(oc.wall_s, 3)
        result.shape_results = [ShapeResult(**s) for s in r_check.get("shape_results", [])]
        result.findings += [Finding(**f) for f in r_check.get("findings", [])]
        result.meta["launch_info"] = r_check.get("launch_info")
        if not r_check.get("ok"):
            if r_check.get("status") in ("timeout", "crash") and cfg.get("probe_after_failure"):
                result.meta["gpu_healthy_after"] = _probe(job_dir, cfg, env)
            return _fail(result, "check", r_check, max_chars)
        hacks = [f for f in result.findings if f.check not in _STATIC_KEYS]
        if hacks:
            return _fail(result, "check", {"status": "hack_detected", "reason": hacks[0].check,
                                           "message": hacks[0].message, "shape": hacks[0].shape,
                                           "hidden": hacks[0].hidden}, max_chars)
        failed = [s for s in result.shape_results if not s.passed]
        if failed:
            f0 = failed[0]
            return _fail(result, "check", {"status": "incorrect", "reason": (f0.error or "mismatch").split(":")[0],
                                           "message": f0.error or "", "shape": f0.shape, "hidden": f0.hidden},
                         max_chars)

        # ---- 4. bench --------------------------------------------------------------
        if not run_bench:
            result.status = Status.CORRECT
            result.stage = "check"
            return result
        r_bench, oc = _run_stage("bench", job_dir, cfg, env)
        result.stage_seconds["bench"] = round(oc.wall_s, 3)
        if not r_bench.get("ok"):
            return _fail(result, "bench", r_bench, max_chars)
        bench_findings = [Finding(**f) for f in r_bench.get("findings", [])]
        t1 = time.perf_counter()
        base = get_baselines(task, specs, cfg, gpu=None if gpu is None else str(gpu))
        result.stage_seconds["baseline"] = round(time.perf_counter() - t1, 3)
        lim = cfg["benchmark"]["inproc_reference_ratio_limit"]
        for t in r_bench["timings"]:
            b = base.get(_k(t["shape"]), {})
            st_ = ShapeTiming(shape=t["shape"], dtype=t["dtype"], hidden=t["hidden"],
                              candidate=TimingStats(**t["candidate"]), eager_inproc=TimingStats(**t["eager_inproc"]),
                              eager=TimingStats(**b["eager"]) if b.get("eager") else None,
                              compiled=TimingStats(**b["compiled"]) if b.get("compiled") else None,
                              wall_over_event=t["wall_over_event"])
            c = st_.candidate.median_ms
            if st_.eager:
                st_.speedup_vs_eager = st_.eager.median_ms / c if c > 0 else None
                ratio = st_.eager_inproc.median_ms / st_.eager.median_ms
                if not (1 / lim <= ratio <= lim):
                    bench_findings.append(Finding("reference_timing_perturbed",
                                                  f"reference ran {ratio:.2f}x its clean-process time next to the "
                                                  "candidate", t["shape"], t["hidden"]))
            if st_.compiled:
                st_.speedup_vs_compile = st_.compiled.median_ms / c if c > 0 else None
            result.timings.append(st_)
        result.findings += bench_findings
        result.speedup_vs_eager = _geomean([t.speedup_vs_eager for t in result.timings])
        result.speedup_vs_compile = _geomean([t.speedup_vs_compile for t in result.timings])
        if bench_findings:
            f = bench_findings[0]
            return _fail(result, "bench", {"status": "hack_detected", "reason": f.check, "message": f.message,
                                           "shape": f.shape, "hidden": f.hidden}, max_chars)
        result.status = Status.CORRECT
        result.stage = "bench"
        return result
    finally:
        result.meta["job_dir"] = str(job_dir) if keep_dir else None
        if not keep_dir:
            shutil.rmtree(job_dir, ignore_errors=True)


# Static-check reasons can appear in findings when skip_static is used; they are not runtime hacks.
_STATIC_KEYS = frozenset({
    "forbidden_import", "forbidden_api", "forbidden_builtin", "dunder_access", "module_escape",
    "module_attribute_assignment", "forbidden_attribute_assignment", "global_statement", "forbidden_tensor_method",
    "forbidden_operator", "missing_entrypoint", "entrypoint_signature", "no_triton_kernel",
})
