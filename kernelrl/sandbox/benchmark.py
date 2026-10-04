"""Timing (runs in fresh children).

Methodology (one timed iteration):
    copy input variant (i % P) into the working buffers   # untimed; defeats caching keyed on pointers
    flush L2 by writing a 256 MiB buffer                   # untimed, but enqueued before `start`
    start.record(); out = fn(); end.record()
Candidate and eager reference are interleaved ABAB/BABA in the same process to cancel clock
drift; this in-process reference is a tamper sentinel. Reported speedups use reference timings
from a clean process that never imported the candidate (baseline stage), measured with the
identical loop. A few randomly chosen timed outputs are verified against the reference.
"""
from __future__ import annotations

import random
import statistics
import time

import torch

from kernelrl.sandbox.child_utils import (
    StageFailure, classify_exception, expected_output_meta, format_exception, reference_hi,
)
from kernelrl.sandbox.compare import DTYPES, clone_inputs, compare

DEVICE = "cuda"


def stats(ms: list[float]) -> dict:
    q = statistics.quantiles(ms, n=4) if len(ms) >= 2 else [ms[0]] * 3
    return {"median_ms": statistics.median(ms), "iqr_ms": q[2] - q[0], "mean_ms": statistics.fmean(ms),
            "std_ms": statistics.pstdev(ms), "min_ms": min(ms), "n": len(ms)}


class Timed:
    """A function plus its own working input buffers and input-variant pool."""

    def __init__(self, fn, variants):
        self.fn = fn
        self.variants = variants
        self.work = clone_inputs(variants[0])

    def prepare(self, i: int) -> None:
        src = self.variants[i % len(self.variants)]
        for dst, s in zip(self.work, src):
            if isinstance(dst, torch.Tensor):
                dst.copy_(s)

    def __call__(self):
        return self.fn(*self.work)


def timed_loop(fns: list[Timed], *, warmup: int, iters: int, flush: torch.Tensor,
               keep: dict[int, set[int]] | None = None):
    """Interleaved timing of 1+ functions. Returns per-fn ms lists and kept outputs."""
    keep = keep or {}
    for i in range(warmup):
        for f in fns:
            f.prepare(i)
            f()
    torch.cuda.synchronize()
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    starts = [[ev() for _ in range(iters)] for _ in fns]
    ends = [[ev() for _ in range(iters)] for _ in fns]
    kept: dict[int, dict[int, torch.Tensor]] = {k: {} for k in range(len(fns))}
    for i in range(iters):
        order = range(len(fns)) if i % 2 == 0 else reversed(range(len(fns)))
        for k in order:
            f = fns[k]
            f.prepare(i)
            flush.zero_()
            starts[k][i].record()
            out = f()
            ends[k][i].record()
            if i in keep.get(k, ()):
                kept[k][i] = out.clone() if isinstance(out, torch.Tensor) else out
    torch.cuda.synchronize()
    times = [[s.elapsed_time(e) for s, e in zip(starts[k], ends[k])] for k in range(len(fns))]
    return times, kept


def wall_clock_ms(f: Timed, iters: int) -> list[float]:
    f.prepare(0)
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _variants(ref_mod, spec, dtype_name, base_seed, n):
    return [ref_mod.make_inputs(spec["dims"], DTYPES[dtype_name], DEVICE, base_seed + v) for v in range(n)]


def run_bench_stage(job, run_fn, ref_mod, progress, sentinel, candidate_path) -> dict:
    cfg = job["config"]
    bc = cfg["benchmark"]
    flush = torch.empty(bc["l2_flush_bytes"], dtype=torch.uint8, device=DEVICE)
    rng = random.Random(job["order_seed"] + 99)
    dt = job["dtypes"][0]
    dev = torch.device(DEVICE, torch.cuda.current_device())
    timings, findings = [], []
    for spec in [s for s in job["shapes"] if s.get("bench")]:
        variants = _variants(ref_mod, spec, dt, bc["bench_seed"], bc["input_variants"])
        his = [reference_hi(ref_mod.reference, v) for v in variants]
        shape, odtype = expected_output_meta(ref_mod.reference, variants[0])
        cand = Timed(run_fn, variants)
        eager = Timed(ref_mod.reference, variants)
        keep_iters = set(rng.sample(range(bc["iters"]), min(bc["verify_iters"], bc["iters"])))
        try:
            with torch.no_grad():
                (tc, te), kept = timed_loop([cand, eager], warmup=bc["warmup"], iters=bc["iters"], flush=flush,
                                            keep={0: keep_iters})
            wall = wall_clock_ms(cand, bc["wall_iters"])
        except Exception as e:  # noqa: BLE001
            status, reason = classify_exception(e)
            raise StageFailure(status, reason, format_exception(e, candidate_path), spec) from None
        for i, out in kept[0].items():
            c = compare(out, his[i % len(variants)], shape, odtype, dev, **job["tolerance"][dt])
            if not c["passed"]:
                findings.append({"check": "bench_output_incorrect", "hidden": bool(spec.get("hidden")),
                                 "shape": spec["dims"],
                                 "message": f"output of timed iteration {i} was wrong: {c['error']}"})
                break
        cs, es, ws = stats(tc), stats(te), statistics.median(wall)
        ratio = ws / cs["median_ms"] if cs["median_ms"] > 0 else float("inf")
        if ratio > bc["wall_over_event_limit"] and ws - cs["median_ms"] > bc["wall_over_event_min_ms"]:
            findings.append({"check": "timing_mismatch", "hidden": bool(spec.get("hidden")), "shape": spec["dims"],
                             "message": f"wall-clock {ws:.3f} ms vs CUDA-event {cs['median_ms']:.3f} ms "
                                        "(work escaping the timed stream?)"})
        timings.append({"shape": spec["dims"], "dtype": dt, "hidden": bool(spec.get("hidden")),
                        "candidate": cs, "eager_inproc": es, "wall_over_event": ratio})
        progress.log(event="bench", shape=spec["dims"], cand_ms=cs["median_ms"], eager_ms=es["median_ms"])
        del variants, his, cand, eager, kept
    changed = sentinel.changed() if cfg["hack_checks"]["api_tamper_sentinel"] else []
    if changed:
        findings.append({"check": "api_tamper", "message": f"replaced: {', '.join(changed[:8])}",
                         "shape": None, "hidden": False})
    return {"ok": True, "timings": timings, "findings": findings}


def run_baseline_stage(job, ref_mod, progress) -> dict:
    """Eager and torch.compile reference timings in a process that never sees a candidate."""
    cfg = job["config"]
    bc = cfg["benchmark"]
    flush = torch.empty(bc["l2_flush_bytes"], dtype=torch.uint8, device=DEVICE)
    dt = job["dtypes"][0]
    out = []
    for spec in [s for s in job["shapes"] if s.get("bench")]:
        variants = _variants(ref_mod, spec, dt, bc["bench_seed"], bc["input_variants"])
        rec = {"shape": spec["dims"], "dtype": dt, "eager": None, "compiled": None, "compile_error": None}
        with torch.no_grad():
            (te,), _ = timed_loop([Timed(ref_mod.reference, variants)], warmup=bc["warmup"], iters=bc["iters"],
                                  flush=flush)
            rec["eager"] = stats(te)
            try:
                torch._dynamo.reset()
                cfn = torch.compile(ref_mod.reference, mode=bc["compile_mode"], dynamic=False)
                (tc,), _ = timed_loop([Timed(cfn, variants)], warmup=bc["warmup"], iters=bc["iters"], flush=flush)
                rec["compiled"] = stats(tc)
            except Exception as e:  # noqa: BLE001
                rec["compile_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        out.append(rec)
        progress.log(event="baseline", shape=spec["dims"], eager_ms=rec["eager"]["median_ms"],
                     compiled_ms=(rec["compiled"] or {}).get("median_ms"))
    return {"ok": True, "baselines": out}
