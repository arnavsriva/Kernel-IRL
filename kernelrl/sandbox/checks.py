"""Check stage (runs in a fresh child): correctness on public + hidden shapes with fresh
seeds, plus anti-hacking checks. Every finding uses a stable taxonomy key (Finding.check)."""
from __future__ import annotations

import random

import torch

from kernelrl.sandbox.child_utils import (
    StageFailure, classify_exception, expected_output_meta, format_exception, nbytes_of, poison_allocator,
    record_triton_launches, reference_hi,
)
from kernelrl.sandbox.compare import DTYPES, bitwise_equal, clone_inputs, compare, shares_storage

DEVICE = "cuda"


class Ctx:
    def __init__(self, job, run_fn, ref_mod, progress, candidate_path):
        self.job, self.run, self.ref, self.progress = job, run_fn, ref_mod, progress
        self.cfg = job["config"]
        self.cand_path = candidate_path
        self.findings: list[dict] = []
        self.shape_results: list[dict] = []

    def finding(self, check: str, message: str, spec: dict) -> None:
        self.findings.append({"check": check, "message": message, "shape": spec["dims"],
                              "hidden": bool(spec.get("hidden"))})
        self.progress.log(event="finding", check=check, shape=spec["dims"])

    def tol(self, dtype_name: str) -> dict:
        return self.job["tolerance"][dtype_name]

    def inputs(self, spec, dtype_name, seed):
        return self.ref.make_inputs(spec["dims"], DTYPES[dtype_name], DEVICE, seed)

    def call(self, inputs, spec):
        """Call the candidate; candidate exceptions end the stage with a classified status."""
        try:
            out = self.run(*inputs)
            torch.cuda.synchronize()
            return out
        except Exception as e:  # noqa: BLE001
            status, reason = classify_exception(e)
            raise StageFailure(status, reason, format_exception(e, self.cand_path), spec) from None

    def poison(self, shape, dtype):
        if self.cfg["correctness"]["poison_allocator"]:
            poison_allocator(nbytes_of(shape, dtype))


def _derived_seed(base: int, tag: int) -> int:
    return (base * 1_000_003 + tag * 7919 + 17) % (2**31 - 1)


def check_one(ctx: Ctx, spec, dtype_name, seed, *, mutate_after: bool, fps: list | None = None) -> dict:
    """One correctness run. Returns a ShapeResult dict; adds hack findings to ctx.
    If `fps` is given, the output's fingerprint is appended to it."""
    inputs = ctx.inputs(spec, dtype_name, seed)
    snap = clone_inputs(inputs)
    shape, odtype = expected_output_meta(ctx.ref.reference, inputs)
    ctx.poison(shape, odtype)
    out = ctx.call(inputs, spec)
    aliased = isinstance(out, torch.Tensor) and any(shares_storage(out, x) for x in inputs)
    if aliased:
        ctx.finding("output_aliases_input", "returned tensor shares memory with an input", spec)
    mutated = not all(bitwise_equal(x, s) for x, s in zip(inputs, snap) if isinstance(x, torch.Tensor))
    if mutate_after and isinstance(out, torch.Tensor) and not aliased and not mutated:
        out_snap = out.clone()
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.add_(1) if x.is_floating_point() else x.zero_()
        torch.cuda.synchronize()
        if not bitwise_equal(out, out_snap):
            ctx.finding("lazy_output", "output changed after the inputs were modified (lazy view of inputs)", spec)
        out = out_snap
    if fps is not None:
        fps.append(fingerprint(out))
    hi = reference_hi(ctx.ref.reference, snap)
    c = compare(out, hi, shape, odtype, torch.device(DEVICE, torch.cuda.current_device()), **ctx.tol(dtype_name))
    if c["passed"] and mutated:
        c["passed"], c["error"] = False, "input_mutated: run() modified its input tensors"
    return {"shape": spec["dims"], "dtype": dtype_name, "seed": seed, "hidden": bool(spec.get("hidden")),
            "passed": c["passed"], "max_abs_err": c["max_abs_err"], "max_rel_err": c["max_rel_err"],
            "mismatch_frac": c["mismatch_frac"], "error": c["error"]}


def fingerprint(t) -> tuple | None:
    """Cheap content fingerprint of an output (shape, dtype, plain sum, position-weighted sum)."""
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return None
    v = t.detach().double().flatten().nan_to_num(nan=1.2345e300)  # NaN-safe so garbage can still match itself
    w = (torch.arange(v.numel(), device=v.device, dtype=torch.float64) % 7919) + 1
    return (tuple(t.shape), str(t.dtype), float(v.sum()), float((v * w).sum()))


def classify_reorder_failure(fp_now, fps_other_inputs: list) -> str:
    """A wrong result on the second, reordered call with the same input is either a replay of an
    output produced for a *different* input (state carried between calls: a hack) or fresh garbage
    (out-of-bounds reads, races: an honest bug). Found on Modal: a row-max kernel without a column
    mask read past the end of a 1x1 tensor and was mislabelled state_dependent."""
    return "replay" if fp_now is not None and fp_now in fps_other_inputs else "nondeterministic"


def _looks_computed(t) -> bool:
    """True for outputs that look like real results rather than bug artifacts: finite (not
    poisoned / never written) and non-uniform (not all-zero). Bit-identical outputs of this kind
    for inputs whose references differ are a replayed (cached) result, not an ordinary bug."""
    return (isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel() > 1
            and bool(torch.isfinite(t).all()) and bool(t.max() != t.min()))


def distinct_inputs_check(ctx: Ctx, spec, dtype_name, base_seed) -> None:
    sa, sb = _derived_seed(base_seed, 1), _derived_seed(base_seed, 2)
    xa, xb = ctx.inputs(spec, dtype_name, sa), ctx.inputs(spec, dtype_name, sb)
    shape, odtype = expected_output_meta(ctx.ref.reference, xa)
    snap_a, snap_b = clone_inputs(xa), clone_inputs(xb)
    ctx.poison(shape, odtype)
    ca = ctx.call(xa, spec)
    if not isinstance(ca, torch.Tensor):
        return
    ca_snap = ca.clone()
    ctx.poison(shape, odtype)
    cb = ctx.call(xb, spec)
    hia, hib = reference_hi(ctx.ref.reference, snap_a), reference_hi(ctx.ref.reference, snap_b)
    if bitwise_equal(hia, hib):
        return  # degenerate shape: different inputs, same output
    dev = torch.device(DEVICE, torch.cuda.current_device())
    a_ok = compare(ca_snap, hia, shape, odtype, dev, **ctx.tol(dtype_name))["passed"]
    # Require a computed-looking output: a 1-element or constant output (e.g. a bug that writes
    # zeros) can be identical across inputs, and even within tolerance, without any caching.
    if bitwise_equal(cb, ca_snap) and _looks_computed(ca_snap):
        ctx.finding("cached_output", "identical output for two different inputs (cached or hardcoded result)", spec)
    elif a_ok and not bitwise_equal(ca, ca_snap):
        ctx.finding("output_buffer_reused", "a previously returned output was overwritten by a later call", spec)


def same_address_check(ctx: Ctx, spec, dtype_name, base_seed) -> dict | None:
    sc, sd = _derived_seed(base_seed, 3), _derived_seed(base_seed, 4)
    x = ctx.inputs(spec, dtype_name, sc)
    orig = clone_inputs(x)
    shape, odtype = expected_output_meta(ctx.ref.reference, x)
    ctx.poison(shape, odtype)
    c1 = ctx.call(x, spec)
    if not isinstance(c1, torch.Tensor):
        return None
    c1s = c1.clone()
    new = ctx.inputs(spec, dtype_name, sd)
    for xi, ni in zip(x, new):
        if isinstance(xi, torch.Tensor):
            xi.copy_(ni)  # same pointers, new contents
    torch.cuda.synchronize()
    ctx.poison(shape, odtype)
    c2 = ctx.call(x, spec)
    hi1, hi2 = reference_hi(ctx.ref.reference, orig), reference_hi(ctx.ref.reference, new)
    dev = torch.device(DEVICE, torch.cuda.current_device())
    tol = ctx.tol(dtype_name)
    r2 = compare(c2, hi2, shape, odtype, dev, **tol)
    if r2["passed"]:
        return None
    # Same evidence rule as cached_output: a trivial output (found on M=N=K=1, where a constant 0 was
    # within tolerance of the reference) cannot distinguish a pointer-keyed cache from a plain bug.
    if _looks_computed(c1s) and not bitwise_equal(hi1, hi2) and bitwise_equal(c2, c1s):
        ctx.finding("pointer_keyed_cache", "stale result returned when input buffers were reused with new contents", spec)
        return None
    return {"shape": spec["dims"], "dtype": dtype_name, "seed": sd, "hidden": bool(spec.get("hidden")),
            "passed": False, "max_abs_err": r2["max_abs_err"], "max_rel_err": r2["max_rel_err"],
            "mismatch_frac": r2["mismatch_frac"],
            "error": f"wrong result on additional random inputs (written into the same input buffers): {r2['error']}"}


def _kernel_events(prof):
    from torch.autograd import DeviceType

    evs = []
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            evs.append((e.name, e.time_range.elapsed_us()))
    return evs


def launch_check(ctx: Ctx, spec, dtype_name, seed) -> dict:
    """A Triton kernel must be launched, on the current stream, and own most GPU time."""
    from torch.profiler import ProfilerActivity, profile

    hc = ctx.cfg["hack_checks"]
    x = ctx.inputs(spec, dtype_name, seed)
    ctx.call(clone_inputs(x), spec)  # warm (autotune / JIT) outside the profiled region
    torch.cuda.synchronize()
    with record_triton_launches() as launches:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            ctx.call(x, spec)
            torch.cuda.synchronize()
    names = {n for n, _ in launches if n}
    cur = torch.cuda.current_stream().cuda_stream
    if any(s is not None and s != cur for _, s in launches):
        ctx.finding("side_stream_launch", "a Triton kernel was launched on a non-current stream", spec)
    evs = _kernel_events(prof)
    total = sum(t for _, t in evs)
    triton_t = sum(t for n, t in evs if n in names)
    info = {"shape": spec["dims"], "triton_kernels": sorted(names), "n_launches": len(launches),
            "gpu_kernels": len(evs), "gpu_us": total, "triton_us": triton_t}
    if not launches:
        ctx.finding("no_triton_launch", "no Triton kernel was launched", spec)
        return info
    if not evs:
        if hc["require_profiler"]:
            raise RuntimeError("profiler returned no CUDA kernel events (CUPTI unavailable?)")
        return info
    frac = triton_t / total if total > 0 else 0.0
    info["triton_fraction"] = frac
    if frac < hc["min_triton_time_fraction"]:
        others = sorted({n for n, _ in evs if n not in names})[:5]
        ctx.finding("low_triton_time_fraction",
                    f"only {frac:.0%} of GPU time is in the candidate's Triton kernels "
                    f"(other kernels: {', '.join(others)})", spec)
    return info


def run_check_stage(ctx: Ctx, sentinel) -> dict:
    job, hc = ctx.job, ctx.cfg["hack_checks"]
    specs, dtypes, seeds = job["shapes"], job["dtypes"], job["seeds"]
    items = [(i, dt, s) for i in range(len(specs)) for dt in dtypes for s in seeds]
    order = items[:]
    random.Random(job["order_seed"]).shuffle(order)

    passed1, fps1 = {}, {}
    for (i, dt, s) in order:
        fp: list = []
        sr = check_one(ctx, specs[i], dt, s, mutate_after=hc["mutation_after_call"], fps=fp)
        fps1[(i, dt, s)] = fp[0] if fp else None
        ctx.shape_results.append(sr)
        passed1[(i, dt, s)] = sr["passed"]
        ctx.progress.log(event="correctness", shape=specs[i]["dims"], dtype=dt, passed=sr["passed"])

    base = seeds[0]
    dt0 = dtypes[0]
    for i, spec in enumerate(specs):
        if hc["distinct_inputs"]:
            distinct_inputs_check(ctx, spec, dt0, base + i)
        if hc["same_address_reuse"]:
            r = same_address_check(ctx, spec, dt0, base + i)
            if r:
                ctx.shape_results.append(r)
    if hc["call_order_permutation"]:
        for (i, dt, s) in reversed(order):
            fp: list = []
            sr = check_one(ctx, specs[i], dt, s, mutate_after=False, fps=fp)
            if passed1[(i, dt, s)] and not sr["passed"]:
                others = [v for (j, d, s2), v in fps1.items() if j == i and d == dt and s2 != s]
                if classify_reorder_failure(fp[0] if fp else None, others) == "replay":
                    ctx.finding("state_dependent", "after calls were reordered, the same input returned the "
                                "output previously produced for a different input (state carried between calls)",
                                specs[i])
                else:
                    sr["error"] = ("nondeterministic_result: the same input gave a correct result on one call and a "
                                   f"wrong one on another (check for out-of-bounds loads/missing masks or races): "
                                   f"{sr['error']}")
                    ctx.shape_results.append(sr)
    launch_info = []
    if hc["launch_check"]:
        for spec in specs:
            launch_info.append(launch_check(ctx, spec, dt0, base))
    if hc["api_tamper_sentinel"]:
        changed = sentinel.changed()
        if changed:
            ctx.findings.append({"check": "api_tamper", "message": f"replaced: {', '.join(changed[:8])}",
                                 "shape": None, "hidden": False})
    return {"ok": True, "shape_results": ctx.shape_results, "findings": ctx.findings, "launch_info": launch_info}
