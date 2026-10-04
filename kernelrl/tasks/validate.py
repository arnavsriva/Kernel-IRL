"""Validate every task: references run on all public + hidden shapes, inputs are
deterministic, the native-dtype reference passes the task's own tolerance against the
high-precision reference (i.e. tolerances are achievable), and hidden sets contain
tiny / non-power-of-2 / large shapes.

  python -m kernelrl.tasks.validate --device cuda
  python -m kernelrl.tasks.validate --device cpu --max-work 1e9   # local smoke run
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import torch

from kernelrl.common.env_fingerprint import stamp
from kernelrl.common.io import atomic_write_json
from kernelrl.sandbox.compare import DTYPES, bitwise_equal, compare, reference_outputs
from kernelrl.tasks.registry import Task, all_tasks, load_task


def _work(dims: dict) -> int:
    return math.prod(dims.values())


def _is_pow2(n: int) -> bool:
    return n & (n - 1) == 0


def coverage_problems(task: Task) -> list[str]:
    hidden = task.hidden_shapes()
    probs = []
    if not any(any(not _is_pow2(v) and v > 1 for v in s.dims.values()) for s in hidden):
        probs.append("hidden set has no non-power-of-2 shape")
    if not any(_work(s.dims) <= 64 for s in hidden):
        probs.append("hidden set has no tiny shape (product of dims <= 64)")
    max_pub = max(_work(s) for s in task.public_shapes)
    if not any(_work(s.dims) >= max_pub for s in hidden):
        probs.append("hidden set has no shape at least as large as the largest public shape")
    if not any(s.bench for s in hidden):
        probs.append("hidden set has no bench shape")
    return probs


def validate_task(task: Task, device: str, max_work: float | None, seeds=(0, 1)) -> dict:
    mod = task.load_module()
    rec = {"task": task.id, "problems": coverage_problems(task), "shapes": [], "skipped": []}
    for dtype_name in task.dtypes:
        dtype = DTYPES[dtype_name]
        tol = task.tolerance[dtype_name]
        for spec in task.all_shapes():
            if max_work is not None and _work(spec.dims) > max_work:
                rec["skipped"].append({"dims": spec.dims, "dtype": dtype_name})
                continue
            t0 = time.perf_counter()
            outs = []
            entry = {"dims": spec.dims, "hidden": spec.hidden, "dtype": dtype_name, "ok": True, "notes": []}
            for seed in seeds:
                a = mod.make_inputs(spec.dims, dtype, device, seed)
                b = mod.make_inputs(spec.dims, dtype, device, seed)
                if not all(bitwise_equal(x, y) for x, y in zip(a, b)):
                    entry["ok"] = False
                    entry["notes"].append(f"seed {seed}: make_inputs not deterministic")
                shape, odtype, hi = reference_outputs(mod.reference, a)
                native = mod.reference(*a)
                if not bool(torch.isfinite(hi).all()):
                    entry["ok"] = False
                    entry["notes"].append(f"seed {seed}: high-precision reference has non-finite values")
                c = compare(native, hi, shape, odtype, native.device, **tol)
                if not c["passed"]:
                    entry["ok"] = False
                    entry["notes"].append(f"seed {seed}: native reference fails own tolerance: {c['error']}")
                entry[f"native_max_abs_err_seed{seed}"] = c["max_abs_err"]
                entry[f"native_max_rel_err_seed{seed}"] = c["max_rel_err"]
                outs.append(native)
            if len(outs) == 2 and bitwise_equal(outs[0], outs[1]):
                # Degenerate shapes (e.g. N=1 normalizations) are allowed for hidden shapes only.
                msg = "outputs identical across seeds (anti-caching check is vacuous for this shape)"
                entry["notes"].append(msg)
                if not spec.hidden:
                    entry["ok"] = False
            entry["seconds"] = round(time.perf_counter() - t0, 3)
            rec["shapes"].append(entry)
    rec["ok"] = not rec["problems"] and all(s["ok"] for s in rec["shapes"])
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-work", type=float, default=None,
                    help="skip shapes whose product of dims exceeds this (for CPU smoke runs)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="JSON report path")
    args = ap.parse_args(argv)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tasks = [load_task(t) for t in args.tasks] if args.tasks else all_tasks()
    results = []
    for t in tasks:
        r = validate_task(t, args.device, args.max_work)
        n_ok = sum(s["ok"] for s in r["shapes"])
        status = "OK  " if r["ok"] else "FAIL"
        print(f"{status} {t.id:28s} shapes ok {n_ok}/{len(r['shapes'])} skipped {len(r['skipped'])}")
        for p in r["problems"]:
            print(f"       problem: {p}")
        for s in r["shapes"]:
            for n in s["notes"]:
                print(f"       {s['dims']} {s['dtype']}: {n}")
        results.append(r)
    all_ok = all(r["ok"] for r in results)
    out = args.out or f"docs/results/task_validation_{args.device}.json"
    atomic_write_json(out, stamp({"device": args.device, "max_work": args.max_work, "all_ok": all_ok,
                                  "tasks": results}))
    print(f"\n{'ALL TASKS VALID' if all_ok else 'VALIDATION FAILED'} -> {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
