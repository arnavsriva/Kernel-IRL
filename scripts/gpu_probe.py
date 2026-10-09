"""Fast pre-flight on a new GPU host (any provider, including gVisor-sandboxed ones like Modal).

Checks every platform assumption the sandbox makes, then runs a few fixtures end to end:
  python scripts/gpu_probe.py --out docs/results/gpu_probe.json
Exit code 1 if any check fails. Costs a few GPU-minutes.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelrl.common.env_fingerprint import fingerprint  # noqa: E402

FIXTURES = [  # (fixture id, run_bench)
    ("good/add", True), ("good/softmax", False), ("buggy/add_offbyone", False),
    ("hacks/no_kernel_launch", False), ("hacks/trivial_triton_wrapper", False), ("hacks/cache_first_result", False),
    ("hacks/lazy_after_call_counter", True),
    ("hostile/illegal_memory_access", False), ("hostile/infinite_loop_kernel", False), ("hostile/segfault", False),
]


def check_profiler_and_hook() -> dict:
    """CUPTI kernel tracing via torch.profiler + Triton launch hook, and that their names agree."""
    import torch
    import triton
    import triton.language as tl
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    from kernelrl.sandbox.child_utils import record_triton_launches

    @triton.jit
    def probe_add_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1.0, mask=m)

    x = torch.randn(1 << 20, device="cuda")
    out = torch.empty_like(x)
    probe_add_kernel[(1024,)](x, out, x.numel(), BLOCK=1024)  # compile outside profiling
    torch.cuda.synchronize()
    with record_triton_launches() as launches:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            probe_add_kernel[(1024,)](x, out, x.numel(), BLOCK=1024)
            y = x * 2.0  # an aten kernel
            torch.cuda.synchronize()
    kernels = [(e.name, e.time_range.elapsed_us()) for e in prof.events() if e.device_type == DeviceType.CUDA]
    hook_names = sorted({n for n, _ in launches if n})
    names = [n for n, _ in kernels]
    cur = torch.cuda.current_stream().cuda_stream
    return {
        "cupti_kernel_events": len(kernels), "kernel_names": names[:10], "hook_names": hook_names,
        "hook_stream_is_current": all(s == cur for _, s in launches),
        "ok": bool(kernels) and "probe_add_kernel" in hook_names and "probe_add_kernel" in names
              and any(n != "probe_add_kernel" for n in names) and bool(y.numel()),
    }


def check_isolation(tmp: Path) -> dict:
    from kernelrl.sandbox.isolation import _rss_bytes, child_env, run_child

    env = child_env(["PATH", "HOME", "LD_LIBRARY_PATH"], {})
    t = run_child([sys.executable, "-c", "import subprocess,sys,time\n"
                   "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\ntime.sleep(60)"],
                  cwd=tmp, env=env, timeout_s=2, rss_limit_bytes=1 << 34, name="timeout")
    seg = run_child([sys.executable, "-c", "import ctypes; ctypes.string_at(0)"], cwd=tmp, env=env, timeout_s=10,
                    rss_limit_bytes=1 << 34, name="segv")
    fs = run_child([sys.executable, "-c", "import resource\nresource.setrlimit(resource.RLIMIT_FSIZE,(1<<20,1<<20))\n"
                    "while True: print('x'*100000, flush=True)"], cwd=tmp, env=env, timeout_s=10,
                   rss_limit_bytes=1 << 34, name="fsize")
    mem = run_child([sys.executable, "-c", "import time\nx=b'\\x01'*(3*1024**3)\ntime.sleep(30)"], cwd=tmp, env=env,
                    timeout_s=20, rss_limit_bytes=1 << 30, name="mem")
    res = {
        "timeout_kill": t.timed_out and t.wall_s < 6,
        "segfault_signal": seg.signal_name,
        "file_size_limit": fs.signal_name == "SIGXFSZ" or "File too large" in fs.tail(),
        "rss_watchdog": mem.rss_exceeded,
        "proc_rss_readable": _rss_bytes(1) >= 0,
        "rlimit_fsize_settable": True,
    }
    try:
        resource.getrlimit(resource.RLIMIT_FSIZE)
    except (ValueError, OSError):
        res["rlimit_fsize_settable"] = False
    res["ok"] = (res["timeout_kill"] and res["segfault_signal"] in ("SIGSEGV", "SIGBUS") and res["file_size_limit"]
                 and res["rss_watchdog"])
    return res


def check_fixtures() -> list[dict]:
    from kernelrl.sandbox.config import load_config
    from kernelrl.sandbox.evaluate import evaluate
    from tests.fixture_meta import all_fixtures

    by_id = {f.id: f for f in all_fixtures()}
    cfg = load_config(overrides={"benchmark": {"warmup": 5, "iters": 20, "wall_iters": 20},
                                 "timeouts_s": {"compile": 60, "check": 120, "bench": 180},
                                 "correctness": {"fixed_seeds": [11, 12]}})
    hostile = load_config(overrides={"timeouts_s": {"compile": 20, "check": 20}, "correctness": {"fixed_seeds": [11, 12]}})
    rows = []
    for fid, bench in FIXTURES:
        fx = by_id[fid]
        t0 = time.perf_counter()
        r = evaluate(fx.task, fx.source, config=hostile if fx.category == "hostile" else cfg,
                     run_bench=bench or fx.needs_bench, skip_static=fx.skip_static)
        ok = r.status.value == fx.expect and (not fx.reasons or r.reason in fx.reasons) and not r.meta.get("infra_error")
        if fx.category == "hostile":
            ok = ok and r.meta.get("gpu_healthy_after", True)
        rows.append({"fixture": fid, "expect": fx.expect, "got": r.status.value, "reason": r.reason, "ok": ok,
                     "speedup_vs_eager": r.speedup_vs_eager, "seconds": round(time.perf_counter() - t0, 1),
                     "gpu_healthy_after": r.meta.get("gpu_healthy_after"),
                     "message": None if ok else r.message[:800]})
        print(f"  {'OK ' if ok else 'BAD'} {fid:36s} expect={fx.expect:14s} got={r.status.value}/{r.reason}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/gpu_probe.json")
    ap.add_argument("--skip-fixtures", action="store_true")
    args = ap.parse_args()
    report = {"env": fingerprint(), "kernel_release": platform.release()}
    try:
        report["proc_version"] = Path("/proc/version").read_text().strip()
    except OSError:
        report["proc_version"] = None
    report["gvisor"] = "gvisor" in (report["proc_version"] or "").lower() or platform.release().startswith("4.4.0")
    tmp = Path("/tmp/kernelrl_probe")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, fn in (("profiler_and_launch_hook", check_profiler_and_hook), ("isolation", lambda: check_isolation(tmp))):
        try:
            report[name] = fn()
        except Exception:  # noqa: BLE001
            report[name] = {"ok": False, "error": traceback.format_exc()[-2000:]}
        print(f"{name}: {'OK' if report[name]['ok'] else 'FAIL'} {json.dumps(report[name])[:400]}", flush=True)
    if not args.skip_fixtures:
        report["fixtures"] = check_fixtures()
    report["ok"] = (report["profiler_and_launch_hook"]["ok"] and report["isolation"]["ok"]
                    and all(r["ok"] for r in report.get("fixtures", [])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nPROBE {'PASSED' if report['ok'] else 'FAILED'} -> {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
