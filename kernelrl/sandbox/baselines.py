"""Clean-process reference timings (eager + torch.compile), cached on disk.

Cache key covers everything that changes the numbers: task reference source, shape, dtype,
benchmark methodology/config, GPU model + driver, and torch/triton versions.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from kernelrl.common.env_fingerprint import REPO_ROOT, fingerprint
from kernelrl.common.io import atomic_write_json, read_json
from kernelrl.sandbox.isolation import child_env, run_child
from kernelrl.tasks.registry import Task


def cache_dir() -> Path:
    return Path(os.environ.get("KERNELRL_CACHE_DIR", REPO_ROOT / ".kernelrl_cache")) / "baselines"


def _gpu_key(gpu_index: int = 0) -> dict:
    fp = fingerprint()
    gpus = fp["gpus"]
    g = next((x for x in gpus if x["index"] == gpu_index), gpus[0] if gpus else {})
    # uuid: speedups must compare timings from the same physical card, not just the same model
    # (parallel cloud runs land on different H100s that share one cache volume).
    return {"gpu": g.get("name"), "gpu_uuid": g.get("uuid"), "driver": g.get("driver"),
            "torch": fp["libs"]["torch"], "triton": fp["libs"]["triton"]}


def baseline_key(task: Task, dims: dict, dtype: str, cfg: dict, gpu_index: int = 0) -> str:
    payload = {
        "task": task.id, "ref_sha": hashlib.sha256(task.reference_source().encode()).hexdigest(),
        "dims": dims, "dtype": dtype, "methodology": cfg["methodology_version"], "bench": cfg["benchmark"],
        **_gpu_key(gpu_index),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def get_baselines(task: Task, specs: list[dict], cfg: dict, *, gpu: str | None = None,
                  gpu_index: int = 0) -> dict[str, dict]:
    """Return {shape_key: {"eager": stats, "compiled": stats|None, ...}} for bench specs,
    computing missing ones in a clean child process."""
    dtype = task.dtypes[0]
    bench = [s for s in specs if s.get("bench")]
    out, missing = {}, []
    for s in bench:
        p = cache_dir() / task.id / f"{baseline_key(task, s['dims'], dtype, cfg, gpu_index)}.json"
        if p.exists():
            out[_k(s["dims"])] = read_json(p)
        else:
            missing.append((s, p))
    if not missing:
        return out
    with tempfile.TemporaryDirectory(prefix="kernelrl_baseline_") as d:
        jd = Path(d)
        job = {"task_id": task.id, "config": cfg, "shapes": [s for s, _ in missing], "dtypes": [dtype],
               "tolerance": task.tolerance}
        (jd / "job.json").write_text(json.dumps(job))
        extra = {"TRITON_CACHE_DIR": str(jd / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(cache_dir() / "inductor"),
                 "TMPDIR": str(jd)}
        if gpu is not None:
            extra["CUDA_VISIBLE_DEVICES"] = str(gpu)
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            extra["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        oc = run_child([sys.executable, "-m", "kernelrl.sandbox.runner", str(jd), "baseline"], cwd=jd,
                       env=child_env(cfg["child_env_passthrough"], extra), timeout_s=cfg["timeouts_s"]["baseline"],
                       rss_limit_bytes=int(cfg["limits"]["host_rss_gb"] * 2**30), name="baseline")
        rp = jd / "baseline.result.json"
        if not rp.exists():
            raise RuntimeError(f"baseline stage failed for {task.id}: rc={oc.returncode} "
                               f"timeout={oc.timed_out}\n{oc.tail()}")
        res = read_json(rp)
        if not res.get("ok"):
            raise RuntimeError(f"baseline stage failed for {task.id}: {res.get('message')}")
        by_key = {_k(b["shape"]): b for b in res["baselines"]}
        for s, p in missing:
            rec = by_key[_k(s["dims"])]
            atomic_write_json(p, {**rec, "env": _gpu_key(gpu_index)})
            out[_k(s["dims"])] = rec
    return out


def _k(dims: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in dims.items())
