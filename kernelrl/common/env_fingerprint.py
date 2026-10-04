"""Environment fingerprint embedded in every result JSON (reproducibility rule).

Does not import torch/triton in the calling process: library versions come
from package metadata, GPU info from nvidia-smi. Call `runtime_fingerprint()`
inside a process that already imported torch for the CUDA runtime version.
"""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS = ("torch", "triton", "sglang", "openai", "numpy", "ray", "transformers")


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> dict:
    commit = _run(["git", "rev-parse", "HEAD"], REPO_ROOT)
    dirty = _run(["git", "status", "--porcelain"], REPO_ROOT)
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def gpu_info() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,clocks.max.sm,uuid",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            gpus.append({
                "index": int(parts[0]), "name": parts[1], "driver": parts[2],
                "memory_mib": int(parts[3]), "max_sm_clock_mhz": parts[4], "uuid": parts[5],
            })
    cuda = _run(["nvidia-smi"])
    if cuda and gpus:
        for tok in cuda.split("CUDA Version:")[1:2]:
            gpus[0]["driver_cuda_version"] = tok.split()[0]
    return gpus


def lib_versions() -> dict:
    out = {}
    for lib in LIBS:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            out[lib] = None
    return out


@lru_cache(maxsize=1)
def fingerprint() -> dict:
    return {
        "git": git_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "gpus": gpu_info(),
        "libs": lib_versions(),
    }


def runtime_fingerprint() -> dict:
    """Extra info only available inside a process that has torch imported."""
    import torch  # noqa: PLC0415

    info = {"torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        info.update({"device_name": p.name, "capability": f"{p.major}.{p.minor}",
                     "sm_count": p.multi_processor_count})
    return info


def stamp(payload: dict) -> dict:
    """Return payload wrapped with environment + timestamp metadata."""
    return {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "env": fingerprint(),
        **payload,
    }
