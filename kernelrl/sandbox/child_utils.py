"""Helpers used only inside sandbox child processes (imports torch)."""
from __future__ import annotations

import errno
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

from kernelrl.sandbox.compare import clone_inputs, upcast


class StageFailure(Exception):
    """A candidate-attributable failure that ends the stage."""

    def __init__(self, status: str, reason: str, message: str, spec: dict | None = None):
        super().__init__(message)
        self.status, self.reason, self.message, self.spec = status, reason, message, spec

    def to_result(self) -> dict:
        return {
            "ok": False, "status": self.status, "reason": self.reason, "message": self.message,
            "shape": self.spec["dims"] if self.spec else None,
            "hidden": bool(self.spec and self.spec.get("hidden")),
        }


def classify_exception(e: BaseException) -> tuple[str, str]:
    name, mod, msg = type(e).__name__, type(e).__module__ or "", str(e)
    if isinstance(e, torch.OutOfMemoryError):
        return "crash", "cuda_oom"
    if isinstance(e, OSError) and e.errno == errno.EFBIG:  # RLIMIT_FSIZE (CPython ignores SIGXFSZ)
        return "crash", "file_size_limit"
    if mod.startswith("triton"):
        if name == "OutOfResources":
            return "compile_error", "triton_out_of_resources"
        return "compile_error", "triton_compilation_error"
    if "CUDA error" in msg or "cudaError" in msg or "illegal memory access" in msg or "CUBLAS_STATUS" in msg:
        return "crash", "cuda_error"
    return "crash", f"runtime_exception:{name}"


def format_exception(e: BaseException, candidate_path: str | None, limit_frames: int = 6) -> str:
    """Exception text plus only the stack frames from the candidate's own file."""
    frames = traceback.extract_tb(e.__traceback__)
    if candidate_path:
        frames = [f for f in frames if f.filename == candidate_path]
    lines = []
    if frames:
        lines.append("Traceback (candidate frames only):")
        for f in frames[-limit_frames:]:
            lines.append(f'  line {f.lineno}, in {f.name}\n    {(f.line or "").strip()}')
    lines.append("".join(traceback.format_exception_only(type(e), e)).strip())
    return "\n".join(lines)


class Progress:
    """Append-only JSONL progress log, so a hard crash still tells the parent where it happened."""

    def __init__(self, path: Path):
        self.f = open(path, "a", buffering=1)

    def log(self, **kw) -> None:
        kw["t"] = round(time.time(), 3)
        self.f.write(json.dumps(kw, default=str) + "\n")
        self.f.flush()


def poison_allocator(nbytes: int, device: str = "cuda") -> None:
    """Fill a block with 0xFF bytes (NaN for every float dtype) and release it to the
    caching allocator, so an output that is allocated but never written reads as NaN
    instead of stale data that might happen to be correct."""
    if nbytes <= 0:
        return
    t = torch.empty(nbytes, dtype=torch.uint8, device=device)
    t.fill_(255)
    del t


def expected_output_meta(ref_fn, inputs) -> tuple[tuple, torch.dtype]:
    """Output shape/dtype of the reference without materializing a native-dtype output on
    the GPU (a stale native output in freed memory could be read back by a no-op kernel)."""
    try:
        meta = tuple(torch.empty_like(x, device="meta") if isinstance(x, torch.Tensor) else x for x in inputs)
        with torch.no_grad():
            out = ref_fn(*meta)
        return tuple(out.shape), out.dtype
    except Exception:
        with torch.no_grad():
            out = ref_fn(*clone_inputs(inputs))
        shape, dtype = tuple(out.shape), out.dtype
        if out.is_floating_point():  # scrub before freeing
            out.fill_(float("nan"))
        else:
            out.fill_(-1)
        del out
        return shape, dtype


@torch.no_grad()
def reference_hi(ref_fn, inputs) -> torch.Tensor:
    """High-precision reference values (always a different dtype than the expected output)."""
    return ref_fn(*upcast(clone_inputs(inputs))).detach()


def nbytes_of(shape: tuple, dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= s
    return n * torch.empty((), dtype=dtype).element_size()


class ApiSentinel:
    """Snapshot identities of callables a candidate could monkeypatch to fake results or
    timings (torch ops, CUDA events, timers, the reference module). Any change => tamper."""

    def __init__(self, extra_modules=()):
        import time as _time

        import torch.nn.functional as F

        self.targets = {
            "torch": torch, "torch.nn.functional": F, "torch.cuda": torch.cuda, "torch.Tensor": torch.Tensor,
            "torch.cuda.Event": torch.cuda.Event, "time": _time,
        }
        for i, m in enumerate(extra_modules):
            self.targets[f"extra{i}:{getattr(m, '__name__', '?')}"] = m
        self.snap = {}
        for label, obj in self.targets.items():
            for k, v in list(vars(obj).items()):
                if not k.startswith("_") and callable(v):
                    self.snap[(label, k)] = v

    def changed(self) -> list[str]:
        out = []
        for (label, k), v in self.snap.items():
            if vars(self.targets[label]).get(k) is not v:
                out.append(f"{label}.{k}")
        return out


@contextmanager
def record_triton_launches():
    """Collect (kernel_name, stream) for every Triton launch (triton>=3.4 knobs API)."""
    import triton

    launches: list[tuple[str | None, int | None]] = []

    def hook(md):
        try:
            d = md.get() if md is not None else {}
        except Exception:
            d = {}
        launches.append((d.get("name"), d.get("stream")))

    chain = triton.knobs.runtime.launch_enter_hook
    chain.add(hook)
    try:
        yield launches
    finally:
        chain.remove(hook)
