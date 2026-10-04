"""Result schemas shared by sandbox, reward, and eval.

Everything here is plain dataclasses + JSON so it can cross process boundaries
(sandbox subprocesses today, Ray actors in Phase 1) without importing torch.
"""
from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Any


class Status(str, enum.Enum):
    """Final verdict for one candidate kernel. Order of precedence when several
    apply: static_reject > compile_error > timeout > crash > hack_detected >
    incorrect > correct. (hack_detected outranks incorrect so that hacks are
    counted in the taxonomy even when they also produce wrong numbers.)"""

    STATIC_REJECT = "static_reject"
    COMPILE_ERROR = "compile_error"
    TIMEOUT = "timeout"
    CRASH = "crash"
    INCORRECT = "incorrect"
    HACK_DETECTED = "hack_detected"
    CORRECT = "correct"

    @property
    def compiled(self) -> bool:
        return self in (Status.INCORRECT, Status.HACK_DETECTED, Status.CORRECT, Status.CRASH)


FAILURE_STATUSES = frozenset(
    {Status.STATIC_REJECT, Status.COMPILE_ERROR, Status.TIMEOUT, Status.CRASH, Status.HACK_DETECTED}
)


@dataclass
class Finding:
    """One anti-hacking or static-check finding. `check` is a stable taxonomy key."""

    check: str
    message: str
    shape: dict | None = None
    hidden: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ShapeResult:
    shape: dict
    dtype: str
    seed: int
    hidden: bool
    passed: bool
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    mismatch_frac: float | None = None
    error: str | None = None  # e.g. "nan_in_output", "shape_mismatch: got (3,) expected (4,)"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TimingStats:
    median_ms: float
    iqr_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    n: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ShapeTiming:
    shape: dict
    dtype: str
    hidden: bool
    candidate: TimingStats
    eager_inproc: TimingStats | None = None  # paired ABAB in candidate process (sentinel)
    eager: TimingStats | None = None  # clean process baseline
    compiled: TimingStats | None = None  # torch.compile, clean process baseline
    speedup_vs_eager: float | None = None
    speedup_vs_compile: float | None = None
    wall_over_event: float | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class EvalResult:
    task_id: str
    status: Status
    reason: str = ""  # short machine-readable reason, e.g. "forbidden_torch_api"
    message: str = ""  # human/model readable, truncated
    stage: str = ""  # static | compile | check | bench
    findings: list[Finding] = field(default_factory=list)
    shape_results: list[ShapeResult] = field(default_factory=list)
    timings: list[ShapeTiming] = field(default_factory=list)
    speedup_vs_eager: float | None = None  # geomean over bench shapes
    speedup_vs_compile: float | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "EvalResult":
        d = dict(d)
        d["status"] = Status(d["status"])
        d["findings"] = [Finding(**f) for f in d.get("findings", [])]
        d["shape_results"] = [ShapeResult(**s) for s in d.get("shape_results", [])]
        timings = []
        for t in d.get("timings", []):
            t = dict(t)
            for k in ("candidate", "eager_inproc", "eager", "compiled"):
                if t.get(k) is not None:
                    t[k] = TimingStats(**t[k])
            timings.append(ShapeTiming(**t))
        d["timings"] = timings
        return EvalResult(**d)
