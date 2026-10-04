"""Correctness-gated, clipped log-speedup reward."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from kernelrl.common.schemas import EvalResult, Status

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=None)
def load_reward_config(path: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


@dataclass
class RewardBreakdown:
    reward: float | None
    status: str
    speedup: float | None = None
    normalized_speedup: float | None = None
    components: dict = field(default_factory=dict)


def compute_reward(result: EvalResult, cfg: dict | None = None) -> RewardBreakdown:
    cfg = cfg or load_reward_config()
    if result.meta.get("infra_error"):
        return RewardBreakdown(cfg.get("infra_error_reward"), result.status.value, components={"infra_error": True})
    if result.status is not Status.CORRECT:
        r = float(cfg["status_rewards"][result.status.value])
        return RewardBreakdown(r, result.status.value, components={"status": r})
    c = cfg["correct"]
    speedup = result.speedup_vs_eager if c["speedup_baseline"] == "eager" else result.speedup_vs_compile
    if speedup is None or not math.isfinite(speedup) or speedup <= 0:
        r = float(c["base"]) if c["no_timing_reward"] == "base" else 0.0
        return RewardBreakdown(r, "correct", components={"base": r, "timing": None})
    lo, hi = float(c["log_clip_min"]), float(c["log_clip_max"])
    lg = math.log(speedup) / math.log(float(c["log_base"]))
    norm = (min(max(lg, lo), hi) - lo) / (hi - lo)
    r = float(c["base"]) + float(c["speedup_weight"]) * norm
    return RewardBreakdown(r, "correct", speedup, norm, {"base": float(c["base"]),
                                                        "speedup_term": float(c["speedup_weight"]) * norm})
