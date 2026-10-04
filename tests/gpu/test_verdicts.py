"""End-to-end verdicts for every adversarial fixture (GPU box only).

If any hack fixture gets "correct", the verifier has a hole: stop and fix it before Phase 1.
"""
import pytest

from kernelrl.common.schemas import Status
from kernelrl.sandbox.config import load_config
from kernelrl.sandbox.evaluate import evaluate
from tests.fixture_meta import all_fixtures

pytestmark = pytest.mark.gpu

# Faster than production settings, same checks.
FAST = {
    "benchmark": {"warmup": 5, "iters": 20, "wall_iters": 20},
    "timeouts_s": {"compile": 60, "check": 120, "bench": 180},
    "correctness": {"fixed_seeds": [11, 12]},
}
HOSTILE = {"timeouts_s": {"compile": 20, "check": 20}, "limits": {"host_rss_gb": 4, "max_file_bytes": 16 << 20}}

RUNTIME = [f for f in all_fixtures() if f.runtime and (f.expect != "static_reject" or f.skip_static)]


@pytest.mark.parametrize("fx", RUNTIME, ids=[f.id for f in RUNTIME])
def test_fixture_verdict(fx):
    overrides = {**FAST, **(HOSTILE if fx.category == "hostile" else {})}
    cfg = load_config(overrides=overrides)
    run_bench = fx.expect == "correct" or fx.needs_bench
    r = evaluate(fx.task, fx.source, config=cfg, run_bench=run_bench, skip_static=fx.skip_static)
    assert r.status.value == fx.expect, f"{fx.id}: got {r.status.value} ({r.reason}): {r.message}"
    if fx.reasons:
        assert r.reason in fx.reasons, f"{fx.id}: reason {r.reason!r} not in {fx.reasons}: {r.message}"
    assert not r.meta.get("infra_error"), r.message
    if fx.category == "hostile" and r.status in (Status.TIMEOUT, Status.CRASH):
        assert r.meta.get("gpu_healthy_after", True), "GPU unhealthy after hostile fixture"
    if fx.expect == "correct":
        assert r.speedup_vs_eager is not None and 0.2 < r.speedup_vs_eager < 20, r.speedup_vs_eager
        for t in r.timings:
            assert t.candidate.median_ms > 0 and t.candidate.iqr_ms < t.candidate.median_ms * 2


def test_no_hack_scores_correct():
    """Summary guard: run every hack fixture; none may be judged correct."""
    cfg = load_config(overrides=FAST)
    leaked = []
    for fx in [f for f in all_fixtures() if f.category == "hacks" and f.runtime]:
        r = evaluate(fx.task, fx.source, config=cfg, run_bench=True)
        if r.status is Status.CORRECT:
            leaked.append(fx.id)
    assert not leaked, f"HACKS PASSED THE VERIFIER: {leaked}"


def test_parent_survives_back_to_back_hostile_then_good():
    """After a hostile candidate, the same GPU still evaluates a good kernel correctly."""
    cfg = load_config(overrides={**FAST, **HOSTILE})
    by_id = {f.id: f for f in all_fixtures()}
    for hid in ("hostile/illegal_memory_access", "hostile/infinite_loop_kernel"):
        evaluate(by_id[hid].task, by_id[hid].source, config=cfg, run_bench=False)
    good = by_id["good/add"]
    r = evaluate(good.task, good.source, config=load_config(overrides=FAST), run_bench=False)
    assert r.status is Status.CORRECT, r.message
