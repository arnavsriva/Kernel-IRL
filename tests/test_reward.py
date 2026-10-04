import math

import pytest

from kernelrl.common.schemas import EvalResult, Status
from kernelrl.reward.reward import compute_reward, load_reward_config


def _r(status, **kw):
    return compute_reward(EvalResult("l1_vector_add", status, **kw))


@pytest.mark.parametrize("status", [s for s in Status if s is not Status.CORRECT])
def test_failures_zero_by_default(status):
    assert _r(status).reward == 0.0


def test_correct_curve():
    assert _r(Status.CORRECT, speedup_vs_eager=0.5).reward == pytest.approx(0.3)
    assert _r(Status.CORRECT, speedup_vs_eager=1.0).reward == pytest.approx(0.3)
    assert _r(Status.CORRECT, speedup_vs_eager=2.0).reward == pytest.approx(0.65)
    assert _r(Status.CORRECT, speedup_vs_eager=4.0).reward == pytest.approx(1.0)
    assert _r(Status.CORRECT, speedup_vs_eager=100.0).reward == pytest.approx(1.0)


def test_monotone_and_bounded():
    xs = [0.1 * i for i in range(1, 80)]
    rs = [_r(Status.CORRECT, speedup_vs_eager=x).reward for x in xs]
    assert all(b >= a for a, b in zip(rs, rs[1:]))
    assert min(rs) >= 0.3 and max(rs) <= 1.0


def test_config_controls_incorrect_and_baseline():
    cfg = dict(load_reward_config())
    cfg = {**cfg, "status_rewards": {**cfg["status_rewards"], "incorrect": 0.05},
           "correct": {**cfg["correct"], "speedup_baseline": "compile"}}
    assert compute_reward(EvalResult("t", Status.INCORRECT), cfg).reward == 0.05
    r = compute_reward(EvalResult("t", Status.CORRECT, speedup_vs_eager=4.0, speedup_vs_compile=1.0), cfg)
    assert r.reward == pytest.approx(0.3)


def test_missing_timing_and_infra():
    assert _r(Status.CORRECT).reward == pytest.approx(0.3)
    assert _r(Status.CORRECT, speedup_vs_eager=math.nan).reward == pytest.approx(0.3)
    r = compute_reward(EvalResult("t", Status.CRASH, meta={"infra_error": True}))
    assert r.reward is None
