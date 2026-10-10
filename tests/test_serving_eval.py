"""Prompt, extraction, feedback redaction, pass@k, and report aggregation (CPU only)."""
import re

import pytest

from kernelrl.common.schemas import EvalResult, ShapeResult, ShapeTiming, Status, TimingStats
from kernelrl.eval.passk import pass_at_k
from kernelrl.eval.report import summarize, write_markdown
from kernelrl.sandbox.config import load_config
from kernelrl.sandbox.feedback import hidden_only_values, render_feedback
from kernelrl.serving.extract import extract_code, strip_thinking
from kernelrl.serving.prompt import initial_messages, revision_messages, user_prompt
from kernelrl.tasks import all_tasks, load_task

CODE = ("import torch\nimport triton\nimport triton.language as tl\n\n\n@triton.jit\ndef k(x):\n    pass\n\n\n"
        "def run(x):\n    return x\n")


@pytest.mark.parametrize("task", all_tasks(), ids=lambda t: t.id)
def test_prompt_never_contains_hidden_shapes(task):
    from kernelrl.serving.prompt import TRITON_GUIDE

    # The v2 guide is identical for every task, so numbers in it cannot reveal a task's hidden shapes.
    p = user_prompt(task).replace(TRITON_GUIDE, "")
    tokens = set(re.findall(r"(?<![\w.])\d+(?![\w.])", p))
    leaked = {v for v in hidden_only_values(task) if str(v) in tokens}
    assert not leaked, f"hidden dims {leaked} appear in the prompt"
    for s in task.hidden_shapes():
        assert ", ".join(f"{k}={v}" for k, v in s.dims.items()) not in p


@pytest.mark.parametrize("resp,method", [
    (f"Here you go:\n```python\n{CODE}```\nHope it helps.", "fenced"),
    (f"<think>draft ```python\ndef run(): pass\n```</think>\n```python\n{CODE}```", "fenced"),
    (f"long reasoning without an opening tag</think>\n```py\n{CODE}```", "fenced"),
    (f"```python\n{CODE}```\n<think>unterminated", "fenced"),
    (f"```python\n{CODE}", "fenced"),  # truncated: closing fence missing
    (f"Sure.\n\n{CODE}\nThis should be fast.", "unfenced"),
    ("```python\nimport triton\n@triton.jit\ndef k(x):\n    pass\n```\nThen:\n```python\ndef run(x):\n"
     "    return x\n```", "fenced_concat"),
    (f"```python\nprint('test')\n```\nFinal:\n```python\n{CODE}```", "fenced"),
])
def test_extraction(resp, method):
    e = extract_code(resp)
    assert e.method == method, e
    assert "def run" in e.code and "triton" in e.code


def test_extraction_failures():
    assert extract_code("I can't help with that.").code is None
    assert extract_code(None).method == "none"
    e = extract_code("```python\ndef helper(:\n```")
    assert e.method == "fenced_incomplete"
    assert extract_code("<think>only thinking, truncated").code is None


def test_strip_thinking():
    assert strip_thinking("<think>a</think>b") == ("b", True)
    assert strip_thinking("plain") == ("plain", False)


def test_revision_messages_drop_thinking():
    t = load_task("l1_relu")
    m = revision_messages(t, "<think>secret</think>answer", "## Evaluation result: INCORRECT")
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]
    assert m[2]["content"] == "answer" and "INCORRECT" in m[3]["content"]
    assert initial_messages(t)[1]["content"] == user_prompt(t)


def test_feedback_redacts_hidden_shapes():
    t = load_task("l1_vector_add")
    cfg = load_config()
    r = EvalResult(t.id, Status.CRASH, reason="cuda_error", stage="check",
                   message="RuntimeError: size 8192 x 1003 failed at line 12",
                   meta={"failure_shape": {"M": 1000, "N": 1003}, "failure_hidden": True})
    fb = render_feedback(r, t, cfg)
    assert "1003" not in fb and "8192" not in fb and "1000" not in fb
    assert "hidden test shape" in fb and "line 12" in fb


def test_feedback_incorrect_lists_public_only():
    t = load_task("l1_vector_add")
    r = EvalResult(t.id, Status.INCORRECT, reason="value_mismatch", message="value_mismatch: ...", stage="check",
                   shape_results=[ShapeResult({"M": 1024, "N": 1024}, "float32", 1, False, True),
                                  ShapeResult({"M": 1000, "N": 1003}, "float32", 1, True, False)],
                   meta={"failure_shape": {"M": 1000, "N": 1003}, "failure_hidden": True})
    fb = render_feedback(r, t, load_config())
    assert "M=1024, N=1024: pass" in fb and "1003" not in fb and "Hidden shapes: 0/1" in fb


def test_feedback_correct_shows_public_timings():
    t = load_task("l1_vector_add")
    ts = TimingStats(0.1, 0.01, 0.1, 0.01, 0.09, 100)
    r = EvalResult(t.id, Status.CORRECT, stage="bench", speedup_vs_eager=1.5, speedup_vs_compile=0.9, timings=[
        ShapeTiming({"M": 4096, "N": 4096}, "float32", False, ts, ts, ts, ts, 1.5, 0.9),
        ShapeTiming({"M": 8192, "N": 8192}, "float32", True, ts, ts, ts, ts, 1.5, 0.9)])
    fb = render_feedback(r, t, load_config())
    assert "M=4096, N=4096" in fb and "8192" not in fb and "1.50x" in fb


def test_pass_at_k():
    assert pass_at_k(8, 0, 1) == 0.0 and pass_at_k(8, 8, 8) == 1.0
    assert pass_at_k(8, 2, 1) == pytest.approx(0.25)
    assert pass_at_k(8, 1, 8) == 1.0
    assert pass_at_k(8, 2, 4) == pytest.approx(1 - (6 * 5 * 4 * 3) / (8 * 7 * 6 * 5))


def _rec(task, i, turn, status, speedup=None, reason=""):
    res = EvalResult(task, Status(status), reason=reason, speedup_vs_eager=speedup, speedup_vs_compile=speedup)
    return {"id": f"{task}:{i}:t{turn}", "task": task, "sample": i, "turn": turn, "result": res.to_dict(),
            "reward": 0.5, "extraction": {"method": "fenced"}, "finish_reason": "stop"}


def test_summary_and_markdown(tmp_path):
    tasks = [load_task("l1_vector_add"), load_task("l2_softmax")]
    r1 = {r["id"]: r for r in [_rec("l1_vector_add", 0, 1, "correct", 1.2), _rec("l1_vector_add", 1, 1, "incorrect"),
                               _rec("l2_softmax", 0, 1, "static_reject", reason="forbidden_api"),
                               _rec("l2_softmax", 1, 1, "hack_detected", reason="cached_output")]}
    r2 = {r["id"]: r for r in [_rec("l1_vector_add", 0, 2, "correct", 2.4), _rec("l1_vector_add", 1, 2, "correct", 1.0),
                               _rec("l2_softmax", 0, 2, "compile_error"), _rec("l2_softmax", 1, 2, "incorrect")]}
    s = summarize(r1, r2, 2, tasks)
    assert s["turn1"]["pass@1"] == pytest.approx(0.25)
    assert s["turn1"]["hack_detected"] == 1 and s["turn1"]["static_rule_violations"] == 1
    assert s["two_turn"]["turn2_correct_rate"] == pytest.approx(0.5)
    assert s["two_turn"]["speedup_ratio_t2_over_t1_geomean"] == pytest.approx(2.0)
    md = write_markdown(s, {"run_id": "t", "model": "m", "thinking": False, "turns": 2, "sampling": {}},
                        tmp_path / "r.md").read_text()
    assert "1-turn vs 2-turn" in md and "l2_softmax" in md


@pytest.mark.parametrize("task", all_tasks(), ids=lambda t: t.id)
def test_prompt_v1_never_contains_hidden_shapes(task):
    p = user_prompt(task, "v1")
    tokens = set(re.findall(r"(?<![\w.])\d+(?![\w.])", p))
    assert not {v for v in hidden_only_values(task) if str(v) in tokens}
    assert "Triton quick reference" not in p


def test_prompt_v2_example_is_valid_and_not_a_task():
    from kernelrl.sandbox.static_check import check_source
    from kernelrl.serving.prompt import TRITON_GUIDE

    p = user_prompt(load_task("l2_softmax"), "v2")
    assert "Triton quick reference" in p
    example = extract_code(TRITON_GUIDE).code
    r = check_source(example, expected_arity=2)
    assert r.ok, r  # the worked example must itself follow every rule we enforce
    # the example operation must not be one of the tasks (no free solutions)
    assert "row_scale" in example
    assert all("row_scale" not in t.id and "per-row factor" not in t.description for t in all_tasks())
    with pytest.raises(ValueError):
        user_prompt(load_task("l1_relu"), "v3")
