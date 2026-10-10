"""End-to-end eval pipeline with a fake model server and a fake sandbox (CPU only)."""
import json

from kernelrl.common.schemas import EvalResult, Status
from kernelrl.eval import run_eval
from kernelrl.serving.client import Completion

CODE = ("```python\nimport torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef k(x):\n"
        "    pass\n\ndef run(x):\n    return x\n```")


class FakeGen:
    def __init__(self, *a, **k):
        self.calls = []

    async def complete(self, messages, seed):
        self.calls.append((len(messages), seed))
        text = CODE if seed % 2 == 0 or len(messages) > 2 else "no code, sorry"
        return Completion(text, None, "stop", 100, 50, 80, 0.01)


def fake_evaluate(task_id, source, **kw):
    return EvalResult(task_id, Status.CORRECT, speedup_vs_eager=1.5, speedup_vs_compile=0.8, stage="bench")


def test_two_turn_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "Generator", FakeGen)
    monkeypatch.setattr(run_eval, "evaluate", fake_evaluate)
    monkeypatch.setattr(run_eval, "server_info", lambda url: {"version": "fake"})
    args = ["--run-id", "t", "--tasks", "l1_relu", "l2_softmax", "--k", "2", "--turns", "2",
            "--out-dir", str(tmp_path / "runs"), "--results-dir", str(tmp_path / "results")]
    assert run_eval.main(args) == 0
    summary = json.loads((tmp_path / "results" / "t.json").read_text())["summary"]
    t1, t2 = summary["turn1"], summary["turn2"]
    assert t1["n_samples"] == 4 and t1["status_counts"]["correct"] == 2 and t1["status_counts"]["static_reject"] == 2
    assert t1["static_reject_reasons"] == {"no_code_found": 2}
    assert t2["status_counts"]["correct"] == 4
    assert summary["two_turn"]["correct_in_either_rate"] == 1.0
    assert (tmp_path / "results" / "t.md").exists()
    # resume: rerunning does not regenerate or re-evaluate anything
    n_lines = len((tmp_path / "runs" / "t" / "turn1_results.jsonl").read_text().splitlines())
    assert run_eval.main(args) == 0
    assert len((tmp_path / "runs" / "t" / "turn1_results.jsonl").read_text().splitlines()) == n_lines


def test_real_generator_survives_two_event_loops_under_contention(tmp_path, monkeypatch):
    """Regression: the first full baseline crashed at turn 2 ('Semaphore is bound to a different
    event loop') because one Generator was shared by two asyncio.run() calls with > concurrency
    requests queued. Uses the real Generator with a stubbed OpenAI call."""
    import asyncio

    from kernelrl.serving import client as client_mod

    async def fake_create(**kw):
        await asyncio.sleep(0.001)

        class U:
            prompt_tokens, completion_tokens, prompt_tokens_details = 10, 5, None

        class M:
            content, reasoning_content = CODE, None

        class C:
            message, finish_reason = M(), "stop"

        class R:
            choices, usage = [C()], U()
        return R()

    real_init = client_mod.Generator.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        monkeypatch.setattr(self.client.chat.completions, "create", fake_create)

    monkeypatch.setattr(client_mod.Generator, "__init__", init)
    monkeypatch.setattr(run_eval, "evaluate", fake_evaluate)
    monkeypatch.setattr(run_eval, "server_info", lambda url: {})
    args = ["--run-id", "c", "--tasks", "l1_relu", "--k", "12", "--turns", "2", "--concurrency", "2",
            "--out-dir", str(tmp_path / "runs"), "--results-dir", str(tmp_path / "results")]
    assert run_eval.main(args) == 0
    lines = (tmp_path / "runs" / "c" / "turn2_samples.jsonl").read_text().splitlines()
    assert len(lines) == 12
