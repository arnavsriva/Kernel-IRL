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
