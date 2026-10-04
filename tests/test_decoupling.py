"""CLAUDE.md rule: the sandbox must not import serving/training/eval code, so it can become a
Ray actor in Phase 1 unchanged. The parent-side modules must not import torch."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "kernelrl"
FORBIDDEN_FOR_SANDBOX = ("kernelrl.serving", "kernelrl.eval", "kernelrl.bench", "kernelrl.train", "sglang",
                         "openai", "ray", "megatron", "slime")
PARENT_SIDE = ["sandbox/static_check.py", "sandbox/isolation.py", "sandbox/evaluate.py", "sandbox/baselines.py",
               "sandbox/feedback.py", "sandbox/config.py", "reward/reward.py", "tasks/registry.py",
               "tasks/schema.py", "common/schemas.py", "common/io.py"]


def _imports(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_sandbox_does_not_import_serving_or_training():
    for p in (PKG / "sandbox").glob("*.py"):
        bad = [m for m in _imports(p) if m.startswith(FORBIDDEN_FOR_SANDBOX)]
        assert not bad, f"{p.name} imports {bad}"


def test_parent_side_is_torch_free():
    for rel in PARENT_SIDE:
        mods = _imports(PKG / rel)
        assert not any(m == "torch" or m.startswith(("torch.", "triton")) for m in mods), rel
        assert "kernelrl.sandbox.compare" not in mods and "kernelrl.sandbox.child_utils" not in mods, rel
