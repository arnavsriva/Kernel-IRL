"""ApiSentinel: no false positives from torch's own lazy patching, still catches candidate patches.
Runs in a subprocess so the snapshot starts from a fresh interpreter (CPU only)."""
import subprocess
import sys
import textwrap

from kernelrl.common.env_fingerprint import REPO_ROOT


def _run(code: str) -> str:
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                       cwd=REPO_ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_lazy_torch_imports_do_not_trip_sentinel():
    out = _run("""
        from kernelrl.sandbox.child_utils import ApiSentinel, preload_trusted_modules
        preload_trusted_modules()
        s = ApiSentinel()
        import torch._dynamo, torch.profiler, torch._inductor  # trusted lazy imports
        print(s.changed())
    """)
    assert out == "[]"


def test_without_preload_torch_dynamo_is_the_known_false_positive():
    out = _run("""
        from kernelrl.sandbox.child_utils import ApiSentinel
        s = ApiSentinel()
        import torch._dynamo
        print(s.changed())
    """)
    assert out == "['torch.manual_seed']"


def test_candidate_patches_are_still_caught():
    out = _run("""
        import time, torch
        from kernelrl.sandbox.child_utils import ApiSentinel, preload_trusted_modules
        preload_trusted_modules()
        s = ApiSentinel()
        torch.softmax = lambda *a, **k: None
        torch.cuda.Event.elapsed_time = lambda self, end: 0.001
        time.perf_counter = lambda: 0.0
        torch.Tensor.sum = lambda self, *a, **k: self
        print(sorted(s.changed()))
    """)
    for name in ("torch.softmax", "torch.cuda.Event.elapsed_time", "time.perf_counter", "torch.Tensor.sum"):
        assert name in out, out
