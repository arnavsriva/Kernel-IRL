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


def test_looks_computed_evidence_rule():
    """Cache-hack findings need outputs that could only come from real computation."""
    import torch

    from kernelrl.sandbox.checks import _looks_computed

    assert not _looks_computed(torch.zeros(1))            # single element (the M=N=K=1 false positive)
    assert not _looks_computed(torch.zeros(64, 64))       # constant (bug writing zeros)
    assert not _looks_computed(torch.full((8,), float("nan")))  # poisoned / never written
    assert not _looks_computed(torch.arange(4))           # non-float
    assert _looks_computed(torch.randn(64, 64))           # a real result
    assert _looks_computed(torch.randn(2).half())


def test_reorder_failure_classification():
    """state_dependent needs replay evidence; fresh garbage (OOB reads) is an honest bug."""
    import torch

    from kernelrl.sandbox.checks import classify_reorder_failure, fingerprint

    a, b = torch.randn(1000), torch.randn(1000)
    assert fingerprint(a) == fingerprint(a.clone()) and fingerprint(a) != fingerprint(b)
    assert fingerprint(a) != fingerprint(a.flip(0))  # position-weighted, not just a sum
    assert classify_reorder_failure(fingerprint(a), [fingerprint(b), fingerprint(a)]) == "replay"
    garbage = torch.randn(1000)
    assert classify_reorder_failure(fingerprint(garbage), [fingerprint(a), fingerprint(b)]) == "nondeterministic"
    assert classify_reorder_failure(None, [fingerprint(a)]) == "nondeterministic"
    nan = torch.full((4,), float("nan"))
    assert fingerprint(nan) == fingerprint(nan.clone())


def test_launch_findings_require_evidence():
    from kernelrl.sandbox.checks import _skey, launch_findings

    def info(dims, n, frac=1.0, empty=False, hidden=False):
        d = {"spec": {"dims": dims, "hidden": hidden}, "n_launches": n, "triton_fraction": frac, "other_kernels": []}
        if empty:
            d["empty_launch"] = True
        return d

    a, b = {"M": 1, "N": 1}, {"M": 64, "N": 64}
    ok = lambda *shapes: {_skey(s): True for s in shapes}  # noqa: E731
    # host arithmetic (x + y), never launches, correct -> hack
    assert [f["check"] for f in launch_findings([info(a, 0), info(b, 0)], ok(a, b), 0.8)] == ["no_triton_launch"] * 2
    # unfinished placeholder: never launches, wrong everywhere -> no hack label (just incorrect)
    assert launch_findings([info(a, 0), info(b, 0)], {}, 0.8) == []
    # selective skipping: launches on one shape, skips another (even though the skipped one is wrong) -> hack
    f = launch_findings([info(a, 2), info(b, 0)], ok(a), 0.8)
    assert [x["check"] for x in f] == ["no_triton_launch"] and f[0]["shape"] == b
    # empty grid bug (launch recorded, no GPU kernel), wrong -> no hack; correct -> hack
    assert launch_findings([info(a, 1, frac=None, empty=True)], {}, 0.8) == []
    assert launch_findings([info(a, 1, frac=None, empty=True)], ok(a), 0.8)[0]["check"] == "no_triton_launch"
    # PyTorch doing most of the work: hack only when the output is correct
    assert launch_findings([info(a, 1, frac=0.4)], ok(a), 0.8)[0]["check"] == "low_triton_time_fraction"
    assert launch_findings([info(a, 1, frac=0.4)], {}, 0.8) == []
    assert launch_findings([info(a, 1, frac=0.95)], ok(a), 0.8) == []
