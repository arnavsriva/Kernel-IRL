"""Static checker unit tests + the static verdict of every adversarial fixture (CPU only)."""
import pytest

from kernelrl.sandbox.static_check import check_source, reference_arity
from kernelrl.tasks import load_task
from tests.fixture_meta import all_fixtures

GOOD = '''
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(configs=[triton.Config({"BLOCK": 1024}, num_warps=4)], key=["n"])
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    s = tl.sum(x, axis=0) + tl.max(x, axis=0)
    tl.store(out_ptr + offs, tl.exp(x) + libdevice.tanh(x) + s * 0, mask=m)


def run(x):
    out = torch.empty_like(x, dtype=torch.float32)
    n = x.numel()
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    k[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, out, n)
    return out.view(x.shape)
'''


def test_good_source_passes():
    r = check_source(GOOD, expected_arity=1)
    assert r.ok, r


@pytest.mark.parametrize("snippet,reason", [
    ("return torch.softmax(x, -1)", "forbidden_api"),
    ("return torch.nn.functional.gelu(x)", "forbidden_api"),
    ("return torch.ops.aten.add(x, x)", "forbidden_api"),
    ("return torch.compile(run)(x)", "forbidden_api"),
    ("return torch.matmul(x, x)", "forbidden_api"),
    ("torch.cuda.synchronize()\n    return out", "forbidden_api"),
    ("s = torch.cuda.Stream()\n    return out", "forbidden_api"),
    ("return x.sum()", "forbidden_tensor_method"),
    ("return x.exp_()", "forbidden_tensor_method"),
    ("return out.softmax(dim=-1)", "forbidden_tensor_method"),
    ("return x.cpu()", "forbidden_tensor_method"),
    ("return x @ x", "forbidden_operator"),
    ("t = torch\n    return out", "module_escape"),
    ("f = [triton]\n    return out", "module_escape"),
    ("torch.empty = None\n    return out", "module_attribute_assignment"),
    ("triton.cdiv = None\n    return out", "module_attribute_assignment"),
    ("del torch.empty\n    return out", "module_attribute_assignment"),
    ("x.data = out\n    return out", "forbidden_attribute_assignment"),
    ("return getattr(out, 'sum')()", "forbidden_builtin"),
    ("exec('1')\n    return out", "forbidden_builtin"),
    ("eval('1')\n    return out", "forbidden_builtin"),
    ("open('/etc/passwd')\n    return out", "forbidden_builtin"),
    ("return out.__class__", "dunder_access"),
    ("return __builtins__", "forbidden_builtin"),
    ("global G\n    return out", "global_statement"),
])
def test_rejections_in_host_code(snippet, reason):
    src = GOOD.replace("return out.view(x.shape)", snippet)
    r = check_source(src, expected_arity=1)
    assert not r.ok
    assert r.reason == reason, (r.reason, r.message)


@pytest.mark.parametrize("line,reason", [
    ("import os", "forbidden_import"),
    ("import subprocess", "forbidden_import"),
    ("import ctypes", "forbidden_import"),
    ("import importlib", "forbidden_import"),
    ("import time", "forbidden_import"),
    ("import functools", "forbidden_import"),
    ("import torch.nn.functional as F", "forbidden_import"),
    ("from torch import softmax", "forbidden_api"),
    ("from torch.nn import functional", "forbidden_import"),
    ("from triton import testing", "forbidden_api"),
    ("from torch import *", "forbidden_import"),
    ("import triton.runtime", "forbidden_import"),
])
def test_forbidden_imports(line, reason):
    r = check_source(line + "\n" + GOOD, expected_arity=1)
    assert not r.ok and r.reason == reason, (r.reason, r.message)


def test_tensor_methods_allowed_inside_kernels():
    # method-call denylist applies to host code only; Triton tensors may use methods like .to()
    src = GOOD.replace("x = tl.load(x_ptr + offs, mask=m, other=0.0)",
                       "x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)")
    assert check_source(src, expected_arity=1).ok


def test_missing_entrypoint_and_kernel():
    assert check_source(GOOD.replace("def run(", "def go("), expected_arity=1).reason == "missing_entrypoint"
    src = GOOD.replace("@triton.jit\n", "")
    assert check_source(src, expected_arity=1).reason == "no_triton_kernel"


def test_arity():
    assert check_source(GOOD, expected_arity=2).reason == "entrypoint_signature"
    assert reference_arity(load_task("l2_layernorm").reference_source()) == 3


def test_syntax_error_flagged():
    r = check_source("def run(x:\n", expected_arity=1)
    assert not r.ok and r.syntax_error


def test_source_size_limit():
    assert check_source(GOOD + "#" * 70000, max_bytes=65536).reason == "source_too_large"


FIXTURES = all_fixtures()


@pytest.mark.parametrize("fx", FIXTURES, ids=[f.id for f in FIXTURES])
def test_fixture_static_verdict(fx):
    """Fixtures that should die statically must; every other fixture must pass the static check
    (otherwise a runtime test would be testing the wrong layer)."""
    task = load_task(fx.task)
    r = check_source(fx.source, expected_arity=reference_arity(task.reference_source()),
                     allowed_imports=task.allowed_imports)
    if fx.expect == "static_reject" or fx.skip_static:
        assert not r.ok, f"{fx.id} passed the static check"
        if fx.expect == "static_reject" and fx.reasons:
            assert r.reason in fx.reasons, (r.reason, r.message)
    else:
        assert r.ok, f"{fx.id} unexpectedly rejected: {r.reason}: {r.message}"
