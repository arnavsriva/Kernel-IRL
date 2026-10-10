"""The worked example shown in prompt v2 must be correct Triton (GPU box only)."""
import importlib.util

import pytest

pytestmark = pytest.mark.gpu


def test_prompt_example_kernel_is_correct(tmp_path):
    import torch

    from kernelrl.serving.extract import extract_code
    from kernelrl.serving.prompt import TRITON_GUIDE

    path = tmp_path / "prompt_example.py"  # Triton needs kernel source in a real file
    path.write_text(extract_code(TRITON_GUIDE).code)
    spec = importlib.util.spec_from_file_location("prompt_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for M, N, dt in [(1, 1, torch.float32), (7, 4097, torch.float16), (1024, 1024, torch.float32)]:
        x = torch.randn(M, N, device="cuda", dtype=dt)
        s = torch.randn(M, device="cuda", dtype=dt)
        out = mod.run(x, s)
        ref = (x.float() * s.float()[:, None]).to(dt)
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
