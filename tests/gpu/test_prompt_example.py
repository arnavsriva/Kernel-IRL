"""The worked example shown in prompt v2 must be correct Triton (GPU box only)."""
import pytest

pytestmark = pytest.mark.gpu


def test_prompt_example_kernel_is_correct():
    import torch

    from kernelrl.serving.extract import extract_code
    from kernelrl.serving.prompt import TRITON_GUIDE

    ns: dict = {}
    exec(compile(extract_code(TRITON_GUIDE).code, "prompt_example", "exec"), ns)  # trusted repo text
    for M, N, dt in [(1, 1, torch.float32), (7, 4097, torch.float16), (1024, 1024, torch.float32)]:
        x = torch.randn(M, N, device="cuda", dtype=dt)
        s = torch.randn(M, device="cuda", dtype=dt)
        out = ns["run"](x, s)
        ref = (x.float() * s.float()[:, None]).to(dt)
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
