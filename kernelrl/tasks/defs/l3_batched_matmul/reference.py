"""Reference for l3_batched_matmul: Batched matmul.

Provenance: Original KernelRL task (written from scratch). Task category inspired by KernelBench level 3 (github.com/ScalingIntelligence/KernelBench, MIT license); no KernelBench code copied.
"""
import torch


def make_inputs(shape: dict, dtype: torch.dtype, device: str, seed: int):
    """Deterministic inputs for a shape. Generated in float32 then cast to `dtype`."""
    g = torch.Generator(device=device).manual_seed(seed)
    Bt, M, N, K = shape['B'], shape['M'], shape['N'], shape['K']
    a = torch.randn(Bt, M, K, generator=g, device=device, dtype=torch.float32).to(dtype)
    b = (torch.randn(Bt, K, N, generator=g, device=device, dtype=torch.float32) * K ** -0.5).to(dtype)
    return (a, b,)


def reference(a, b):
    return torch.bmm(a, b)
