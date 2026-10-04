"""Reference for l3_matmul: Matmul.

Provenance: Original KernelRL task (written from scratch). Task category inspired by KernelBench level 3 (github.com/ScalingIntelligence/KernelBench, MIT license); no KernelBench code copied.
"""
import torch


def make_inputs(shape: dict, dtype: torch.dtype, device: str, seed: int):
    """Deterministic inputs for a shape. Generated in float32 then cast to `dtype`."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = shape['M'], shape['N'], shape['K']
    a = torch.randn(M, K, generator=g, device=device, dtype=torch.float32).to(dtype)
    b = (torch.randn(K, N, generator=g, device=device, dtype=torch.float32) * K ** -0.5).to(dtype)
    return (a, b,)


def reference(a, b):
    return torch.matmul(a, b)
