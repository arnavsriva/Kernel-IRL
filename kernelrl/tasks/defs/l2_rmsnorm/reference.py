"""Reference for l2_rmsnorm: RMSNorm.

Provenance: Original KernelRL task (written from scratch). Task category inspired by KernelBench level 2 (github.com/ScalingIntelligence/KernelBench, MIT license); no KernelBench code copied.
"""
import torch


def make_inputs(shape: dict, dtype: torch.dtype, device: str, seed: int):
    """Deterministic inputs for a shape. Generated in float32 then cast to `dtype`."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape['M'], shape['N']
    x = torch.randn(M, N, generator=g, device=device, dtype=torch.float32).to(dtype)
    weight = ((torch.randn(N, generator=g, device=device, dtype=torch.float32) * 0.1) + 1.0).to(dtype)
    return (x, weight,)


def reference(x, weight):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return (out * weight.float()).to(x.dtype)
