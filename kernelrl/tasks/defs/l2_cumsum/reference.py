"""Reference for l2_cumsum: Row cumulative sum.

Provenance: Original KernelRL task (written from scratch). Task category inspired by KernelBench level 2 (github.com/ScalingIntelligence/KernelBench, MIT license); no KernelBench code copied.
"""
import torch


def make_inputs(shape: dict, dtype: torch.dtype, device: str, seed: int):
    """Deterministic inputs for a shape. Generated in float32 then cast to `dtype`."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape['M'], shape['N']
    x = torch.randn(M, N, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x,)


def reference(x):
    return torch.cumsum(x, dim=-1)
