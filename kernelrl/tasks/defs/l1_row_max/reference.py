"""Reference for l1_row_max: Row max.

Provenance: Original KernelRL task (written from scratch). Task category inspired by KernelBench level 1 (github.com/ScalingIntelligence/KernelBench, MIT license); no KernelBench code copied.
"""
import torch


def make_inputs(shape: dict, dtype: torch.dtype, device: str, seed: int):
    """Deterministic inputs for a shape. Generated in float32 then cast to `dtype`."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape['M'], shape['N']
    # Shifted so every row is negative: a kernel that pads masked lanes with 0 or starts its running
    # max at 0 is wrong on every shape with a partial block, not just by chance (seen in the baseline).
    x = (torch.randn(M, N, generator=g, device=device, dtype=torch.float32) - 6.0).to(dtype)
    return (x,)


def reference(x):
    return x.amax(dim=-1)
