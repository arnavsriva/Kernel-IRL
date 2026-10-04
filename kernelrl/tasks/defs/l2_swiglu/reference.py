"""Reference for l2_swiglu: SwiGLU gate.

Provenance: Original KernelRL task (written from scratch). Task category inspired by KernelBench level 2 (github.com/ScalingIntelligence/KernelBench, MIT license); no KernelBench code copied.
"""
import torch


def make_inputs(shape: dict, dtype: torch.dtype, device: str, seed: int):
    """Deterministic inputs for a shape. Generated in float32 then cast to `dtype`."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape['M'], shape['N']
    gate = torch.randn(M, N, generator=g, device=device, dtype=torch.float32).to(dtype)
    up = torch.randn(M, N, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (gate, up,)


def reference(gate, up):
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)
