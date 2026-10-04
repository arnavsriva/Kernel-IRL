"""Reference for l2_layernorm: LayerNorm.

Provenance: Original KernelRL task (written from scratch). Task category inspired by KernelBench level 2 (github.com/ScalingIntelligence/KernelBench, MIT license); no KernelBench code copied.
"""
import torch


def make_inputs(shape: dict, dtype: torch.dtype, device: str, seed: int):
    """Deterministic inputs for a shape. Generated in float32 then cast to `dtype`."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape['M'], shape['N']
    x = torch.randn(M, N, generator=g, device=device, dtype=torch.float32).to(dtype)
    weight = ((torch.randn(N, generator=g, device=device, dtype=torch.float32) * 0.1) + 1.0).to(dtype)
    bias = (torch.randn(N, generator=g, device=device, dtype=torch.float32) * 0.1).to(dtype)
    return (x, weight, bias,)


def reference(x, weight, bias):
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps=1e-5).to(x.dtype)
