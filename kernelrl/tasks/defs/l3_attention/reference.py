"""Reference for l3_attention: Attention (non-causal).

Provenance: Original KernelRL task (written from scratch). Task category inspired by KernelBench level 3 (github.com/ScalingIntelligence/KernelBench, MIT license); no KernelBench code copied.
"""
import torch


def make_inputs(shape: dict, dtype: torch.dtype, device: str, seed: int):
    """Deterministic inputs for a shape. Generated in float32 then cast to `dtype`."""
    g = torch.Generator(device=device).manual_seed(seed)
    B, H, S, D = shape['B'], shape['H'], shape['S'], shape['D']
    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=torch.float32).to(dtype)
    k = torch.randn(B, H, S, D, generator=g, device=device, dtype=torch.float32).to(dtype)
    v = torch.randn(B, H, S, D, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (q, k, v,)


def reference(q, k, v):
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)
