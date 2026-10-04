"""Reference for l2_cross_entropy: Per-row cross-entropy.

Provenance: Original KernelRL task (written from scratch). Task category inspired by KernelBench level 2 (github.com/ScalingIntelligence/KernelBench, MIT license); no KernelBench code copied.
"""
import torch


def make_inputs(shape: dict, dtype: torch.dtype, device: str, seed: int):
    """Deterministic inputs for a shape. Generated in float32 then cast to `dtype`."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, V = shape['M'], shape['V']
    logits = (torch.randn(M, V, generator=g, device=device, dtype=torch.float32) * 2.0).to(dtype)
    target = torch.randint(0, V, (M,), generator=g, device=device, dtype=torch.int64)
    return (logits, target,)


def reference(logits, target):
    return torch.nn.functional.cross_entropy(logits, target, reduction='none')
