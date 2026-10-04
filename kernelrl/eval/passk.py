from __future__ import annotations

from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if n < k:
        raise ValueError(f"need n >= k (n={n}, k={k})")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
