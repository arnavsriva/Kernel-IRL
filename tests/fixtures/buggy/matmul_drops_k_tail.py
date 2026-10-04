# TASK: l3_matmul
# EXPECT: incorrect
# NOTE: K loop stops at the last full BK tile; public shapes (K multiple of 32) pass, hidden odd K fail
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K - BK + 1, BK):
        kk = k0 + rk
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak, mask=rm[:, None] < M, other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn, mask=rn[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(c_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def run(a, b):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), BM=64, BN=64, BK=32)
    return c
