# TASK: l1_row_sum
# EXPECT: incorrect
# NOTE: program m sums column m instead of row m (loads masked, so no out-of-bounds access)
import torch
import triton
import triton.language as tl


@triton.jit
def colsum_kernel(x_ptr, out_ptr, M, N, stride, BLOCK: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    acc = tl.zeros([BLOCK], tl.float32)
    for start in range(0, M, BLOCK):
        rows = start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + rows * stride + m, mask=(rows < M) & (m < N), other=0.0)
    tl.store(out_ptr + m, tl.sum(acc, axis=0))


def run(x):
    M, N = x.shape
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    colsum_kernel[(M,)](x, out, M, N, x.stride(0), BLOCK=1024)
    return out
