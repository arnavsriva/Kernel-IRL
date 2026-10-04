# TASK: l2_softmax
# EXPECT: incorrect
# NOTE: assumes N <= 1024: only the first block of each row is normalized and written
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=float("-inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out_ptr + row * stride + cols, e / tl.sum(e, axis=0), mask=mask)


def run(x):
    M, N = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(M,)](x, out, N, x.stride(0), BLOCK=1024)
    return out
