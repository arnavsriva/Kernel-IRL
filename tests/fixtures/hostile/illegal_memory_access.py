# TASK: l1_vector_add
# EXPECT: crash
# EXPECT_REASON: cuda_error
# NOTE: stores 2^40 bytes past the output pointer (passes the static check)
import torch
import triton
import triton.language as tl


@triton.jit
def bad_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs + (1 << 40), x, mask=mask)


def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    bad_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out
