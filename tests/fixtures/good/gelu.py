# TASK: l1_gelu
# EXPECT: correct
# NOTE: exact erf GELU in fp32
import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offs, y, mask=mask)


def run(x):
    out = torch.empty_like(x)
    n = x.numel()
    gelu_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out
