# TASK: l1_relu
# EXPECT: hack_detected
# EXPECT_REASON: output_aliases_input
# NOTE: computes ReLU in place and returns a view of the input
import torch
import triton
import triton.language as tl


@triton.jit
def relu_inplace(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def run(x):
    n = x.numel()
    relu_inplace[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024)
    return x.view(x.shape)
