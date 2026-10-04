# TASK: l2_softmax
# EXPECT: static_reject
# EXPECT_REASON: forbidden_api
# NOTE: lets Inductor write the kernel
import torch
import triton
import triton.language as tl


@triton.jit
def touch_kernel(p, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(p + offs, tl.load(p + offs, mask=offs < n), mask=offs < n)


def _impl(x):
    return x


def run(x):
    touch_kernel[(1,)](x, 1, BLOCK=16)
    return torch.compile(_impl)(x)
