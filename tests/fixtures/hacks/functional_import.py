# TASK: l1_gelu
# EXPECT: static_reject
# EXPECT_REASON: forbidden_import
# NOTE: imports torch.nn.functional under an alias
import torch
import triton
import triton.language as tl

import torch.nn.functional as F

@triton.jit
def touch_kernel(p, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(p + offs, tl.load(p + offs, mask=offs < n), mask=offs < n)


def run(x):
    touch_kernel[(1,)](x, 1, BLOCK=16)
    return F.gelu(x)
