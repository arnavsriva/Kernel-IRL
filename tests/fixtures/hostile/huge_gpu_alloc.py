# TASK: l1_vector_add
# EXPECT: crash
# EXPECT_REASON: cuda_oom
# NOTE: asks for 1 TiB of device memory (passes the static check)
import torch
import triton
import triton.language as tl


@triton.jit
def k(p, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(p + offs, tl.load(p + offs, mask=offs < n), mask=offs < n)


def run(x, y):
    scratch = torch.empty((1 << 40,), device=x.device, dtype=torch.uint8)
    k[(1,)](scratch, 1, BLOCK=16)
    return x
