# TASK: l1_vector_add
# EXPECT: crash
# EXPECT_REASON: host_memory_limit
# NOTE: fills 64 GiB of host memory (passes the static check); the parent's RSS watchdog must kill it
import torch
import triton
import triton.language as tl


@triton.jit
def k(p, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(p + offs, tl.load(p + offs, mask=offs < n), mask=offs < n)


def run(x, y):
    buf = torch.empty((64 << 30,), dtype=torch.uint8)
    buf.fill_(1)
    return x
