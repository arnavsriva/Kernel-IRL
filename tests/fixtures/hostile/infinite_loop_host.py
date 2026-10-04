# TASK: l1_vector_add
# EXPECT: timeout
# NOTE: host-side infinite loop inside run()
import torch
import triton
import triton.language as tl


@triton.jit
def k(p, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(p + offs, tl.load(p + offs, mask=offs < n), mask=offs < n)


def run(x, y):
    i = 0
    while True:
        i += 1
    return x
