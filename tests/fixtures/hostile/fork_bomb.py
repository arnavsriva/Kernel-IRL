# TASK: l1_vector_add
# EXPECT: static_reject
# EXPECT_REASON: forbidden_import
# RUNTIME: false
# NOTE: never executed; the static check is the only layer for process creation (see DESIGN.md)
import os

import torch
import triton
import triton.language as tl


@triton.jit
def k(p, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(p + offs, tl.load(p + offs, mask=offs < n), mask=offs < n)


def run(x, y):
    while True:
        os.fork()
