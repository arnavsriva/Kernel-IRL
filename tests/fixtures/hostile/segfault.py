# TASK: l1_vector_add
# EXPECT: crash
# EXPECT_REASON: killed_by_SIGSEGV
# SKIP_STATIC: true
# NOTE: dereferences NULL at import time; the static check rejects ctypes, this proves the process boundary holds anyway
import ctypes

import torch
import triton
import triton.language as tl

ctypes.string_at(0)


@triton.jit
def k(p, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(p + offs, tl.load(p + offs, mask=offs < n), mask=offs < n)


def run(x, y):
    return x
