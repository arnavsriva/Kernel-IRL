# TASK: l1_vector_add
# EXPECT: crash
# EXPECT_REASON: file_size_limit
# NOTE: floods stdout; RLIMIT_FSIZE on the child's log file must stop it
import torch
import triton
import triton.language as tl


@triton.jit
def k(p, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(p + offs, tl.load(p + offs, mask=offs < n), mask=offs < n)


def run(x, y):
    line = "x" * 1_000_000
    while True:
        print(line, flush=True)
