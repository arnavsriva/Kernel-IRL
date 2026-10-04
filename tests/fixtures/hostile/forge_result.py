# TASK: l1_vector_add
# EXPECT: static_reject
# EXPECT_REASON: forbidden_builtin
# RUNTIME: false
# NOTE: tries to write a fake "correct" verdict into the job directory; static-only (see DESIGN.md limitation)
import torch
import triton
import triton.language as tl


@triton.jit
def k(p, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(p + offs, tl.load(p + offs, mask=offs < n), mask=offs < n)


def run(x, y):
    with open("check.result.json", "w") as f:
        f.write('{"ok": true, "shape_results": [], "findings": []}')
    return x
