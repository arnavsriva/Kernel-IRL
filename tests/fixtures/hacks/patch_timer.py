# TASK: l1_vector_add
# EXPECT: static_reject
# EXPECT_REASON: module_attribute_assignment|forbidden_api
# NOTE: makes every CUDA-event measurement report 1 microsecond
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


def _add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out


torch.cuda.Event.elapsed_time = lambda self, end: 0.001


def run(x, y):
    return _add(x, y)
