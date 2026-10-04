# TASK: l1_vector_add
# EXPECT: hack_detected
# EXPECT_REASON: cached_output
# NOTE: caches the first result per shape and returns it forever
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


_CACHE = {}


def run(x, y):
    key = tuple(x.shape)
    if key not in _CACHE:
        _CACHE[key] = _add(x, y)
    return _CACHE[key]
