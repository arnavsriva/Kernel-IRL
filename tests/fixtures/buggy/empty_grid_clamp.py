# TASK: l1_clamp
# EXPECT: incorrect
# NOTE: from the Qwen3-4B baseline: grid = next_power_of_2(n) // 1024 is 0 for n < 1024, so Triton launches
#       nothing on tiny shapes and the output stays uninitialized. Must be 'incorrect', not an infra error.
import torch
import triton
import triton.language as tl


@triton.jit
def clamp_kernel(x_ptr, out_ptr, n, lo, hi, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.minimum(tl.maximum(x, lo), hi), mask=mask)


def run(x):
    n = x.numel()
    out = torch.empty_like(x)
    num_programs = triton.next_power_of_2(n) // 1024
    clamp_kernel[(num_programs,)](x, out, n, -0.5, 0.5, BLOCK=1024)
    return out
