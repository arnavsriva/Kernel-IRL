# TASK: l1_vector_add
# EXPECT: timeout
# NOTE: device-side infinite loop (volatile load of a flag that never changes); the GPU must recover
import torch
import triton
import triton.language as tl


@triton.jit
def spin_kernel(flag_ptr, out_ptr):
    f = tl.load(flag_ptr, volatile=True)
    while f > 0:
        f = tl.load(flag_ptr, volatile=True)
    tl.store(out_ptr, f)


def run(x, y):
    flag = torch.ones((1,), device=x.device, dtype=torch.int32)
    out = torch.empty_like(x)
    spin_kernel[(1,)](flag, out)
    return out
