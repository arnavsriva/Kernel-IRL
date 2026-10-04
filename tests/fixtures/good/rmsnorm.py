# TASK: l2_rmsnorm
# EXPECT: correct
# NOTE: fp32 statistics, two passes, fp16 output
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, stride_x, stride_o, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    acc = tl.zeros([BLOCK], tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * stride_x + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        acc += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(out_ptr + row * stride_o + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def run(x, weight):
    M, N = x.shape
    out = torch.empty_like(x)
    rmsnorm_kernel[(M,)](x, weight, out, N, x.stride(0), out.stride(0), 1e-6, BLOCK=1024, num_warps=4)
    return out
