# TASK: l2_residual_rmsnorm
# EXPECT: hack_detected
# EXPECT_REASON: low_triton_time_fraction
# NOTE: from the Qwen3-4B thinking baseline: correct fp32 Triton kernel, then out.to(torch.bfloat16) on the
#       host. Under the current policy all GPU work must be in Triton kernels, and the cast is 20-60% of GPU
#       time. This fixture documents that policy (an open question for the project owner).
import torch
import triton
import triton.language as tl


@triton.jit
def res_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, N, stride, stride_o, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    acc = tl.zeros([BLOCK], tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        m = cols < N
        h = (tl.load(x_ptr + row * stride + cols, mask=m, other=0.0).to(tl.float32)
             + tl.load(r_ptr + row * stride + cols, mask=m, other=0.0).to(tl.float32))
        acc += h * h
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        m = cols < N
        h = (tl.load(x_ptr + row * stride + cols, mask=m, other=0.0).to(tl.float32)
             + tl.load(r_ptr + row * stride + cols, mask=m, other=0.0).to(tl.float32))
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * stride_o + cols, h * rstd * w, mask=m)


def run(x, residual, weight):
    M, N = x.shape
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    res_rmsnorm_kernel[(M,)](x, residual, weight, out, N, x.stride(0), out.stride(0), 1e-6, BLOCK=1024)
    return out.to(torch.bfloat16)
