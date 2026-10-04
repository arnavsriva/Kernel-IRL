# TASK: l2_softmax
# EXPECT: correct
# NOTE: one program per row, three passes over BLOCK-sized chunks (max, sum, write); any N
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x
    o_row = out_ptr + row * stride_o
    m_acc = tl.full([BLOCK], float("-inf"), tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=float("-inf"))
        m_acc = tl.maximum(m_acc, x)
    row_max = tl.max(m_acc, axis=0)
    s_acc = tl.zeros([BLOCK], tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=float("-inf"))
        s_acc += tl.exp(x - row_max)
    row_sum = tl.sum(s_acc, axis=0)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float("-inf"))
        tl.store(o_row + cols, tl.exp(x - row_max) / row_sum, mask=mask)


def run(x):
    M, N = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(M,)](x, out, N, x.stride(0), out.stride(0), BLOCK=1024, num_warps=4)
    return out
