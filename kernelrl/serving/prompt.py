"""Prompt templates. Only public task information goes in; hidden shapes are never read here."""
from __future__ import annotations

import ast

from kernelrl.sandbox.static_check import ALLOWED_IMPORTS
from kernelrl.tasks.registry import Task

SYSTEM = ("You are an expert GPU kernel engineer. You write correct, fast Triton kernels for NVIDIA GPUs "
          "and follow the output format exactly.")

USER_TMPL = """Write an optimized Triton implementation of the following PyTorch operation.

## Task: {name}
{description}

## Reference implementation (PyTorch)
```python
{reference}
```

## Interface
- Signature: `{signature}`
- Input dtype: {dtypes}. Inputs are contiguous CUDA tensors produced by `make_inputs` above.
- Example shapes: {shapes}
- Your kernel is also tested on other, unseen shapes (tiny, non-power-of-2, and large), so it must handle any size.
- Tolerance: {tolerance}. Speed is measured against PyTorch eager and torch.compile.

## Rules
1. Reply with one complete Python file in a single ```python code block.
2. Define `def run({args})` returning a new tensor with the same shape and dtype as `reference(...)`.
3. Do the computation in one or more `@triton.jit` kernels. Host code may only allocate outputs (e.g. `torch.empty`, `torch.empty_like`, `torch.zeros`), read shapes/strides, compute launch grids, and launch kernels.
4. Allowed imports: {imports}. Do not call PyTorch compute ops (e.g. `torch.softmax`, `x.sum()`, `@`), `torch.compile`, or any file, network, or process APIs.
5. Do not keep state between calls (no caching of inputs or outputs), and do not return a view of an input.
"""

# Prompt v2 additions: a Triton API cheat-sheet and one worked example. The example operation
# (per-row scaling) is deliberately not in the task set. Smoke run 2026-10-10 showed Qwen3-4B
# zero-shot inventing APIs (triton.launch, tl.num_blocks_per_grid) and CUDA-style indexing.
TRITON_GUIDE = """
## Triton quick reference (triton 3.6)
- A kernel is a `@triton.jit` function. Each *program* processes a BLOCK of elements, not one thread per element.
- Pass tensors directly as pointer arguments (not `x.data_ptr()`); pass sizes/strides as ints; block sizes as
  `BLOCK: tl.constexpr` (powers of 2, needed by `tl.arange`).
- Indexing: `pid = tl.program_id(0)`, `offs = pid * BLOCK + tl.arange(0, BLOCK)`, `mask = offs < n`.
- Memory: `tl.load(ptr + offs, mask=mask, other=0.0)`, `tl.store(ptr + offs, value, mask=mask)`.
- Math: `tl.exp`, `tl.log`, `tl.sqrt`, `tl.rsqrt`, `tl.erf`, `tl.sigmoid`, `tl.maximum`, `tl.minimum`, `tl.where`, `tl.abs`;
  cast with `x.to(tl.float32)`.
- Reductions over a block: `tl.sum(x, axis=0)`, `tl.max(x, axis=0)`, `tl.min(x, axis=0)`. For rows longer than one block,
  loop: `for start in range(0, n_cols, BLOCK): ...`.
- Matmul tiles: `acc = tl.zeros((BM, BN), dtype=tl.float32)`; `acc += tl.dot(a_tile, b_tile)` (tile dims >= 16).
- Host side: `triton.cdiv(n, BLOCK)`, `triton.next_power_of_2(n)`, `x.stride(0)`; launch with
  `kernel[(num_programs,)](x, out, n, BLOCK=1024)` or a 2D grid tuple; optional `num_warps=4`.
- These do NOT exist: `triton.launch`, `triton.grid`, `tl.launch_index`, `tl.num_blocks_per_grid`, `tl.device_ptr32`.

## Worked example (a different operation: scale each row by a per-row factor)
```python
import torch
import triton
import triton.language as tl


@triton.jit
def row_scale_kernel(x_ptr, scale_ptr, out_ptr, n_cols, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    s = tl.load(scale_ptr + row).to(tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * stride_o + cols, (x * s).to(out_ptr.dtype.element_ty), mask=mask)


def run(x, scale):
    M, N = x.shape
    out = torch.empty_like(x)
    row_scale_kernel[(M,)](x, scale, out, N, x.stride(0), out.stride(0), BLOCK=1024)
    return out
```
"""

PROMPT_VERSIONS = ("v1", "v2")

FEEDBACK_TMPL = """{feedback}

Revise your kernel based on this result. Reply with the complete updated Python file in a single ```python code block."""


def reference_args(task: Task) -> list[str]:
    for node in ast.parse(task.reference_source()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "reference":
            return [a.arg for a in node.args.args]
    raise ValueError(f"{task.id}: no reference()")


def _shape(d: dict) -> str:
    return "(" + ", ".join(f"{k}={v}" for k, v in d.items()) + ")"


def user_prompt(task: Task, version: str = "v2") -> str:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version {version!r}")
    tol = "; ".join(f"{dt}: atol={t['atol']:g}, rtol={t['rtol']:g}" for dt, t in task.tolerance.items())
    text = USER_TMPL.format(
        name=task.name, description=task.description, reference=task.reference_source().strip(),
        signature=task.signature, dtypes=", ".join(task.dtypes),
        shapes=", ".join(_shape(s) for s in task.public_shapes), tolerance=tol,
        args=", ".join(reference_args(task)), imports=", ".join(task.allowed_imports or ALLOWED_IMPORTS),
    )
    return text + TRITON_GUIDE if version == "v2" else text


def initial_messages(task: Task, version: str = "v2") -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(task, version)}]


def revision_messages(task: Task, first_response: str, feedback: str, version: str = "v2") -> list[dict]:
    """Turn 2: the model's previous answer (thinking stripped) + sandbox feedback."""
    from kernelrl.serving.extract import strip_thinking

    answer, _ = strip_thinking(first_response)
    return initial_messages(task, version) + [
        {"role": "assistant", "content": answer.strip()},
        {"role": "user", "content": FEEDBACK_TMPL.format(feedback=feedback)},
    ]
