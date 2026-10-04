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

FEEDBACK_TMPL = """{feedback}

Revise your kernel based on this result. Reply with the complete updated Python file in a single ```python code block."""


def reference_args(task: Task) -> list[str]:
    for node in ast.parse(task.reference_source()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "reference":
            return [a.arg for a in node.args.args]
    raise ValueError(f"{task.id}: no reference()")


def _shape(d: dict) -> str:
    return "(" + ", ".join(f"{k}={v}" for k, v in d.items()) + ")"


def user_prompt(task: Task) -> str:
    tol = "; ".join(f"{dt}: atol={t['atol']:g}, rtol={t['rtol']:g}" for dt, t in task.tolerance.items())
    return USER_TMPL.format(
        name=task.name, description=task.description, reference=task.reference_source().strip(),
        signature=task.signature, dtypes=", ".join(task.dtypes),
        shapes=", ".join(_shape(s) for s in task.public_shapes), tolerance=tol,
        args=", ".join(reference_args(task)), imports=", ".join(task.allowed_imports or ALLOWED_IMPORTS),
    )


def initial_messages(task: Task) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(task)}]


def revision_messages(task: Task, first_response: str, feedback: str) -> list[dict]:
    """Turn 2: the model's previous answer (thinking stripped) + sandbox feedback."""
    from kernelrl.serving.extract import strip_thinking

    answer, _ = strip_thinking(first_response)
    return initial_messages(task) + [
        {"role": "assistant", "content": answer.strip()},
        {"role": "user", "content": FEEDBACK_TMPL.format(feedback=feedback)},
    ]
