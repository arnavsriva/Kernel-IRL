"""Render an EvalResult as the text the model sees on its next turn.

Invariant: hidden-shape dimensions never appear in feedback. Failures on hidden shapes are
reported as "a hidden test shape", and any large integer that is a hidden-only dimension is
redacted from error text (tracebacks can echo sizes).
"""
from __future__ import annotations

import re

from kernelrl.common.io import truncate
from kernelrl.common.schemas import EvalResult, Status
from kernelrl.tasks.registry import Task

_INT = re.compile(r"(?<![\w.])\d+(?![\w.])")
REDACT_MIN = 10  # tiny dims (1, 3, 5, 7) are not secrets and collide with line numbers

_RULES_HINT = ("Rules: implement the computation in @triton.jit kernels; host code may only allocate "
               "outputs, compute launch grids, and launch kernels. Allowed imports: torch, triton, "
               "triton.language, math, typing.")


def hidden_only_values(task: Task) -> set[int]:
    pub = {v for s in task.public_shapes for v in s.values()}
    hid = {v for s in task.hidden_shapes() for v in s.dims.values()}
    return {v for v in hid - pub if v >= REDACT_MIN}


def redact(text: str, secret: set[int]) -> str:
    if not text or not secret:
        return text or ""
    return _INT.sub(lambda m: "<hidden>" if int(m.group()) in secret else m.group(), text)


def _shape_str(dims: dict | None, hidden: bool) -> str:
    if hidden or dims is None:
        return "a hidden test shape"
    return "shape " + ", ".join(f"{k}={v}" for k, v in dims.items())


def render_feedback(result: EvalResult, task: Task, cfg: dict) -> str:
    fc = cfg["feedback"]
    secret = hidden_only_values(task)
    msg = redact(truncate(result.message, fc["max_error_chars"]), secret)
    where = _shape_str(result.meta.get("failure_shape"), result.meta.get("failure_hidden", False))
    s = result.status
    out = [f"## Evaluation result: {s.value.upper()}"]

    if s is Status.STATIC_REJECT:
        out += [f"Your code was rejected before running: {msg}", _RULES_HINT]
    elif s is Status.COMPILE_ERROR:
        out += [f"Compilation/import failed{'' if result.stage == 'static' else ' on ' + where}:", f"```\n{msg}\n```"]
    elif s is Status.TIMEOUT:
        out += [f"Your kernel exceeded the time limit during the {result.stage} stage. Check for infinite loops "
                "or loop bounds that do not terminate."]
    elif s is Status.CRASH:
        out += [f"Your kernel crashed{'' if not result.meta.get('failure_shape') else ' on ' + where}:",
                f"```\n{msg}\n```", "Common causes: out-of-bounds loads/stores (missing masks), wrong strides or "
                "pointer arithmetic, or excessive memory allocation."]
    elif s is Status.HACK_DETECTED:
        out += [f"Rejected by a verification check ({result.reason}) on {where}: {msg}",
                "The output must be computed by your Triton kernel(s) from the given inputs on every call, "
                "returned in a fresh tensor, without caching results or modifying library functions."]
    elif s is Status.INCORRECT:
        out += [f"Wrong output on {where}: {msg}"]
        out += _correctness_summary(result, fc["max_shapes_listed"])
    elif s is Status.CORRECT:
        out += ["All correctness checks passed."]
        out += _timing_summary(result)
    return "\n".join(out)


def _correctness_summary(result: EvalResult, max_listed: int) -> list[str]:
    pub = {}
    hid_total = hid_fail = 0
    for sr in result.shape_results:
        if sr.hidden:
            hid_total += 1
            hid_fail += not sr.passed
        else:
            k = ", ".join(f"{a}={b}" for a, b in sr.shape.items())
            pub[k] = pub.get(k, True) and sr.passed
    lines = ["Public shapes:"] + [f"  - {k}: {'pass' if ok else 'FAIL'}" for k, ok in list(pub.items())[:max_listed]]
    if hid_total:
        lines.append(f"Hidden shapes: {hid_total - hid_fail}/{hid_total} checks passed "
                     "(hidden shapes include tiny, non-power-of-2, and large sizes).")
    return lines


def _timing_summary(result: EvalResult) -> list[str]:
    lines = []
    pub = [t for t in result.timings if not t.hidden]
    if pub:
        lines.append("Timing on public shapes (median, lower is better):")
        for t in pub:
            k = ", ".join(f"{a}={b}" for a, b in t.shape.items())
            e = f"{t.eager.median_ms:.4f} ms" if t.eager else "n/a"
            c = f"{t.compiled.median_ms:.4f} ms" if t.compiled else "n/a"
            sp = f"{t.speedup_vs_eager:.2f}x" if t.speedup_vs_eager else "n/a"
            lines.append(f"  - {k}: yours {t.candidate.median_ms:.4f} ms | PyTorch eager {e} | "
                         f"torch.compile {c} | speedup vs eager {sp}")
    if result.speedup_vs_eager is not None:
        lines.append(f"Overall speedup vs eager (geomean incl. hidden shapes): {result.speedup_vs_eager:.2f}x")
    if result.speedup_vs_compile is not None:
        lines.append(f"Overall speedup vs torch.compile: {result.speedup_vs_compile:.2f}x")
    lines.append("You may revise the kernel to make it faster while keeping it correct.")
    return lines
