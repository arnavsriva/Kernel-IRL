"""Robust extraction of the candidate file from a model response.

Handles: <think> blocks (closed, unclosed, or missing the opening tag), multiple code blocks
(prefer the last complete one, else concatenate), unclosed fences from truncated responses,
and missing fences (trim surrounding prose until the code parses).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)(?:\n[ \t]*```|\Z)", re.S)
PY_LANGS = {"python", "py", "python3", ""}
CODE_START = re.compile(r"^(import |from |@triton|def |class |[A-Za-z_]\w* = )")
ENTRY = "run"


@dataclass
class Extraction:
    code: str | None
    method: str  # fenced | fenced_concat | unfenced | fenced_incomplete | none
    error: str | None = None
    had_thinking: bool = False
    n_blocks: int = 0


def strip_thinking(text: str) -> tuple[str, bool]:
    had = "<think>" in text or "</think>" in text
    text = THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag was in the prompt template
        text = text.split("</think>")[-1]
    if "<think>" in text:  # unterminated thinking (truncated): nothing after it is an answer
        text = text.split("<think>")[0]
    return text, had


def _complete(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    has_entry = any(isinstance(n, ast.FunctionDef) and n.name == ENTRY for n in tree.body)
    return has_entry and "triton" in code


def _unfenced(body: str) -> str | None:
    lines = body.splitlines()
    starts = [i for i, ln in enumerate(lines) if CODE_START.match(ln)]
    if not starts:
        return None
    start = starts[0]
    for end in range(len(lines), start, -1):
        if len(lines) - end > 400:
            break
        code = "\n".join(lines[start:end])
        if _complete(code):
            return code
    return None


def extract_code(response: str | None) -> Extraction:
    body, had = strip_thinking(response or "")
    blocks = FENCE_RE.findall(body)
    py = [c for lang, c in blocks if lang.lower() in PY_LANGS] or [c for _, c in blocks]
    for c in reversed(py):
        if _complete(c):
            return Extraction(c.strip() + "\n", "fenced", None, had, len(blocks))
    if len(py) > 1:
        joined = "\n\n".join(c.strip() for c in py)
        if _complete(joined):
            return Extraction(joined + "\n", "fenced_concat", None, had, len(blocks))
    if not blocks:
        code = _unfenced(body)
        if code:
            return Extraction(code.strip() + "\n", "unfenced", None, had, 0)
    if py:  # let the static check report what is wrong with it
        best = max(py, key=lambda c: (f"def {ENTRY}" in c, len(c)))
        return Extraction(best.strip() + "\n", "fenced_incomplete", "no complete code block with run()", had,
                          len(blocks))
    return Extraction(None, "none", "no code found in the response", had, 0)
