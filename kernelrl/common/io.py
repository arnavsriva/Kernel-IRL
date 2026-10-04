from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def truncate(text: str | None, limit: int) -> str:
    """Truncate keeping head and tail (tracebacks put the useful line last)."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = limit * 2 // 5
    tail = limit - head - 40
    return f"{text[:head]}\n... [{len(text) - head - tail} chars truncated] ...\n{text[-tail:]}"
