"""Task discovery and loading. Importing this module does not import torch;
only `Task.load_module()` does (it executes reference.py)."""
from __future__ import annotations

import importlib.util
import json
import types
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from kernelrl.tasks.schema import validate_hidden_json, validate_task_json

DEFS_DIR = Path(__file__).resolve().parent / "defs"


@dataclass(frozen=True)
class ShapeSpec:
    dims: dict
    hidden: bool
    bench: bool
    note: str = ""

    def key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in self.dims.items())


@dataclass(frozen=True)
class Task:
    id: str
    level: int
    name: str
    description: str
    signature: str
    dim_names: tuple
    dtypes: tuple
    tolerance: dict
    public_shapes: tuple
    provenance: str
    dir: Path
    allowed_imports: tuple | None = None
    tags: tuple = field(default_factory=tuple)

    # ---- public (safe to show the model) ---------------------------------
    @property
    def reference_path(self) -> Path:
        return self.dir / "reference.py"

    def reference_source(self) -> str:
        return self.reference_path.read_text()

    # ---- private (sandbox / validator only) ------------------------------
    def hidden_shapes(self) -> list[ShapeSpec]:
        d = json.loads((self.dir / "hidden_shapes.json").read_text())
        validate_hidden_json(d, list(self.dim_names), list(self.public_shapes), self.id)
        return [ShapeSpec(s["dims"], True, bool(s.get("bench", False)), s.get("note", "")) for s in d["shapes"]]

    def all_shapes(self) -> list[ShapeSpec]:
        pub = [ShapeSpec(dict(s), False, True) for s in self.public_shapes]
        return pub + self.hidden_shapes()

    def load_module(self) -> types.ModuleType:
        """Execute reference.py (imports torch)."""
        spec = importlib.util.spec_from_file_location(f"kernelrl_task_{self.id}", self.reference_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        for fn in ("reference", "make_inputs"):
            if not callable(getattr(mod, fn, None)):
                raise ValueError(f"{self.id}: reference.py must define {fn}()")
        return mod


def _load(dir_: Path) -> Task:
    d = json.loads((dir_ / "task.json").read_text())
    validate_task_json(d, dir_.name)
    return Task(
        id=d["id"], level=d["level"], name=d["name"], description=d["description"],
        signature=d["signature"], dim_names=tuple(d["dim_names"]), dtypes=tuple(d["dtypes"]),
        tolerance=d["tolerance"], public_shapes=tuple(d["public_shapes"]), provenance=d["provenance"],
        dir=dir_, allowed_imports=tuple(d["allowed_imports"]) if d.get("allowed_imports") else None,
        tags=tuple(d.get("tags", [])),
    )


@lru_cache(maxsize=None)
def _all() -> dict[str, Task]:
    tasks = {}
    for p in sorted(DEFS_DIR.iterdir()):
        if (p / "task.json").exists():
            t = _load(p)
            tasks[t.id] = t
    return tasks


def task_ids(level: int | None = None) -> list[str]:
    return [t.id for t in _all().values() if level is None or t.level == level]


def all_tasks(level: int | None = None) -> list[Task]:
    return [_all()[i] for i in task_ids(level)]


def load_task(task_id: str) -> Task:
    try:
        return _all()[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}") from None
