"""Task schema and validation of task.json / hidden_shapes.json.

Layout per task (kernelrl/tasks/defs/<task_id>/):
  task.json          public metadata, shown (in part) to the model
  reference.py       make_inputs(shape, dtype, device, seed) and reference(*inputs)
  hidden_shapes.json never shown to the model; loaded only by the sandbox/validator
"""
from __future__ import annotations

REQUIRED_TASK_KEYS = {
    "id": str,
    "level": int,
    "name": str,
    "description": str,
    "signature": str,
    "dim_names": list,
    "dtypes": list,
    "tolerance": dict,
    "public_shapes": list,
    "provenance": str,
}
OPTIONAL_TASK_KEYS = {"allowed_imports": (list, type(None)), "tags": list}
VALID_DTYPES = {"float32", "float16", "bfloat16"}


class TaskSchemaError(ValueError):
    pass


def _check_dims(dims: dict, dim_names: list, where: str) -> None:
    if not isinstance(dims, dict) or set(dims) != set(dim_names):
        raise TaskSchemaError(f"{where}: shape keys {sorted(dims)} != dim_names {sorted(dim_names)}")
    for k, v in dims.items():
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise TaskSchemaError(f"{where}: dim {k}={v!r} must be a positive int")


def validate_task_json(d: dict, dir_name: str) -> None:
    for k, typ in REQUIRED_TASK_KEYS.items():
        if k not in d:
            raise TaskSchemaError(f"{dir_name}: missing key {k!r}")
        if not isinstance(d[k], typ):
            raise TaskSchemaError(f"{dir_name}: key {k!r} must be {typ.__name__}")
    unknown = set(d) - set(REQUIRED_TASK_KEYS) - set(OPTIONAL_TASK_KEYS)
    if unknown:
        raise TaskSchemaError(f"{dir_name}: unknown keys {sorted(unknown)}")
    if d["id"] != dir_name:
        raise TaskSchemaError(f"{dir_name}: id {d['id']!r} does not match directory")
    if d["level"] not in (1, 2, 3):
        raise TaskSchemaError(f"{dir_name}: level must be 1..3")
    if not d["dtypes"] or not set(d["dtypes"]) <= VALID_DTYPES:
        raise TaskSchemaError(f"{dir_name}: dtypes must be a non-empty subset of {VALID_DTYPES}")
    for dt in d["dtypes"]:
        tol = d["tolerance"].get(dt)
        if not tol or set(tol) != {"atol", "rtol"} or min(tol.values()) < 0:
            raise TaskSchemaError(f"{dir_name}: tolerance for {dt} must be {{atol, rtol}} >= 0")
    if not d["public_shapes"]:
        raise TaskSchemaError(f"{dir_name}: needs at least one public shape")
    for i, s in enumerate(d["public_shapes"]):
        _check_dims(s, d["dim_names"], f"{dir_name} public_shapes[{i}]")


def validate_hidden_json(d: dict, dim_names: list, public: list, dir_name: str) -> None:
    shapes = d.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise TaskSchemaError(f"{dir_name}: hidden_shapes.json needs a non-empty 'shapes' list")
    pub = {tuple(sorted(p.items())) for p in public}
    for i, s in enumerate(shapes):
        if set(s) - {"dims", "bench", "note"} or "dims" not in s:
            raise TaskSchemaError(f"{dir_name}: hidden shape {i} must have dims (+ optional bench, note)")
        _check_dims(s["dims"], dim_names, f"{dir_name} hidden[{i}]")
        if tuple(sorted(s["dims"].items())) in pub:
            raise TaskSchemaError(f"{dir_name}: hidden shape {s['dims']} is also public")
