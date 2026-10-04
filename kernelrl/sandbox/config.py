from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    return deep_merge(_load(str(path or DEFAULT_PATH)), overrides)
