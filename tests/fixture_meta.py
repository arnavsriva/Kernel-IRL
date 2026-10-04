"""Parse adversarial fixture headers (see tests/fixtures/README.md)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class Fixture:
    path: Path
    category: str
    task: str
    expect: str
    reasons: tuple[str, ...]
    skip_static: bool
    runtime: bool
    needs_bench: bool

    @property
    def id(self) -> str:
        return f"{self.category}/{self.path.stem}"

    @property
    def source(self) -> str:
        return self.path.read_text()


def _parse(path: Path) -> Fixture:
    meta = {}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        key, _, val = line[1:].strip().partition(":")
        meta[key.strip().upper()] = val.strip()
    for k in ("TASK", "EXPECT"):
        if k not in meta:
            raise ValueError(f"{path}: missing '# {k}:' header")
    truthy = lambda k, d: meta.get(k, d).lower() == "true"  # noqa: E731
    return Fixture(path, path.parent.name, meta["TASK"], meta["EXPECT"],
                   tuple(r for r in meta.get("EXPECT_REASON", "").split("|") if r),
                   truthy("SKIP_STATIC", "false"), truthy("RUNTIME", "true"), truthy("NEEDS_BENCH", "false"))


def all_fixtures() -> list[Fixture]:
    return [_parse(p) for p in sorted(FIXTURES.glob("*/*.py"))]
