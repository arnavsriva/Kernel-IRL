"""Evaluate one adversarial fixture and dump every finding and shape result (for debugging verdicts).
  FIXTURES="buggy/matmul_drops_k_tail" python scripts/diagnose_fixture.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelrl.sandbox.config import load_config  # noqa: E402
from kernelrl.sandbox.evaluate import evaluate  # noqa: E402
from tests.fixture_meta import all_fixtures  # noqa: E402

FAST = {"benchmark": {"warmup": 5, "iters": 20, "wall_iters": 20},
        "timeouts_s": {"compile": 60, "check": 120, "bench": 180}, "correctness": {"fixed_seeds": [11, 12]}}


def main() -> int:
    by_id = {f.id: f for f in all_fixtures()}
    out = {}
    for fid in os.environ["FIXTURES"].split(","):
        fx = by_id[fid.strip()]
        r = evaluate(fx.task, fx.source, config=load_config(overrides=FAST), run_bench=fx.needs_bench,
                     skip_static=fx.skip_static)
        d = r.to_dict()
        print(f"== {fid}: {r.status.value} / {r.reason}\n{r.message}")
        for f in d["findings"]:
            print("  finding:", json.dumps(f))
        for s in d["shape_results"]:
            print(f"  shape {s['shape']} seed={s['seed']} hidden={s['hidden']} passed={s['passed']} "
                  f"max_abs={s['max_abs_err']} err={(s['error'] or '')[:160]}")
        out[fid] = d
    Path("docs/results/diagnose_fixture.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
