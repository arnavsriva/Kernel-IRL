"""Aggregate sandbox results into the baseline report (JSON summary + markdown)."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path

from kernelrl.eval.passk import pass_at_k

STATUSES = ["correct", "incorrect", "hack_detected", "static_reject", "compile_error", "crash", "timeout"]
# Static reasons that mean "tried to get PyTorch / the environment to do the work" rather than
# a plain formatting or import mistake. Reported separately from runtime hack detections.
RULE_VIOLATIONS = {"forbidden_api", "forbidden_tensor_method", "forbidden_operator", "module_attribute_assignment",
                   "module_escape", "forbidden_builtin", "dunder_access", "global_statement",
                   "forbidden_attribute_assignment"}


def _geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def turn_stats(results: dict[str, dict], k: int, tasks) -> dict:
    per_task, statuses, static_r, hack_r, extract_m, finish = {}, Counter(), Counter(), Counter(), Counter(), Counter()
    sp_eager, sp_compile, infra = [], [], 0
    for t in tasks:
        rs = [r for r in results.values() if r["task"] == t.id]
        st = Counter(r["result"]["status"] for r in rs)
        n, c = len(rs), st["correct"]
        se = [r["result"]["speedup_vs_eager"] for r in rs if r["result"]["status"] == "correct"]
        sc = [r["result"]["speedup_vs_compile"] for r in rs if r["result"]["status"] == "correct"]
        per_task[t.id] = {
            "level": t.level, "n": n, "correct": c,
            "pass@1": pass_at_k(n, c, 1) if n else None, f"pass@{k}": pass_at_k(n, c, k) if n >= k else None,
            "statuses": dict(st), "speedup_vs_eager_geomean": _geomean(se), "best_speedup_vs_eager": max(
                [x for x in se if x], default=None), "speedup_vs_compile_geomean": _geomean(sc),
            "mean_reward": statistics.fmean([r["reward"] for r in rs if r["reward"] is not None]) if rs else None,
        }
        statuses.update(st)
        sp_eager += [x for x in se if x]
        sp_compile += [x for x in sc if x]
        for r in rs:
            res = r["result"]
            if res["status"] == "static_reject":
                static_r[res["reason"]] += 1
            if res["status"] == "hack_detected":
                hack_r[res["reason"]] += 1
            infra += bool(res.get("meta", {}).get("infra_error"))
            extract_m[r["extraction"]["method"]] += 1
            finish[r.get("finish_reason") or "none"] += 1

    def macro(key, level=None):
        v = [p[key] for p in per_task.values() if p[key] is not None and (level is None or p["level"] == level)]
        return statistics.fmean(v) if v else None

    total = sum(statuses.values())
    return {
        "n_samples": total, "pass@1": macro("pass@1"), f"pass@{k}": macro(f"pass@{k}"),
        "by_level": {lv: {"pass@1": macro("pass@1", lv), f"pass@{k}": macro(f"pass@{k}", lv)} for lv in (1, 2, 3)},
        "status_counts": {s: statuses.get(s, 0) for s in STATUSES},
        "static_reject_reasons": dict(static_r.most_common()), "hack_reasons": dict(hack_r.most_common()),
        "hack_detected": statuses.get("hack_detected", 0),
        "static_rule_violations": sum(v for r, v in static_r.items() if r in RULE_VIOLATIONS),
        "speedup_vs_eager_geomean_correct": _geomean(sp_eager),
        "speedup_vs_eager_median_correct": statistics.median(sp_eager) if sp_eager else None,
        "speedup_vs_compile_geomean_correct": _geomean(sp_compile),
        "fast_1_frac": (sum(x > 1 for x in sp_eager) / total) if total else None,
        "extraction_methods": dict(extract_m), "finish_reasons": dict(finish), "infra_errors": infra,
        "per_task": per_task,
    }


def two_turn_stats(r1: dict, r2: dict) -> dict:
    trans, both_sp, n, c1, c2, either = Counter(), [], 0, 0, 0, 0
    for sid, a in r1.items():
        b = r2.get(sid.replace(":t1", ":t2"))
        if b is None:
            continue
        n += 1
        s1, s2 = a["result"]["status"], b["result"]["status"]
        trans[f"{s1}->{s2}"] += 1
        c1 += s1 == "correct"
        c2 += s2 == "correct"
        either += s1 == "correct" or s2 == "correct"
        if s1 == s2 == "correct" and a["result"]["speedup_vs_eager"] and b["result"]["speedup_vs_eager"]:
            both_sp.append(b["result"]["speedup_vs_eager"] / a["result"]["speedup_vs_eager"])
    return {"n_pairs": n, "turn1_correct_rate": c1 / n if n else None, "turn2_correct_rate": c2 / n if n else None,
            "correct_in_either_rate": either / n if n else None, "transitions": dict(trans.most_common()),
            "speedup_ratio_t2_over_t1_geomean": _geomean(both_sp), "n_correct_both": len(both_sp)}


def summarize(r1: dict, r2: dict | None, k: int, tasks) -> dict:
    out = {"k": k, "turn1": turn_stats(r1, k, tasks)}
    if r2:
        out["turn2"] = turn_stats(r2, k, tasks)
        out["two_turn"] = two_turn_stats(r1, r2)
    return out


def _f(x, pct=False, digits=3):
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.{digits}g}"


def write_markdown(summary: dict, meta: dict, path: Path) -> Path:
    k = summary["k"]
    t1 = summary["turn1"]
    L = [f"# Baseline: {meta['run_id']}", "",
         f"Model `{meta['model']}` (thinking={meta['thinking']}), k={k}, turns={meta['turns']}, "
         f"sampling {meta['sampling']}. Server: {meta.get('server', {})}.", "",
         "## Pass rates (macro-average over tasks)", "",
         "| Turn | pass@1 | pass@k | L1 pass@1 | L2 pass@1 | L3 pass@1 | speedup vs eager (geomean, correct) "
         "| vs torch.compile | fast_1 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name in ("turn1", "turn2"):
        if name not in summary:
            continue
        t = summary[name]
        L.append(f"| {name[-1]} | {_f(t['pass@1'], True)} | {_f(t[f'pass@{k}'], True)} | "
                 + " | ".join(_f(t["by_level"][lv]["pass@1"], True) for lv in (1, 2, 3))
                 + f" | {_f(t['speedup_vs_eager_geomean_correct'])}x | {_f(t['speedup_vs_compile_geomean_correct'])}x"
                 f" | {_f(t['fast_1_frac'], True)} |")
    L += ["", "## Status distribution", "", "| Turn | " + " | ".join(STATUSES) + " |",
          "|---|" + "---|" * len(STATUSES)]
    for name in ("turn1", "turn2"):
        if name in summary:
            L.append(f"| {name[-1]} | " + " | ".join(str(summary[name]["status_counts"][s]) for s in STATUSES) + " |")
    L += ["", "## Hack attempts and rule violations (turn 1)", "",
          f"- Runtime hacks detected: {t1['hack_detected']} {t1['hack_reasons'] or ''}",
          f"- Static rule violations: {t1['static_rule_violations']}",
          f"- All static rejection reasons: {t1['static_reject_reasons'] or '{}'}",
          f"- Extraction methods: {t1['extraction_methods']}; finish reasons: {t1['finish_reasons']}",
          f"- Infrastructure errors (excluded from reward): {t1['infra_errors']}"]
    if "two_turn" in summary:
        tt = summary["two_turn"]
        L += ["", "## 1-turn vs 2-turn", "",
              f"- Correct after turn 1: {_f(tt['turn1_correct_rate'], True)}; after one revision: "
              f"{_f(tt['turn2_correct_rate'], True)}; correct in either turn: {_f(tt['correct_in_either_rate'], True)}",
              f"- Speedup change for samples correct in both turns (geomean t2/t1): "
              f"{_f(tt['speedup_ratio_t2_over_t1_geomean'])}x over {tt['n_correct_both']} samples",
              "", "| Transition (turn 1 -> turn 2) | count |", "|---|---|"]
        L += [f"| {a} | {b} |" for a, b in tt["transitions"].items()]
    L += ["", "## Per task (turn 1)", "",
          f"| Task | L | pass@1 | pass@{k} | statuses | speedup vs eager | best | vs compile |",
          "|---|---|---|---|---|---|---|---|"]
    for tid, p in t1["per_task"].items():
        st = ", ".join(f"{a}:{b}" for a, b in sorted(p["statuses"].items()))
        L.append(f"| {tid} | {p['level']} | {_f(p['pass@1'], True)} | {_f(p[f'pass@{k}'], True)} | {st} | "
                 f"{_f(p['speedup_vs_eager_geomean'])} | {_f(p['best_speedup_vs_eager'])} | "
                 f"{_f(p['speedup_vs_compile_geomean'])} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return path
