"""Baseline eval: sample k kernels per task from the SGLang server, run every one through the
sandbox, optionally feed sandbox feedback back for one revision, and write JSON + markdown.

  python -m kernelrl.eval.run_eval --run-id qwen3-4b_nothink_k8
  python -m kernelrl.eval.run_eval --run-id smoke --tasks l1_vector_add l2_softmax --k 2 --turns 2

Phases run strictly in sequence (generate, then evaluate) so kernel timings never overlap with
LLM inference on the same GPU. Everything is appended to runs/<run_id>/*.jsonl, and a rerun with
the same run id resumes where it stopped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

from kernelrl.common.env_fingerprint import REPO_ROOT, stamp
from kernelrl.common.io import atomic_write_json
from kernelrl.eval.report import summarize, write_markdown
from kernelrl.reward.reward import compute_reward
from kernelrl.sandbox.config import load_config
from kernelrl.sandbox.evaluate import evaluate
from kernelrl.sandbox.feedback import render_feedback
from kernelrl.serving.client import Generator, server_info
from kernelrl.serving.extract import extract_code
from kernelrl.serving.prompt import initial_messages, revision_messages
from kernelrl.tasks.registry import all_tasks, load_task

CONFIG = Path(__file__).with_name("config.yaml")
TURN2_SEED_OFFSET = 100_000


def _load_jsonl(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _append(p: Path, rec: dict) -> None:
    with open(p, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


async def generate(requests: list[dict], make_gen, out: Path) -> None:
    # The client (and its semaphore / HTTP pool) must be created inside the running event loop:
    # each turn uses its own asyncio.run(), and asyncio primitives bind to the first loop that
    # contends on them (crashed the first full baseline at turn 2 once >concurrency requests queued).
    gen = make_gen()
    done = _load_jsonl(out)
    todo = [r for r in requests if r["id"] not in done]
    print(f"[generate] {len(todo)} requests ({len(done)} cached) -> {out.name}", flush=True)

    async def one(r):
        c = await gen.complete(r["messages"], r["seed"])
        _append(out, {**{k: v for k, v in r.items() if k != "messages"}, "completion": c.to_dict()})

    t0 = time.perf_counter()
    await asyncio.gather(*(one(r) for r in todo))
    print(f"[generate] done in {time.perf_counter() - t0:.1f}s", flush=True)


def evaluate_samples(samples: dict[str, dict], out: Path, sandbox_cfg: dict, gpu, run_bench: bool) -> None:
    done = _load_jsonl(out)
    todo = [s for sid, s in sorted(samples.items()) if sid not in done]
    print(f"[sandbox] {len(todo)} candidates ({len(done)} cached) -> {out.name}", flush=True)
    for i, s in enumerate(todo):
        comp = s["completion"]
        ext = extract_code(comp["content"])
        rec = {"id": s["id"], "task": s["task"], "sample": s["sample"], "turn": s["turn"],
               "extraction": {"method": ext.method, "error": ext.error, "had_thinking": ext.had_thinking},
               "finish_reason": comp.get("finish_reason"), "completion_tokens": comp.get("completion_tokens"),
               "gen_error": comp.get("error")}
        task = load_task(s["task"])
        if ext.code is None:
            from kernelrl.common.schemas import EvalResult, Status

            res = EvalResult(task.id, Status.STATIC_REJECT, reason="no_code_found",
                             message="No Python code block was found in your response.", stage="extract")
        else:
            res = evaluate(task.id, ext.code, config=sandbox_cfg, gpu=gpu, run_bench=run_bench)
        rw = compute_reward(res)
        rec.update(result=res.to_dict(), reward=rw.reward, feedback=render_feedback(res, task, sandbox_cfg),
                   code=ext.code)
        _append(out, rec)
        sp = f" {res.speedup_vs_eager:.2f}x" if res.speedup_vs_eager else ""
        print(f"[sandbox] {i + 1}/{len(todo)} {s['id']}: {res.status.value} {res.reason}{sp}", flush=True)


def main(argv=None) -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--k", type=int, default=cfg["k"])
    ap.add_argument("--turns", type=int, choices=(1, 2), default=cfg["turns"])
    ap.add_argument("--thinking", action="store_true", default=cfg["thinking"])
    ap.add_argument("--prompt", choices=("v1", "v2"), default=cfg["prompt"])
    ap.add_argument("--server-url", default=cfg["server_url"])
    ap.add_argument("--model", default=cfg["model"])
    ap.add_argument("--concurrency", type=int, default=cfg["concurrency"])
    ap.add_argument("--sandbox-gpu", default=cfg["sandbox_gpu"])
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "docs" / "results"))
    args = ap.parse_args(argv)

    run_dir = Path(args.out_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = [load_task(t) for t in args.tasks] if args.tasks else all_tasks()
    sampling = cfg["sampling"]["thinking" if args.thinking else "non_thinking"]
    sandbox_cfg = load_config()
    def make_gen():
        return Generator(args.server_url, args.model, sampling, thinking=args.thinking,
                         concurrency=args.concurrency)
    meta = {"run_id": args.run_id, "model": args.model, "k": args.k, "turns": args.turns, "thinking": args.thinking,
            "prompt": args.prompt,
            "sampling": sampling, "seed": cfg["seed"], "tasks": [t.id for t in tasks],
            "server": server_info(args.server_url), "sandbox_config": sandbox_cfg}
    (run_dir / "meta.json").write_text(json.dumps(stamp(meta), indent=2, default=str))

    # ---- turn 1 ----
    reqs = [{"id": f"{t.id}:{i}:t1", "task": t.id, "sample": i, "turn": 1, "seed": cfg["seed"] + i,
             "messages": initial_messages(t, args.prompt)} for t in tasks for i in range(args.k)]
    asyncio.run(generate(reqs, make_gen, run_dir / "turn1_samples.jsonl"))
    s1 = {k: v for k, v in _load_jsonl(run_dir / "turn1_samples.jsonl").items() if k in {r["id"] for r in reqs}}
    evaluate_samples(s1, run_dir / "turn1_results.jsonl", sandbox_cfg, args.sandbox_gpu, not args.no_bench)
    r1 = _load_jsonl(run_dir / "turn1_results.jsonl")

    # ---- turn 2: one revision per turn-1 sample, conditioned on its sandbox feedback ----
    r2 = None
    if args.turns == 2:
        reqs2 = []
        for sid, s in sorted(s1.items()):
            task = load_task(s["task"])
            msgs = revision_messages(task, s["completion"]["content"], r1[sid]["feedback"], args.prompt)
            reqs2.append({"id": sid.replace(":t1", ":t2"), "task": s["task"], "sample": s["sample"], "turn": 2,
                          "seed": s["seed"] + TURN2_SEED_OFFSET, "messages": msgs})
        asyncio.run(generate(reqs2, make_gen, run_dir / "turn2_samples.jsonl"))
        s2 = {k: v for k, v in _load_jsonl(run_dir / "turn2_samples.jsonl").items()
              if k in {r["id"] for r in reqs2}}
        evaluate_samples(s2, run_dir / "turn2_results.jsonl", sandbox_cfg, args.sandbox_gpu, not args.no_bench)
        r2 = _load_jsonl(run_dir / "turn2_results.jsonl")

    summary = summarize(r1, r2, args.k, tasks)
    out_json = Path(args.results_dir) / f"{args.run_id}.json"
    atomic_write_json(out_json, stamp({**meta, "summary": summary}))
    out_md = write_markdown(summary, meta, out_json.with_suffix(".md"))
    print(f"\nwrote {out_json}\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
