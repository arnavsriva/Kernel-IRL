"""Prefix-caching benchmark: radix cache on vs off under the multi-turn eval traffic pattern.

Traffic (per repetition, cache flushed first):
  phase "turn1": for each task, k concurrent requests with an identical long prompt
                 (system + task + reference code), i.e. k-way shared prefix;
  phase "turn2": for each task, k requests whose prompt = turn-1 conversation + a fixed
                 assistant answer + feedback, i.e. the turn-1 prompt is a shared prefix again.
Output length is fixed (ignore_eos) so both servers do identical decode work.

  GPU=0 RADIX=on  bash kernelrl/serving/launch_sglang.sh            # terminal 1
  python -m kernelrl.bench.prefix_cache run --label radix_on
  (restart the server with RADIX=off)
  python -m kernelrl.bench.prefix_cache run --label radix_off
  python -m kernelrl.bench.prefix_cache compare docs/results/prefix_cache_radix_on.json \\
      docs/results/prefix_cache_radix_off.json
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from openai import AsyncOpenAI

from kernelrl.common.env_fingerprint import REPO_ROOT, stamp
from kernelrl.common.io import atomic_write_json, read_json
from kernelrl.serving.client import flush_cache, server_info
from kernelrl.serving.prompt import initial_messages, revision_messages
from kernelrl.tasks.registry import all_tasks, load_task

FIXED_ANSWER = (Path(REPO_ROOT) / "tests" / "fixtures" / "good" / "softmax.py")
FEEDBACK = ("## Evaluation result: INCORRECT\nWrong output on shape M=4096, N=4096: value_mismatch: 3.10% of "
            "elements outside tolerance (atol=1e-06, rtol=1e-05)\nPublic shapes:\n  - M=1024, N=1024: pass\n"
            "  - M=4096, N=4096: FAIL\nHidden shapes: 9/14 checks passed.")
METRICS = ("sglang:cached_tokens_total", "sglang:prompt_tokens_total", "sglang:generation_tokens_total",
           "sglang:num_requests_total")


def scrape_metrics(base_url: str) -> dict[str, float]:
    root = base_url.rstrip("/").removesuffix("/v1")
    with urllib.request.urlopen(f"{root}/metrics", timeout=10) as r:
        return parse_metrics(r.read().decode())


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each counter of interest across label sets in Prometheus text format."""
    out = {m: 0.0 for m in METRICS}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for m in METRICS:
            if line.startswith(m + "{") or line.startswith(m + " "):
                out[m] += float(line.rsplit(" ", 1)[1])
    return out


async def one_request(client, model, messages, max_tokens, seed, sem) -> dict:
    async with sem:
        t0 = time.perf_counter()
        ttft, usage, n_chunks = None, None, 0
        stream = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.7, top_p=0.8, seed=seed,
            stream=True, stream_options={"include_usage": True},
            extra_body={"ignore_eos": True, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
        async for chunk in stream:
            if chunk.choices and ttft is None:
                d = chunk.choices[0].delta
                if (d.content or getattr(d, "reasoning_content", None)):
                    ttft = time.perf_counter() - t0
            if chunk.choices:
                n_chunks += 1
            if chunk.usage is not None:
                usage = chunk.usage
        lat = time.perf_counter() - t0
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        return {"ttft_s": ttft, "latency_s": lat, "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "cached_tokens": getattr(details, "cached_tokens", None) if details else None}


async def run_phase(client, model, convs: list[list[dict]], k: int, max_tokens: int, concurrency: int, seed0: int):
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    reqs = [one_request(client, model, msgs, max_tokens, seed0 + j * 1000 + i, sem)
            for j, msgs in enumerate(convs) for i in range(k)]
    res = await asyncio.gather(*reqs)
    wall = time.perf_counter() - t0
    ttfts = [r["ttft_s"] for r in res if r["ttft_s"] is not None]
    out_tok = sum(r["completion_tokens"] or 0 for r in res)
    prompt_tok = sum(r["prompt_tokens"] or 0 for r in res)
    cached = sum(r["cached_tokens"] or 0 for r in res)
    q = statistics.quantiles(ttfts, n=20) if len(ttfts) >= 2 else [None] * 19
    return {"n_requests": len(res), "wall_s": wall, "req_per_s": len(res) / wall, "output_tok_per_s": out_tok / wall,
            "ttft_mean_s": statistics.fmean(ttfts) if ttfts else None,
            "ttft_p50_s": statistics.median(ttfts) if ttfts else None, "ttft_p95_s": q[18],
            "latency_p50_s": statistics.median([r["latency_s"] for r in res]),
            "prompt_tokens": prompt_tok, "cached_tokens_reported": cached,
            "cache_hit_rate_reported": cached / prompt_tok if prompt_tok else None}


async def bench(args) -> dict:
    client = AsyncOpenAI(base_url=args.server_url, api_key="EMPTY", timeout=3600, max_retries=0)
    tasks = [load_task(t) for t in args.tasks] if args.tasks else all_tasks()
    answer = "```python\n" + FIXED_ANSWER.read_text() + "```"
    turn1 = [initial_messages(t) for t in tasks]
    turn2 = [revision_messages(t, answer, FEEDBACK) for t in tasks]
    reps = []
    for rep in range(args.reps):
        if not flush_cache(args.server_url):
            print("warning: /flush_cache failed; repetition may start warm", file=sys.stderr)
        m0 = scrape_metrics(args.server_url)
        p1 = await run_phase(client, args.model, turn1, args.k, args.max_tokens, args.concurrency, 1000 * rep)
        p2 = await run_phase(client, args.model, turn2, args.k, args.max_tokens, args.concurrency, 1000 * rep + 500)
        m1 = scrape_metrics(args.server_url)
        d = {m: m1[m] - m0[m] for m in METRICS}
        reps.append({"turn1": p1, "turn2": p2, "metrics_delta": d,
                     "cache_hit_rate_metrics": (d["sglang:cached_tokens_total"] / d["sglang:prompt_tokens_total"]
                                                if d["sglang:prompt_tokens_total"] else None)})
        print(f"rep {rep}: turn1 ttft_p50={p1['ttft_p50_s']:.3f}s tok/s={p1['output_tok_per_s']:.0f} | "
              f"turn2 ttft_p50={p2['ttft_p50_s']:.3f}s tok/s={p2['output_tok_per_s']:.0f} | "
              f"hit={reps[-1]['cache_hit_rate_metrics']}", flush=True)
    return {"repetitions": reps, "median": _median_reps(reps)}


def _median_reps(reps: list[dict]) -> dict:
    out = {}
    for phase in ("turn1", "turn2"):
        out[phase] = {k: statistics.median([r[phase][k] for r in reps if r[phase][k] is not None] or [0])
                      for k in reps[0][phase]}
    hits = [r["cache_hit_rate_metrics"] for r in reps if r["cache_hit_rate_metrics"] is not None]
    out["cache_hit_rate_metrics"] = statistics.median(hits) if hits else None
    return out


def cmd_run(args) -> int:
    info = server_info(args.server_url)
    radix_off = info.get("disable_radix_cache")
    if radix_off is not None and args.label in ("radix_on", "radix_off") and radix_off != (args.label == "radix_off"):
        sys.exit(f"label {args.label!r} does not match server disable_radix_cache={radix_off}")
    res = asyncio.run(bench(args))
    out = Path(args.results_dir) / f"prefix_cache_{args.label}.json"
    atomic_write_json(out, stamp({"label": args.label, "server": info, "k": args.k, "max_tokens": args.max_tokens,
                                  "concurrency": args.concurrency, "reps": args.reps,
                                  "tasks": args.tasks or "all", **res}))
    print(f"wrote {out}")
    return 0


def _n_reps(doc: dict) -> int:
    # Early result files stored the repetition list under "reps", which overwrote the count.
    r = doc.get("reps")
    return r if isinstance(r, int) else len(r or doc.get("repetitions", []))


def _ratio(a, b):
    return a / b if a and b else None


def cmd_compare(args) -> int:
    on, off = read_json(args.on), read_json(args.off)
    mo, mf = on["median"], off["median"]
    lines = ["# Prefix caching: radix cache on vs off", "",
             f"Workload: {on['tasks'] if on['tasks'] != 'all' else 'all 30'} tasks, k={on['k']} samples per prompt, "
             f"{on['max_tokens']} output tokens (ignore_eos), concurrency {on['concurrency']}, "
             f"median of {_n_reps(on)} repetitions, cache flushed before each repetition. "
             f"Server: sglang {on['server'].get('version')}, model {on['server'].get('model_path')}.", "",
             "| Phase | Metric | Radix on | Radix off | On/Off |", "|---|---|---|---|---|"]
    for ph in ("turn1", "turn2"):
        for k, fmt in (("ttft_p50_s", "{:.3f}"), ("ttft_p95_s", "{:.3f}"), ("output_tok_per_s", "{:.0f}"),
                       ("req_per_s", "{:.2f}"), ("cache_hit_rate_reported", "{:.1%}")):
            a, b = mo[ph].get(k), mf[ph].get(k)
            r = _ratio(a, b)
            lines.append(f"| {ph} | {k} | {fmt.format(a) if a is not None else '–'} | "
                         f"{fmt.format(b) if b is not None else '–'} | {f'{r:.2f}x' if r else '–'} |")
    fmt_hit = lambda x: "–" if x is None else f"{x:.1%}"  # noqa: E731
    lines.append(f"| all | cache hit rate (/metrics) | {fmt_hit(mo['cache_hit_rate_metrics'])} | "
                 f"{fmt_hit(mf['cache_hit_rate_metrics'])} | |")
    t1 = _ratio(mf["turn1"]["ttft_p50_s"], mo["turn1"]["ttft_p50_s"])
    t2 = _ratio(mf["turn2"]["ttft_p50_s"], mo["turn2"]["ttft_p50_s"])
    thr = _ratio(mo["turn2"]["output_tok_per_s"], mf["turn2"]["output_tok_per_s"])
    lines += ["", "## Interpretation (auto-generated from the numbers; edit after review)", "",
              f"- With k={on['k']} samples per prompt, the radix cache lets k-1 of every k requests reuse the shared "
              f"prompt prefix. Median time to first token drops {t1:.2f}x on turn 1." if t1 else "- turn 1: n/a",
              f"- On turn 2 the whole turn-1 conversation is a reusable prefix, so the gain is {t2:.2f}x on time to "
              f"first token and {thr:.2f}x on output throughput." if t2 and thr else "- turn 2: n/a",
              "- Decode work is identical in both runs, so the differences come from skipped prefill. Multi-turn "
              "RL rollouts (Phase 3) have the same structure, which is why the rollout engine keeps the radix "
              "cache on."]
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--server-url", default="http://127.0.0.1:30000/v1")
    r.add_argument("--model", default="Qwen/Qwen3-4B")
    r.add_argument("--tasks", nargs="*")
    r.add_argument("--k", type=int, default=8)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--results-dir", default=str(REPO_ROOT / "docs" / "results"))
    c = sub.add_parser("compare")
    c.add_argument("on")
    c.add_argument("off")
    c.add_argument("--out", default=str(REPO_ROOT / "docs" / "results" / "prefix_cache.md"))
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
