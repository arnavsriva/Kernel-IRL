import json

from kernelrl.bench.prefix_cache import _median_reps, cmd_compare, parse_metrics


def test_parse_metrics():
    text = ('# HELP sglang:cached_tokens_total x\n# TYPE sglang:cached_tokens_total counter\n'
            'sglang:cached_tokens_total{model_name="q",engine_type="unified"} 120.0\n'
            'sglang:prompt_tokens_total{model_name="q"} 200.0\n'
            'sglang:prompt_tokens_total{model_name="q2"} 100.0\n'
            'sglang:cache_hit_rate{model_name="q"} 0.6\n')
    m = parse_metrics(text)
    assert m["sglang:cached_tokens_total"] == 120 and m["sglang:prompt_tokens_total"] == 300


def _phase(ttft, tok):
    return {"n_requests": 16, "wall_s": 2.0, "req_per_s": 8.0, "output_tok_per_s": tok, "ttft_mean_s": ttft,
            "ttft_p50_s": ttft, "ttft_p95_s": ttft * 2, "latency_p50_s": 1.0, "prompt_tokens": 1000,
            "cached_tokens_reported": 0, "cache_hit_rate_reported": 0.0}


def test_compare(tmp_path):
    def doc(label, ttft, tok, hit):
        reps = [{"turn1": _phase(ttft, tok), "turn2": _phase(ttft, tok), "metrics_delta": {},
                 "cache_hit_rate_metrics": hit}] * 3
        return {"label": label, "server": {"version": "0.5.15.post1", "model_path": "Qwen/Qwen3-4B"}, "k": 8,
                "max_tokens": 256, "concurrency": 64, "reps": 3, "tasks": "all", "reps_data": reps,
                "median": _median_reps(reps)}
    on, off = tmp_path / "on.json", tmp_path / "off.json"
    on.write_text(json.dumps(doc("radix_on", 0.1, 3000, 0.8)))
    off.write_text(json.dumps(doc("radix_off", 0.4, 2000, 0.0)))

    class A:
        pass
    a = A()
    a.on, a.off, a.out = str(on), str(off), str(tmp_path / "pc.md")
    assert cmd_compare(a) == 0
    md = (tmp_path / "pc.md").read_text()
    assert "4.00x" in md and "Interpretation" in md
