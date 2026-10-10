# Prefix caching: radix cache on vs off

Workload: all 30 tasks, k=8 samples per prompt, 256 output tokens (ignore_eos), concurrency 64, median of 3 repetitions, cache flushed before each repetition. Server: sglang 0.5.15.post1, model Qwen/Qwen3-4B.

| Phase | Metric | Radix on | Radix off | On/Off |
|---|---|---|---|---|
| turn1 | ttft_p50_s | 0.290 | 0.642 | 0.45x |
| turn1 | ttft_p95_s | 0.578 | 1.175 | 0.49x |
| turn1 | output_tok_per_s | 6626 | 4283 | 1.55x |
| turn1 | req_per_s | 25.88 | 16.73 | 1.55x |
| turn1 | cache_hit_rate_reported | 74.8% | 0.0% | – |
| turn2 | ttft_p50_s | 0.234 | 0.881 | 0.27x |
| turn2 | ttft_p95_s | 0.376 | 1.670 | 0.23x |
| turn2 | output_tok_per_s | 6239 | 3411 | 1.83x |
| turn2 | req_per_s | 24.37 | 13.32 | 1.83x |
| turn2 | cache_hit_rate_reported | 95.5% | 0.0% | – |
| all | cache hit rate (/metrics) | 87.0% | 0.0% | |

## Interpretation (reviewed)

Setup: one Modal H100 80GB, Qwen3-4B, prompt v2, non-thinking. Each repetition sends 240 turn-1 requests
(30 tasks x 8 samples, about 1,340 prompt tokens each), then 240 turn-2 requests whose prompt is the turn-1
conversation plus an answer and feedback (about 1,920 tokens). Output is fixed at 256 tokens, so decode work is
identical with the cache on and off, and every difference comes from prefill.

- **Turn 1, k-way shared prompts.** Time to first token drops from 0.64 s to 0.29 s (2.2x), and throughput
  rises 1.55x. The reported hit rate is 75% rather than the ideal 7/8 = 87.5%, because the 8 samples of a prompt
  arrive concurrently. Some of them start prefill before the first one has populated the cache.
- **Turn 2, conversation prefix.** The gain grows to 3.8x on time to first token and 1.8x on throughput, with a
  95.5% hit rate. Turn-2 prompts are longer, and almost all of each one (the turn-1 conversation) is already
  cached. With the cache off, turn 2 is slower than turn 1 (0.88 s vs 0.64 s), because there is more prefill
  per request.
- **Tail latency improves the same way.** p95 time to first token drops 2.0x on turn 1 and 4.4x on turn 2.
  Without the cache, long prefills queue behind each other.
- **Implications for Phase 3.**
  - Multi-turn rollouts have exactly this structure: k samples per task, and each later turn extends the
    previous conversation. Keep the radix cache on in the rollout engine.
  - The benefit grows with the number of turns and with the length of the shared history.
  - The remaining 12% of turn-1 misses from concurrent arrival are worth a look in the systems study. One
    option is to prefill the shared prompt once before fanning out k samples, for example with SGLang's `n`
    parameter.
