# Baseline: smoke_v2_nothink

Model `Qwen/Qwen3-4B` (thinking=False), k=2, turns=2, sampling {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'max_tokens': 8192}. Server: {'version': '0.5.15.post1', 'model_path': 'Qwen/Qwen3-4B', 'revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'disable_radix_cache': False, 'mem_fraction_static': 0.5, 'context_length': 32768, 'random_seed': 0, 'reasoning_parser': 'qwen3', 'enable_metrics': True, 'tp_size': 1, 'dtype': 'auto', 'max_running_requests': None}.

## Pass rates (macro-average over tasks)

| Turn | pass@1 | pass@k | L1 pass@1 | L2 pass@1 | L3 pass@1 | speedup vs eager (geomean, correct) | vs torch.compile | fast_1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0% | 0.0% | 0.0% | 0.0% | – | –x | –x | 0.0% |
| 2 | 0.0% | 0.0% | 0.0% | 0.0% | – | –x | –x | 0.0% |

## Status distribution

| Turn | correct | incorrect | hack_detected | static_reject | compile_error | crash | timeout |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 3 | 0 | 0 | 0 | 1 | 0 |
| 2 | 0 | 2 | 0 | 0 | 1 | 1 | 0 |

## Hack attempts and rule violations (turn 1)

- Runtime hacks detected: 0 
- Static rule violations: 0
- All static rejection reasons: {}
- Extraction methods: {'fenced': 4}; finish reasons: {'stop': 4}
- Infrastructure errors (excluded from reward): 0

## 1-turn vs 2-turn

- Correct after turn 1: 0.0%; after one revision: 0.0%; correct in either turn: 0.0%
- Speedup change for samples correct in both turns (geomean t2/t1): –x over 0 samples

| Transition (turn 1 -> turn 2) | count |
|---|---|
| incorrect->incorrect | 2 |
| crash->crash | 1 |
| incorrect->compile_error | 1 |

## Per task (turn 1)

| Task | L | pass@1 | pass@2 | statuses | speedup vs eager | best | vs compile |
|---|---|---|---|---|---|---|---|
| l1_vector_add | 1 | 0.0% | 0.0% | crash:1, incorrect:1 | – | – | – |
| l2_softmax | 2 | 0.0% | 0.0% | incorrect:2 | – | – | – |
