# Baseline: qwen3-4b_v1_nothink_k8_t2

Model `Qwen/Qwen3-4B` (thinking=False), k=8, turns=2, sampling {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'max_tokens': 8192}. Server: {'version': '0.5.15.post1', 'model_path': 'Qwen/Qwen3-4B', 'revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'disable_radix_cache': False, 'mem_fraction_static': 0.5, 'context_length': 32768, 'random_seed': 0, 'reasoning_parser': 'qwen3', 'enable_metrics': True, 'tp_size': 1, 'dtype': 'auto', 'max_running_requests': None}.

## Pass rates (macro-average over tasks)

| Turn | pass@1 | pass@k | L1 pass@1 | L2 pass@1 | L3 pass@1 | speedup vs eager (geomean, correct) | vs torch.compile | fast_1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | –x | –x | 0.0% |
| 2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | –x | –x | 0.0% |

## Status distribution

| Turn | correct | incorrect | hack_detected | static_reject | compile_error | crash | timeout |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 127 | 102 | 11 | 0 |
| 2 | 0 | 0 | 0 | 57 | 172 | 11 | 0 |

## Hack attempts and rule violations (turn 1)

- Runtime hacks detected: 0 
- Static rule violations: 121
- All static rejection reasons: {'forbidden_api': 119, 'missing_entrypoint': 6, 'forbidden_tensor_method': 2}
- Extraction methods: {'fenced': 214, 'fenced_incomplete': 26}; finish reasons: {'stop': 223, 'length': 17}
- Infrastructure errors (excluded from reward): 0

## 1-turn vs 2-turn

- Correct after turn 1: 0.0%; after one revision: 0.0%; correct in either turn: 0.0%
- Speedup change for samples correct in both turns (geomean t2/t1): –x over 0 samples

| Transition (turn 1 -> turn 2) | count |
|---|---|
| compile_error->compile_error | 93 |
| static_reject->compile_error | 75 |
| static_reject->static_reject | 48 |
| compile_error->static_reject | 9 |
| crash->crash | 7 |
| crash->compile_error | 4 |
| static_reject->crash | 4 |

## Per task (turn 1)

| Task | L | pass@1 | pass@8 | statuses | speedup vs eager | best | vs compile |
|---|---|---|---|---|---|---|---|
| l1_clamp | 1 | 0.0% | 0.0% | compile_error:7, static_reject:1 | – | – | – |
| l1_col_sum | 1 | 0.0% | 0.0% | compile_error:2, static_reject:6 | – | – | – |
| l1_exp | 1 | 0.0% | 0.0% | compile_error:7, static_reject:1 | – | – | – |
| l1_fused_add_relu | 1 | 0.0% | 0.0% | compile_error:3, crash:1, static_reject:4 | – | – | – |
| l1_gelu | 1 | 0.0% | 0.0% | compile_error:1, static_reject:7 | – | – | – |
| l1_mul | 1 | 0.0% | 0.0% | compile_error:5, static_reject:3 | – | – | – |
| l1_relu | 1 | 0.0% | 0.0% | compile_error:8 | – | – | – |
| l1_row_max | 1 | 0.0% | 0.0% | compile_error:6, static_reject:2 | – | – | – |
| l1_row_sum | 1 | 0.0% | 0.0% | compile_error:7, static_reject:1 | – | – | – |
| l1_scalar_affine | 1 | 0.0% | 0.0% | compile_error:3, static_reject:5 | – | – | – |
| l1_silu | 1 | 0.0% | 0.0% | compile_error:6, crash:1, static_reject:1 | – | – | – |
| l1_vector_add | 1 | 0.0% | 0.0% | compile_error:4, static_reject:4 | – | – | – |
| l2_bias_gelu | 2 | 0.0% | 0.0% | static_reject:8 | – | – | – |
| l2_cross_entropy | 2 | 0.0% | 0.0% | compile_error:4, static_reject:4 | – | – | – |
| l2_cumsum | 2 | 0.0% | 0.0% | compile_error:3, static_reject:5 | – | – | – |
| l2_layernorm | 2 | 0.0% | 0.0% | static_reject:8 | – | – | – |
| l2_log_softmax | 2 | 0.0% | 0.0% | compile_error:3, static_reject:5 | – | – | – |
| l2_logsumexp | 2 | 0.0% | 0.0% | compile_error:2, static_reject:6 | – | – | – |
| l2_mean_center | 2 | 0.0% | 0.0% | compile_error:1, static_reject:7 | – | – | – |
| l2_residual_rmsnorm | 2 | 0.0% | 0.0% | compile_error:1, crash:1, static_reject:6 | – | – | – |
| l2_rmsnorm | 2 | 0.0% | 0.0% | compile_error:3, static_reject:5 | – | – | – |
| l2_row_l2norm | 2 | 0.0% | 0.0% | compile_error:5, static_reject:3 | – | – | – |
| l2_softmax | 2 | 0.0% | 0.0% | compile_error:6, static_reject:2 | – | – | – |
| l2_swiglu | 2 | 0.0% | 0.0% | compile_error:2, static_reject:6 | – | – | – |
| l3_attention | 3 | 0.0% | 0.0% | compile_error:4, static_reject:4 | – | – | – |
| l3_attention_causal | 3 | 0.0% | 0.0% | compile_error:4, static_reject:4 | – | – | – |
| l3_batched_matmul | 3 | 0.0% | 0.0% | compile_error:2, crash:1, static_reject:5 | – | – | – |
| l3_matmul | 3 | 0.0% | 0.0% | compile_error:1, crash:3, static_reject:4 | – | – | – |
| l3_matmul_bias_relu | 3 | 0.0% | 0.0% | compile_error:1, crash:3, static_reject:4 | – | – | – |
| l3_matmul_scale_residual | 3 | 0.0% | 0.0% | compile_error:1, crash:1, static_reject:6 | – | – | – |
