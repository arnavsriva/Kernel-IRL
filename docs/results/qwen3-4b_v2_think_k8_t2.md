# Baseline: qwen3-4b_v2_think_k8_t2

Model `Qwen/Qwen3-4B` (thinking=True), k=8, turns=2, sampling {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'max_tokens': 24576}. Server: {'version': '0.5.15.post1', 'model_path': 'Qwen/Qwen3-4B', 'revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'disable_radix_cache': False, 'mem_fraction_static': 0.5, 'context_length': 32768, 'random_seed': 0, 'reasoning_parser': 'qwen3', 'enable_metrics': True, 'tp_size': 1, 'dtype': 'auto', 'max_running_requests': None}.

## Pass rates (macro-average over tasks)

| Turn | pass@1 | pass@k | L1 pass@1 | L2 pass@1 | L3 pass@1 | speedup vs eager (geomean, correct) | vs torch.compile | fast_1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 22.1% | 56.7% | 45.8% | 9.4% | 0.0% | 0.58x | 0.45x | 7.5% |
| 2 | 24.2% | 56.7% | 46.9% | 13.5% | 0.0% | 0.539x | 0.383x | 9.2% |

## Status distribution

| Turn | correct | incorrect | hack_detected | static_reject | compile_error | crash | timeout |
|---|---|---|---|---|---|---|---|
| 1 | 53 | 28 | 1 | 8 | 121 | 29 | 0 |
| 2 | 58 | 36 | 1 | 6 | 121 | 18 | 0 |

## Hack attempts and rule violations (turn 1)

- Runtime hacks detected: 1 {'no_triton_launch': 1}
- Static rule violations: 2
- All static rejection reasons: {'missing_entrypoint': 3, 'source_too_large': 2, 'forbidden_tensor_method': 1, 'no_code_found': 1, 'forbidden_api': 1}
- Extraction methods: {'fenced': 233, 'fenced_incomplete': 6, 'none': 1}; finish reasons: {'stop': 235, 'length': 5}
- Infrastructure errors (excluded from reward): 1

## 1-turn vs 2-turn

- Correct after turn 1: 22.1%; after one revision: 24.2%; correct in either turn: 32.1%
- Speedup change for samples correct in both turns (geomean t2/t1): 0.929x over 34 samples

| Transition (turn 1 -> turn 2) | count |
|---|---|
| compile_error->compile_error | 88 |
| correct->correct | 34 |
| crash->compile_error | 14 |
| compile_error->incorrect | 13 |
| correct->incorrect | 12 |
| compile_error->correct | 12 |
| incorrect->correct | 10 |
| crash->crash | 9 |
| incorrect->compile_error | 9 |
| incorrect->incorrect | 8 |
| correct->compile_error | 7 |
| compile_error->crash | 6 |
| static_reject->static_reject | 4 |
| crash->incorrect | 3 |
| static_reject->compile_error | 2 |
| crash->correct | 2 |
| static_reject->crash | 2 |
| crash->static_reject | 1 |
| compile_error->hack_detected | 1 |
| compile_error->static_reject | 1 |
| incorrect->crash | 1 |
| hack_detected->compile_error | 1 |

## Per task (turn 1)

| Task | L | pass@1 | pass@8 | statuses | speedup vs eager | best | vs compile |
|---|---|---|---|---|---|---|---|
| l1_clamp | 1 | 75.0% | 100.0% | correct:6, crash:2 | 0.091 | 1 | 0.0914 |
| l1_col_sum | 1 | 12.5% | 100.0% | compile_error:3, correct:1, crash:2, incorrect:1, static_reject:1 | 0.566 | 0.566 | 0.449 |
| l1_exp | 1 | 50.0% | 100.0% | compile_error:2, correct:4, crash:1, incorrect:1 | 0.973 | 0.977 | 0.981 |
| l1_fused_add_relu | 1 | 37.5% | 100.0% | compile_error:4, correct:3, incorrect:1 | 1.39 | 1.43 | 0.947 |
| l1_gelu | 1 | 25.0% | 100.0% | compile_error:5, correct:2, incorrect:1 | 0.962 | 0.968 | 0.972 |
| l1_mul | 1 | 75.0% | 100.0% | compile_error:1, correct:6, crash:1 | 0.976 | 0.98 | 0.982 |
| l1_relu | 1 | 62.5% | 100.0% | compile_error:2, correct:5, crash:1 | 0.989 | 0.994 | 1 |
| l1_row_max | 1 | 12.5% | 100.0% | compile_error:1, correct:1, incorrect:6 | 1.21 | 1.21 | 0.927 |
| l1_row_sum | 1 | 62.5% | 100.0% | compile_error:2, correct:5, incorrect:1 | 1.21 | 1.21 | 0.921 |
| l1_scalar_affine | 1 | 37.5% | 100.0% | compile_error:4, correct:3, incorrect:1 | 0.0668 | 1.61 | 0.0403 |
| l1_silu | 1 | 12.5% | 100.0% | compile_error:7, correct:1 | 0.963 | 0.963 | 0.952 |
| l1_vector_add | 1 | 87.5% | 100.0% | correct:7, incorrect:1 | 0.141 | 0.972 | 0.143 |
| l2_bias_gelu | 2 | 12.5% | 100.0% | compile_error:4, correct:1, crash:1, incorrect:2 | 4.76 | 4.76 | 1.03 |
| l2_cross_entropy | 2 | 0.0% | 0.0% | compile_error:7, incorrect:1 | – | – | – |
| l2_cumsum | 2 | 12.5% | 100.0% | compile_error:6, correct:1, crash:1 | 0.347 | 0.347 | 0.102 |
| l2_layernorm | 2 | 0.0% | 0.0% | compile_error:7, crash:1 | – | – | – |
| l2_log_softmax | 2 | 0.0% | 0.0% | compile_error:6, incorrect:2 | – | – | – |
| l2_logsumexp | 2 | 0.0% | 0.0% | compile_error:7, incorrect:1 | – | – | – |
| l2_mean_center | 2 | 25.0% | 100.0% | compile_error:4, correct:2, incorrect:2 | 1.63 | 1.63 | 0.874 |
| l2_residual_rmsnorm | 2 | 0.0% | 0.0% | compile_error:8 | – | – | – |
| l2_rmsnorm | 2 | 0.0% | 0.0% | compile_error:7, incorrect:1 | – | – | – |
| l2_row_l2norm | 2 | 37.5% | 100.0% | compile_error:3, correct:3, incorrect:2 | 1.74 | 1.75 | 0.867 |
| l2_softmax | 2 | 0.0% | 0.0% | compile_error:7, incorrect:1 | – | – | – |
| l2_swiglu | 2 | 25.0% | 100.0% | compile_error:4, correct:2, crash:1, incorrect:1 | 4.58 | 4.6 | 0.982 |
| l3_attention | 3 | 0.0% | 0.0% | compile_error:3, crash:4, incorrect:1 | – | – | – |
| l3_attention_causal | 3 | 0.0% | 0.0% | compile_error:1, crash:5, hack_detected:1, static_reject:1 | – | – | – |
| l3_batched_matmul | 3 | 0.0% | 0.0% | compile_error:4, crash:2, static_reject:2 | – | – | – |
| l3_matmul | 3 | 0.0% | 0.0% | compile_error:4, crash:4 | – | – | – |
| l3_matmul_bias_relu | 3 | 0.0% | 0.0% | compile_error:4, crash:2, incorrect:1, static_reject:1 | – | – | – |
| l3_matmul_scale_residual | 3 | 0.0% | 0.0% | compile_error:4, crash:1, static_reject:3 | – | – | – |
