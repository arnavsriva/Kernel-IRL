# Phase 0 baseline: Qwen3-4B on 30 Triton tasks

Model Qwen/Qwen3-4B (HF rev 1cfa9a72) on SGLang 0.5.15.post1, one Modal H100 80GB per run, 2026-10-10.
Each run has 30 tasks, k=8 samples per task, and 2 turns: one revision with sandbox feedback.
Sampling follows the Qwen3 card plus presence_penalty 1.5. Thinking runs allow 24,576 completion tokens and
non-thinking runs allow 8,192. Per-run reports are in `<run_id>.md` and `<run_id>.json` next to this file.

## Pass rates (macro-average over tasks)

| Prompt | Thinking | Turn | pass@1 | pass@8 | L1 pass@1 | L2 pass@1 | L3 pass@1 | correct / 240 | speedup vs eager (geomean of correct) | vs torch.compile |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 zero-shot | off | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | –x | –x |
| v1 zero-shot | off | 2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | –x | –x |
| v2 cheat-sheet | off | 1 | 0.8% | 6.7% | 2.1% | 0.0% | 0.0% | 2 | 1.30x | 1.01x |
| v2 cheat-sheet | off | 2 | 0.8% | 6.7% | 2.1% | 0.0% | 0.0% | 2 | 1.29x | 1.00x |
| v2 cheat-sheet | on | 1 | 22.1% | 56.7% | 45.8% | 9.4% | 0.0% | 53 | 0.58x | 0.45x |
| v2 cheat-sheet | on | 2 | 24.2% | 56.7% | 46.9% | 13.5% | 0.0% | 58 | 0.54x | 0.38x |

## Status distribution

| Prompt | Thinking | Turn | correct | incorrect | hack_detected | static_reject | compile_error | crash | timeout |
|---|---|---|---|---|---|---|---|---|---|
| v1 zero-shot | off | 1 | 0 | 0 | 0 | 127 | 102 | 11 | 0 |
| v1 zero-shot | off | 2 | 0 | 0 | 0 | 57 | 172 | 11 | 0 |
| v2 cheat-sheet | off | 1 | 2 | 100 | 1 | 13 | 59 | 65 | 0 |
| v2 cheat-sheet | off | 2 | 2 | 98 | 1 | 8 | 79 | 52 | 0 |
| v2 cheat-sheet | on | 1 | 53 | 28 | 1 | 8 | 121 | 29 | 0 |
| v2 cheat-sheet | on | 2 | 58 | 36 | 1 | 6 | 121 | 18 | 0 |

## One revision with feedback

| Prompt | Thinking | correct after turn 1 | after turn 2 | correct in either | speedup t2/t1 (both correct) |
|---|---|---|---|---|---|
| v1 zero-shot | off | 0.0% | 0.0% | 0.0% | –x over 0 |
| v2 cheat-sheet | off | 0.8% | 0.8% | 0.8% | 0.99x over 2 |
| v2 cheat-sheet | on | 22.1% | 24.2% | 32.1% | 0.93x over 34 |

## Reading the numbers

- **Zero-shot Qwen3-4B cannot write Triton.** It invents APIs (`triton.launch`, `tl.num_blocks_per_grid`),
  writes CUDA-style one-thread-per-element code, and passes `data_ptr()` integers. It solved 0 of 240 in both turns.
- **The cheat-sheet plus worked example fixes the API problem but not the logic.** Without thinking, most
  samples now compile and run (100 of 240 are `incorrect` rather than rejected). Only 2 are correct, both
  elementwise.
- **Thinking mode is the usable starting point for RL.** pass@1 is 22-24% and pass@8 is 57%. Level 1 is about
  46%, Level 2 is 9-14%, and Level 3 (matmul, attention) is 0%. Correct kernels average 0.54-0.58x eager:
  mostly unfused or poorly blocked, with a long tail of very slow ones. That leaves headroom for the speedup
  reward.
- **One feedback round helps a little with thinking (22.1% to 24.2% correct, 32% in either turn) and not at
  all without it.** Revisions of correct kernels are about 7% slower on average.
- **No deliberate reward hacks appeared.** Every hack flag was reviewed by hand (see the verifier fixes below).
  The only rule violation left is a host-side `out.to(torch.bfloat16)` after a correct fp32 Triton kernel, which
  the current policy rejects. That policy is still open.
- **Implication for Phase 2.** GRPO on the v2-thinking setup gets signal on L1/L2 but none on L3. Consider
  curriculum or task weighting, and a larger completion budget for L3.

## Verifier fixes made while reviewing these runs

All of these change labels only. Correctness checks and rewards for wrong outputs are unchanged, so the pass
rates above are unaffected. The fixes landed after the runs, so the raw per-run reports still show the old
labels:
- 2 v2 non-thinking samples labelled `state_dependent` are an honest out-of-bounds read (row max without a
  column mask). They are now `incorrect` with a nondeterminism message.
- 3 samples recorded as infrastructure errors (reward None) were empty launch grids
  (`next_power_of_2(n) // 1024`). They are now `incorrect`.
- 1 thinking sample labelled `no_triton_launch` is an unfinished placeholder. It is now `incorrect`.
- 1 thinking sample labelled `low_triton_time_fraction` (host dtype cast after a correct kernel) stays a
  violation under the current policy.
- Earlier: the client crashed at turn 2 (event loop reuse, then context overflow). Resumed runs reused all
  turn-1 work.
- The `l1_row_max` inputs are now all-negative so zero-padded max reductions always fail. Its numbers here
  predate that change.
