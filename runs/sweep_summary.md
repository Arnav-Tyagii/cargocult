# Phase 4 sweep

Dev-tier evals throughout (90 problems x 4 samples, temperature 0.8). The dev baseline is **0.2472**; the full-tier baseline is 0.2281 and is *not* what these are compared against.

## Dev pass@1

| run | dev pass@1 | vs baseline | best across checkpoints | steps |
|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | 0.2306 | -0.0166 | 0.2639 | 135 |

## Likelihood displacement

DPO constrains the gap, never the levels. A run whose `logp_chosen` fell while its margin grew is displacing likelihood — the loss curve will look fine.

| run | logp_chosen | logp_rejected | chosen drift | margin | reward_acc |
|---|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | -61.98 | -247.38 | +21.0 | 1.44 | 1.00 |

## Length and style

`*_terminated` excludes completions cut off at the token budget. The two columns differ because 11.6% of the rejected side of the corpus was truncated, and a truncated completion is long for a reason unrelated to verbosity.

| run | train len (all) | train len (terminated) | dev mean tokens | stub_args |
|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | 103.88 | 103.88 | 99.79 | 88.3% |

Baseline `stub_args_rate` is 80.8% on dev. The prediction registered in `notes/readme_draft.md` before any of these runs was that DPO would push it up, because the pair corpus prefers placeholder retention by 16.8 points.

## Per-run detail

- [`dpo_b0.1_lr1e-5`](./dpo_b0.1_lr1e-5/summary.md) — loss 0.6730 -> 0.5871, peak 2266 MB

## Stopped early

**Stopped after `dpo_b0.1_lr1e-5`.** reward_accuracy saturated at 1.000 while dev pass@1 (0.2306) stayed at or below the 0.2472 baseline — the policy is learning to rank the pairs, not to write code

Remaining runs were not started. Nothing was retried — a broken config fails the same way the second time.
