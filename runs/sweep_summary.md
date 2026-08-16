# Phase 4 sweep

Dev-tier evals throughout (90 problems x 4 samples, temperature 0.8). The dev baseline is **0.2472**; the full-tier baseline is 0.2281 and is *not* what these are compared against.

## Dev pass@1

| run | dev pass@1 | vs baseline | best across checkpoints | steps |
|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | 0.2306 | -0.0166 | 0.2639 | 135 |
| `rft_lr1e-5` | 0.2556 | +0.0084 | 0.2611 | 134 |
| `dpo_b0.05_lr1e-5` | 0.2500 | +0.0028 | 0.2500 | 135 |

## Likelihood displacement

DPO constrains the gap, never the levels. A run whose `logp_chosen` fell while its margin grew is displacing likelihood — the loss curve will look fine.

| run | logp_chosen | logp_rejected | chosen drift | margin | reward_acc |
|---|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | -61.98 | -247.38 | +21.0 | 1.44 | 1.00 |
| `rft_lr1e-5` | — | — | +nan | — | — |
| `dpo_b0.05_lr1e-5` | -65.44 | -257.50 | +17.6 | 1.05 | 1.00 |

## Length and style

`*_terminated` excludes completions cut off at the token budget. The two columns differ because 11.6% of the rejected side of the corpus was truncated, and a truncated completion is long for a reason unrelated to verbosity.

| run | train len (all) | train len (terminated) | dev mean tokens | stub_args |
|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | 103.88 | 103.88 | 99.79 | 88.3% |
| `rft_lr1e-5` | 125.50 | 125.50 | 138.23 | 77.5% |
| `dpo_b0.05_lr1e-5` | 103.88 | 103.88 | 79.80 | 88.1% |

Baseline `stub_args_rate` is 80.8% on dev. The prediction registered in `notes/readme_draft.md` before any of these runs was that DPO would push it up, because the pair corpus prefers placeholder retention by 16.8 points.

## How the stop conditions are measured

- **divergence**: mean loss over the last third of steps above the first third.
- **saturated ranking**: mean `reward_accuracy` over the last third above 0.95, *and* the best dev pass@1 across all checkpoints at or below baseline. Both halves are trends on purpose. At batch size 1 with grad_accum 8, `reward_accuracy` is a mean of 8 single-pair judgments and swings between 0.5 and 1.0 between adjacent steps, so a single final-step reading of 1.000 says nothing; and a run that beat baseline at step 90 before falling back has produced a checkpoint and a finding, which is not the pathology this screens for.
- **OOM**: any allocation failure. Not retried.

## Per-run detail

- [`dpo_b0.1_lr1e-5`](./dpo_b0.1_lr1e-5/summary.md) — loss 0.6730 -> 0.5871, peak 2266 MB
- [`rft_lr1e-5`](./rft_lr1e-5/summary.md) — loss 0.4322 -> 0.4219, peak 2250 MB
- [`dpo_b0.05_lr1e-5`](./dpo_b0.05_lr1e-5/summary.md) — loss 0.6796 -> 0.5998, peak 2266 MB
