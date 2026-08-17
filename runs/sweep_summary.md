# Phase 4 sweep

Dev-tier evals throughout (90 problems x 4 samples, temperature 0.8). The dev baseline is **0.2472**; the full-tier baseline is 0.2281 and is *not* what these are compared against.

## Dev pass@1

| run | dev pass@1 | vs baseline | best across checkpoints | steps |
|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | 0.2306 | -0.0166 | 0.2639 | 135 |
| `rft_lr1e-5` | 0.2556 | +0.0084 | 0.2611 | 134 |
| `dpo_b0.05_lr1e-5` | 0.2500 | +0.0028 | 0.2500 | 135 |
| `dpo_b0.3_lr1e-5` | 0.2556 | +0.0084 | 0.2556 | 135 |
| `dpo_b0.5_lr1e-5` | 0.2639 | +0.0167 | 0.2639 | 135 |
| `dpo_b0.1_lr5e-6` | 0.2528 | +0.0056 | 0.2778 | 135 |
| `dpo_b0.1_lr5e-5` | 0.2806 | +0.0334 | 0.2806 | 135 |

## Likelihood displacement

DPO constrains the gap, never the levels. A run whose `logp_chosen` fell while its margin grew is displacing likelihood — the loss curve will look fine.

| run | logp_chosen | logp_rejected | chosen drift | margin | reward_acc |
|---|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | -61.98 | -247.38 | +21.0 | 1.44 | 1.00 |
| `rft_lr1e-5` | — | — | +nan | — | — |
| `dpo_b0.05_lr1e-5` | -65.44 | -257.50 | +17.6 | 1.05 | 1.00 |
| `dpo_b0.3_lr1e-5` | -58.45 | -236.38 | +24.6 | 2.09 | 1.00 |
| `dpo_b0.5_lr1e-5` | -57.88 | -233.50 | +25.1 | 2.33 | 1.00 |
| `dpo_b0.1_lr5e-6` | -58.11 | -233.62 | +24.9 | 0.46 | 1.00 |
| `dpo_b0.1_lr5e-5` | -74.70 | -276.50 | +8.3 | 3.09 | 0.88 |

## Length and style

`*_terminated` excludes completions cut off at the token budget. The two columns differ because 11.6% of the rejected side of the corpus was truncated, and a truncated completion is long for a reason unrelated to verbosity.

| run | train len (all) | train len (terminated) | dev mean tokens | stub_args |
|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | 103.88 | 103.88 | 99.79 | 88.3% |
| `rft_lr1e-5` | 125.50 | 125.50 | 138.23 | 77.5% |
| `dpo_b0.05_lr1e-5` | 103.88 | 103.88 | 79.80 | 88.1% |
| `dpo_b0.3_lr1e-5` | 103.88 | 103.88 | 111.32 | 84.2% |
| `dpo_b0.5_lr1e-5` | 103.88 | 103.88 | 119.52 | 82.2% |
| `dpo_b0.1_lr5e-6` | 103.88 | 103.88 | 111.46 | 82.8% |
| `dpo_b0.1_lr5e-5` | 103.88 | 103.88 | 187.62 | 70.6% |

Baseline `stub_args_rate` is 80.8% on dev. The prediction registered in `notes/limitations.md` before any of these runs was that DPO would push it up, because the pair corpus prefers placeholder retention by 16.8 points.

## Is any of this outside the noise?

Best checkpoint of each run against the baseline, paired on the same 90 dev problems. Paired because both are scored on the same problems, which removes problem difficulty from the comparison — it is the most powerful test available here, and it still is not powerful enough.

| run | best pass@1 | diff vs baseline | paired SE | z | 95% CI |
|---|---|---|---|---|---|
| `dpo_b0.1_lr1e-5` | 0.2639 | +0.0167 | 0.0249 | 0.67 | [-0.0322, +0.0655] |
| `rft_lr1e-5` | 0.2611 | +0.0139 | 0.0207 | 0.67 | [-0.0266, +0.0544] |
| `dpo_b0.05_lr1e-5` | 0.2500 | +0.0028 | 0.0245 | 0.11 | [-0.0453, +0.0508] |
| `dpo_b0.3_lr1e-5` | 0.2556 | +0.0083 | 0.0254 | 0.33 | [-0.0415, +0.0582] |
| `dpo_b0.5_lr1e-5` | 0.2639 | +0.0167 | 0.0226 | 0.74 | [-0.0277, +0.0610] |
| `dpo_b0.1_lr5e-6` | 0.2778 | +0.0306 | 0.0230 | 1.33 | [-0.0145, +0.0756] |
| `dpo_b0.1_lr5e-5` | 0.2806 | +0.0333 | 0.0304 | 1.10 | [-0.0262, +0.0929] |

**Every interval contains zero.** The largest effect in the sweep is z = 1.33. The point estimates all favour training, which is weak evidence that something real is happening, but at 90 problems nothing here clears noise — including DPO's best against RFT's best, which is +0.019 with a standard error of 0.025.

This is the §4 week-4 gate. Resolving it needs the full tier (200 problems x 8 samples), where the same effect would carry roughly half the standard error, not more sweeping at dev.

## How the stop conditions are measured

- **divergence**: mean loss over the last third of steps above the first third.
- **saturated ranking**: mean `reward_accuracy` over the last third above 0.95, *and* the best dev pass@1 across all checkpoints at or below baseline. Both halves are trends on purpose. At batch size 1 with grad_accum 8, `reward_accuracy` is a mean of 8 single-pair judgments and swings between 0.5 and 1.0 between adjacent steps, so a single final-step reading of 1.000 says nothing; and a run that beat baseline at step 90 before falling back has produced a checkpoint and a finding, which is not the pathology this screens for.
- **OOM**: any allocation failure. Not retried.

## Per-run detail

- [`dpo_b0.1_lr1e-5`](./dpo_b0.1_lr1e-5/summary.md) — loss 0.6730 -> 0.5871, peak 2266 MB
- [`rft_lr1e-5`](./rft_lr1e-5/summary.md) — loss 0.4322 -> 0.4219, peak 2250 MB
- [`dpo_b0.05_lr1e-5`](./dpo_b0.05_lr1e-5/summary.md) — loss 0.6796 -> 0.5998, peak 2266 MB
- [`dpo_b0.3_lr1e-5`](./dpo_b0.3_lr1e-5/summary.md) — loss 0.6791 -> 0.5905, peak 2266 MB
- [`dpo_b0.5_lr1e-5`](./dpo_b0.5_lr1e-5/summary.md) — loss 0.6874 -> 0.6160, peak 2266 MB
- [`dpo_b0.1_lr5e-6`](./dpo_b0.1_lr5e-6/summary.md) — loss 0.6866 -> 0.6373, peak 2266 MB
- [`dpo_b0.1_lr5e-5`](./dpo_b0.1_lr5e-5/summary.md) — loss 0.6088 -> 0.4781, peak 2266 MB
