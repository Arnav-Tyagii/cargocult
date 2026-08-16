# rft_lr1e-5

`rft` · 134 steps · effective batch 2x4 · peak 2250 MB

## Configuration

| key | value |
|---|---|
| lr | `1e-05` |
| epochs | `1` |
| batch_size | `2` |
| grad_accum | `4` |
| seed | `0` |
| examples | `1075` |
| n_steps | `134` |

## Training

- loss 0.4322 -> 0.4219 (first third vs last third)
- final grad norm: 1.2301

## Length

- len_completion: 125.5
- len_completion_terminated: 125.5
- n_token_limited: 0
- `*_terminated` excludes completions cut off at the token budget; they are long for a reason unrelated to verbosity

## Dev-tier evals

Baseline dev pass@1 is **0.2472** (runs/baseline/dev_temp0.8.json). The full-tier baseline is 0.2281; these evals are dev-tier.

| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |
|---|---|---|---|---|---|---|
| 45 | 0.2611 | +0.0139 | 129 | 99.2% | 82.2% | 0.8% |
| 90 | 0.2583 | +0.0111 | 143 | 99.7% | 76.1% | 0.0% |
| 134 | 0.2556 | +0.0084 | 138 | 99.4% | 77.5% | 0.3% |
