# rft_s2

`rft` · 134 steps · effective batch 2x4 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| lr | `1e-05` |
| epochs | `1` |
| batch_size | `2` |
| grad_accum | `4` |
| seed | `2` |
| examples | `1075` |
| n_steps | `134` |

## Training

- loss 0.4196 -> 0.4458 (first third vs last third)
- final grad norm: 1.2676

## Length

- len_completion: 127.75
- len_completion_terminated: 127.75
- n_token_limited: 0
- `*_terminated` excludes completions cut off at the token budget; they are long for a reason unrelated to verbosity

## Dev-tier evals

Baseline dev pass@1 is **0.2472** (runs/baseline/dev_temp0.8.json). The full-tier baseline is 0.2281; these evals are dev-tier.

| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |
|---|---|---|---|---|---|---|
| — | no checkpoint evals were run | | | | | |
