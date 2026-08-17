# dpo_on_rft_s2

`dpo` · 135 steps · effective batch 1x8 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| beta | `0.5` |
| lr | `5e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `2` |
| pairs | `1080` |
| n_steps | `135` |

## Training

- loss 0.4577 -> 0.0729 (first third vs last third)
- final implicit reward margin: 9.1641
- final reward accuracy: 1.0000
- final logp chosen (absolute level): -48.4062
- final logp rejected (absolute level): -120.9062
- final grad norm: 1.7845
- logp_chosen moved +21.7 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

## Length

- len_chosen: 124.75
- len_chosen_terminated: 124.75
- len_rejected: 180.25
- len_rejected_terminated: 180.25
- n_token_limited: 0
- `*_terminated` excludes completions cut off at the token budget; they are long for a reason unrelated to verbosity

## Dev-tier evals

Baseline dev pass@1 is **0.2472** (runs/baseline/dev_temp0.8.json). The full-tier baseline is 0.2281; these evals are dev-tier.

| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |
|---|---|---|---|---|---|---|
| — | no checkpoint evals were run | | | | | |
