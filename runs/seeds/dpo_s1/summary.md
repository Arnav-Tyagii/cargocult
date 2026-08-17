# dpo_s1

`dpo` · 135 steps · effective batch 1x8 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| beta | `0.5` |
| lr | `5e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `1` |
| pairs | `1080` |
| n_steps | `135` |

## Training

- loss 0.5059 -> 0.0788 (first third vs last third)
- final implicit reward margin: 9.1406
- final reward accuracy: 1.0000
- final logp chosen (absolute level): -90.0625
- final logp rejected (absolute level): -158.8750
- final grad norm: 14.3210
- logp_chosen moved -13.3 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

## Length

- len_chosen: 147.0
- len_chosen_terminated: 147.0
- len_rejected: 282.25
- len_rejected_terminated: 248.33
- n_token_limited: 2
- `*_terminated` excludes completions cut off at the token budget; they are long for a reason unrelated to verbosity

## Dev-tier evals

Baseline dev pass@1 is **0.2472** (runs/baseline/dev_temp0.8.json). The full-tier baseline is 0.2281; these evals are dev-tier.

| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |
|---|---|---|---|---|---|---|
| — | no checkpoint evals were run | | | | | |
