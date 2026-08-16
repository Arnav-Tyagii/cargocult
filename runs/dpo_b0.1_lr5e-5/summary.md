# dpo_b0.1_lr5e-5

`dpo` · 135 steps · effective batch 1x8 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| beta | `0.1` |
| lr | `5e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `0` |
| pairs | `1080` |
| n_steps | `135` |

## Training

- loss 0.6088 -> 0.4781 (first third vs last third)
- final implicit reward margin: 3.0859
- final reward accuracy: 0.8750
- final logp chosen (absolute level): -74.7031
- final logp rejected (absolute level): -276.5000
- final grad norm: 16.3973
- logp_chosen moved +8.3 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

## Length

- len_chosen: 103.88
- len_chosen_terminated: 103.88
- len_rejected: 299.62
- len_rejected_terminated: 249.0
- n_token_limited: 3
- `*_terminated` excludes completions cut off at the token budget; they are long for a reason unrelated to verbosity

## Dev-tier evals

Baseline dev pass@1 is **0.2472** (runs/baseline/dev_temp0.8.json). The full-tier baseline is 0.2281; these evals are dev-tier.

| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |
|---|---|---|---|---|---|---|
| 45 | 0.2556 | +0.0084 | 143 | 99.4% | 83.9% | 5.6% |
| 90 | 0.2444 | -0.0028 | 186 | 99.4% | 75.6% | 0.3% |
| 135 | 0.2806 | +0.0334 | 188 | 98.3% | 70.6% | 0.3% |
