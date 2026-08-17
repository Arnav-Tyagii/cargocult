# dpo_on_rft

`dpo` · 135 steps · effective batch 1x8 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| beta | `0.5` |
| lr | `5e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `0` |
| pairs | `1080` |
| n_steps | `135` |

## Training

- loss 0.4759 -> 0.0554 (first third vs last third)
- final implicit reward margin: 9.0859
- final reward accuracy: 1.0000
- final logp chosen (absolute level): -46.9688
- final logp rejected (absolute level): -120.4531
- final grad norm: 0.7404
- logp_chosen moved +7.0 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

## Length

- len_chosen: 101.0
- len_chosen_terminated: 101.0
- len_rejected: 209.62
- len_rejected_terminated: 184.71
- n_token_limited: 1
- `*_terminated` excludes completions cut off at the token budget; they are long for a reason unrelated to verbosity

## Dev-tier evals

Baseline dev pass@1 is **0.2472** (runs/baseline/dev_temp0.8.json). The full-tier baseline is 0.2281; these evals are dev-tier.

| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |
|---|---|---|---|---|---|---|
| 45 | 0.2917 | +0.0445 | 182 | 98.1% | 73.6% | 0.3% |
| 90 | 0.3000 | +0.0528 | 187 | 96.1% | 66.1% | 0.3% |
| 135 | 0.3000 | +0.0528 | 217 | 95.8% | 70.6% | 0.3% |
