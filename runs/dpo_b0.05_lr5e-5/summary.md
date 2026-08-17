# dpo_b0.05_lr5e-5

`dpo` · 135 steps · effective batch 1x8 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| beta | `0.05` |
| lr | `5e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `0` |
| pairs | `1080` |
| n_steps | `135` |

## Training

- loss 0.5477 -> 0.2284 (first third vs last third)
- final implicit reward margin: 2.5898
- final reward accuracy: 0.8750
- final logp chosen (absolute level): -78.8594
- final logp rejected (absolute level): -186.0000
- final grad norm: 9.1658
- logp_chosen moved -24.6 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

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
| 45 | 0.0750 | -0.1722 | 34 | 100.0% | 50.8% | 25.3% |
| 90 | 0.1917 | -0.0555 | 180 | 98.3% | 76.4% | 4.2% |
| 135 | 0.1722 | -0.0750 | 199 | 97.8% | 85.8% | 3.9% |
