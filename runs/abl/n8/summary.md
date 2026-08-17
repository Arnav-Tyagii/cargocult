# n8

`dpo` · 118 steps · effective batch 1x8 · peak 2267 MB

## Configuration

| key | value |
|---|---|
| beta | `0.5` |
| lr | `5e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `0` |
| pairs | `948` |
| n_steps | `118` |

## Training

- loss 0.5170 -> 0.1027 (first third vs last third)
- final implicit reward margin: 5.7109
- final reward accuracy: 0.8750
- final logp chosen (absolute level): -63.7969
- final logp rejected (absolute level): -111.4531
- final grad norm: 33.3075
- logp_chosen moved +22.3 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

## Length

- len_chosen: 138.5
- len_chosen_terminated: 138.5
- len_rejected: 169.62
- len_rejected_terminated: 139.0
- n_token_limited: 1
- `*_terminated` excludes completions cut off at the token budget; they are long for a reason unrelated to verbosity

## Dev-tier evals

Baseline dev pass@1 is **0.2472** (runs/baseline/dev_temp0.8.json). The full-tier baseline is 0.2281; these evals are dev-tier.

| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |
|---|---|---|---|---|---|---|
| 45 | 0.2917 | +0.0445 | 234 | 88.9% | 82.5% | 1.1% |
| 90 | 0.2889 | +0.0417 | 226 | 89.7% | 84.2% | 0.6% |
| 118 | 0.2778 | +0.0306 | 235 | 90.8% | 84.4% | 0.6% |
