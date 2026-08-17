# binary

`dpo` · 135 steps · effective batch 1x8 · peak 2267 MB

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

- loss 0.5802 -> 0.1149 (first third vs last third)
- final implicit reward margin: 6.0625
- final reward accuracy: 1.0000
- final logp chosen (absolute level): -50.8750
- final logp rejected (absolute level): -108.9844
- final grad norm: 3.5461
- logp_chosen moved +22.0 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

## Length

- len_chosen: 105.0
- len_chosen_terminated: 105.0
- len_rejected: 174.75
- len_rejected_terminated: 174.75
- n_token_limited: 0
- `*_terminated` excludes completions cut off at the token budget; they are long for a reason unrelated to verbosity

## Dev-tier evals

Baseline dev pass@1 is **0.2472** (runs/baseline/dev_temp0.8.json). The full-tier baseline is 0.2281; these evals are dev-tier.

| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |
|---|---|---|---|---|---|---|
| 45 | 0.2556 | +0.0084 | 135 | 99.2% | 82.8% | 0.6% |
| 90 | 0.2528 | +0.0056 | 190 | 97.2% | 73.6% | 0.8% |
| 135 | 0.2917 | +0.0445 | 198 | 97.2% | 77.5% | 0.6% |
