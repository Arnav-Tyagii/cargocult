# dpo_b0.3_lr1e-5

`dpo` · 135 steps · effective batch 1x8 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| beta | `0.3` |
| lr | `1e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `0` |
| pairs | `1080` |
| n_steps | `135` |

## Training

- loss 0.6791 -> 0.5905 (first third vs last third)
- final implicit reward margin: 2.0859
- final reward accuracy: 1.0000
- final logp chosen (absolute level): -58.4531
- final logp rejected (absolute level): -236.3750
- final grad norm: 26.4758
- logp_chosen moved +24.6 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

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
| 45 | 0.2500 | +0.0028 | 114 | 99.2% | 85.3% | 0.3% |
| 90 | 0.2556 | +0.0084 | 113 | 99.7% | 81.7% | 0.0% |
| 135 | 0.2556 | +0.0084 | 111 | 99.7% | 84.2% | 0.8% |
