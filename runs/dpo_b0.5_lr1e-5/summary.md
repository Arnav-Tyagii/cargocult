# dpo_b0.5_lr1e-5

`dpo` · 135 steps · effective batch 1x8 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| beta | `0.5` |
| lr | `1e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `0` |
| pairs | `1080` |
| n_steps | `135` |

## Training

- loss 0.6874 -> 0.6160 (first third vs last third)
- final implicit reward margin: 2.3281
- final reward accuracy: 1.0000
- final logp chosen (absolute level): -57.8750
- final logp rejected (absolute level): -233.5000
- final grad norm: 43.8333
- logp_chosen moved +25.1 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

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
| 45 | 0.2583 | +0.0111 | 116 | 99.2% | 85.8% | 0.3% |
| 90 | 0.2556 | +0.0084 | 122 | 99.4% | 83.1% | 0.0% |
| 135 | 0.2639 | +0.0167 | 120 | 99.7% | 82.2% | 0.0% |
