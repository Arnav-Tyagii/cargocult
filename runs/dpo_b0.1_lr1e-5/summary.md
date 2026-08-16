# dpo_b0.1_lr1e-5

`dpo` · 135 steps · effective batch 1x8 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| beta | `0.1` |
| lr | `1e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `0` |
| pairs | `1080` |
| n_steps | `135` |

## Training

- loss 0.6730 -> 0.5871 (first third vs last third)
- final implicit reward margin: 1.4409
- final reward accuracy: 1.0000
- final logp chosen (absolute level): -61.9844
- final logp rejected (absolute level): -247.3750
- final grad norm: 13.9010
- logp_chosen moved +21.0 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

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
| 45 | 0.2611 | +0.0139 | 109 | 99.7% | 86.4% | 0.6% |
| 90 | 0.2639 | +0.0167 | 106 | 99.4% | 88.6% | 0.6% |
| 135 | 0.2306 | -0.0166 | 100 | 99.4% | 88.3% | 1.7% |
