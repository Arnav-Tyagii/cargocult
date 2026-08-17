# dpo_balanced_b0.1_lr5e-5

`dpo` · 105 steps · effective batch 1x8 · peak 2265 MB

## Configuration

| key | value |
|---|---|
| beta | `0.1` |
| lr | `5e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `0` |
| pairs | `841` |
| n_steps | `105` |

## Training

- loss 0.6414 -> 0.5970 (first third vs last third)
- final implicit reward margin: 0.0752
- final reward accuracy: 0.3750
- final logp chosen (absolute level): -67.0625
- final logp rejected (absolute level): -71.3906
- final grad norm: 19.2205
- logp_chosen moved +12.1 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

## Length

- len_chosen: 141.38
- len_chosen_terminated: 141.38
- len_rejected: 144.5
- len_rejected_terminated: 144.5
- n_token_limited: 0
- `*_terminated` excludes completions cut off at the token budget; they are long for a reason unrelated to verbosity

## Dev-tier evals

Baseline dev pass@1 is **0.2472** (runs/baseline/dev_temp0.8.json). The full-tier baseline is 0.2281; these evals are dev-tier.

| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |
|---|---|---|---|---|---|---|
| 45 | 0.2583 | +0.0111 | 167 | 98.6% | 87.2% | 5.3% |
| 90 | 0.2611 | +0.0139 | 245 | 89.7% | 82.2% | 0.8% |
| 105 | 0.2556 | +0.0084 | 241 | 91.9% | 82.2% | 0.8% |
