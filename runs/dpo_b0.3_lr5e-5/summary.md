# dpo_b0.3_lr5e-5

`dpo` · 135 steps · effective batch 1x8 · peak 2266 MB

## Configuration

| key | value |
|---|---|
| beta | `0.3` |
| lr | `5e-05` |
| epochs | `1` |
| batch_size | `1` |
| grad_accum | `8` |
| seed | `0` |
| pairs | `1080` |
| n_steps | `135` |

## Training

- loss 0.7183 -> 0.4804 (first third vs last third)
- final implicit reward margin: 3.1133
- final reward accuracy: 0.8750
- final logp chosen (absolute level): -65.6406
- final logp rejected (absolute level): -247.0000
- final grad norm: 48.4702
- logp_chosen moved +17.4 nats over the run — negative here is likelihood displacement, and the loss curve will not show it

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
| 45 | 0.2500 | +0.0028 | 201 | 95.6% | 90.6% | 2.8% |
| 90 | 0.2611 | +0.0139 | 230 | 93.1% | 83.1% | 0.3% |
| 135 | 0.2833 | +0.0361 | 233 | 93.3% | 81.4% | 0.3% |
