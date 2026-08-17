# Phase 5 — seeds, the beta x lr grid, and ablations

## Headline: seed replication on the full tier

200 problems x 8 samples from the test split. Three seeds per arm, with a **fixed checkpoint per arm** (DPO step 100, RFT step 50) rather than re-selecting the best checkpoint per seed, so no seed is individually cherry-picked. Seeds shuffle the training data as well as initialisation and dropout.

| arm | seeds | mean +- SD | vs base | z |
|---|---|---|---|---|
| base | — | 0.2281 | — | — |
| RFT lr 1e-5 | 0.2344 / 0.2325 / 0.2444 | 0.2371 +- 0.0064 | +0.0090 | 1.15 (ns) |
| **DPO beta 0.5 lr 5e-5** | 0.2800 / 0.2931 / 0.2762 | **0.2831 +- 0.0089** | **+0.0550** | **4.85** |

**DPO vs RFT, paired per problem pooling seeds: +0.0460 +- 0.0097, z = 4.73, 95% CI [+0.0270, +0.0651].** Treating the seed as the unit of analysis instead gives t(4) = 7.30 against a critical 2.78, so the conclusion does not depend on the choice of test.

RFT does not reliably beat the base model. That is the sharper half of the result: it is not that DPO wins by a little, it is that training on positives alone barely moves this model while training on the contrast does.

pass@8 is 0.4717 (DPO) against 0.4600 (base) — the gain is in making the first sample right, not in expanding reachable coverage.

## The beta x lr grid

Dev tier, since this is a question about shape rather than significance.

| dev pass@1 (best checkpoint) | beta=0.05 | beta=0.1 | beta=0.3 | beta=0.5 |
|---|---|---|---|---|
| lr 1e-5 | 0.2500 | 0.2639 | 0.2556 | 0.2639 |
| lr 5e-5 | **0.1917** | 0.2806 | 0.2833 | **0.2861** |

The axes are not separable, and the grid shows why a one-axis sweep from beta=0.1, lr=1e-5 could not find the answer:

- At **lr 1e-5** the beta axis is nearly flat (0.250-0.264) and RFT matches all of it.
- At **lr 5e-5** beta matters a great deal and the ordering is monotonic upward.
- **beta=0.05 at lr 5e-5 collapses** — 0.075 dev pass@1 at step 45 with generations down to 34 tokens, recovering only to 0.19. This is beta collapse with both endpoints visible: the same beta is harmless at lr 1e-5 (0.2500) and catastrophic at lr 5e-5. The KL penalty and the step size trade off directly, and neither axis means anything without the other.

## Ablations

All at beta=0.5, lr=5e-5 — the winning configuration — on the dev tier (baseline 0.2472).

| ablation | dev pass@1 (best) | vs default | reading |
|---|---|---|---|
| no near-duplicate filter | — | — | corpus **byte-identical** to the default: 0 differing pairs. The filter dropped 9 of 7,537 candidates and none were in any problem's top 6, because the per-problem cap of 6 binds first. It cannot have removed gradient noise, because it removed nothing. |
| binary reward vs ladder | 0.2917 | +0.0056 | same 1,080 pairs and 180 problems, but 976 of the pairs differ and the length skew falls from -55 to -38 tokens. Performance is indistinguishable, so the shaped ladder is **not** doing work in DPO pair construction — with `chosen` restricted to full passes its only influence is which candidates survive the cap. It still matters for RFT weighting and GRPO group advantages. |
| N=8 vs N=16 samples | 0.2917 | +0.0056 | 948 pairs from 158 problems against 1,080 from 180. Doubling samples rescued 28 of 167 previously all-fail problems (16.8%) and pushed the length skew from -22.8 to -55.0 tokens. Halving the sample budget costs little here. |
| DPO on top of RFT | 0.3000 | +0.0139 | **the best result in the project on dev.** The sequential recipe beats either method alone, which is consistent with the two objectives contributing different things rather than competing. Not confirmed on the full tier. |
| length-balanced corpus | — | — | +0.0094 +- 0.0129 (z=0.73) on the **full** tier at matched beta and lr. Removing the length skew directly does approximately nothing — see below. |

## Length remains a marker, not a mechanism

The correlation is perfect across every checkpoint evaluated: those generating longer than the base model's 149 tokens beat it, those at or below do not. The tempting conclusion is that DPO learns the pair corpus's -55 token skew and that fixing the skew fixes the model.

**The balanced-corpus ablation falsifies that.** Balancing lengths explicitly, at a cost of 239 of 1,080 pairs, moves the full-tier result by +0.0094 +- 0.0129 — nothing. The large effects came from the beta/lr corner, on the unbalanced corpus. Length co-varies with escaping the base distribution's surface statistics; it is not the lever.

This should not be softened in the writeup. The cleaner causal story is available and wrong, and the ablation that refutes it was run precisely so the claim would not have to rest on the correlation.

Establishing what length is standing in for needs a controlled intervention — clamping generation length at eval time on a fixed checkpoint — not another observation.
