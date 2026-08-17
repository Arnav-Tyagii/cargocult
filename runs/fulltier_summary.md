# Full-tier results — the Phase 4 gate

200 problems x 8 samples from the test split, temperature 0.8. Checkpoints were selected on the dev tier and evaluated here, so selection and measurement do not share problems.

**Base model: 0.2281**, 149 tokens, 82.4% stub_args.

## Against the base, paired per problem

| checkpoint | pass@1 | diff | SE | z | 95% CI | significant |
|---|---|---|---|---|---|---|
| `dpo_b0.5_lr5e-5@checkpoint_100` | 0.2925 | +0.0644 | 0.0132 | 4.89 | [+0.0386, +0.0902] | **yes** |
| `dpo_b0.3_lr5e-5@checkpoint_final` | 0.2888 | +0.0606 | 0.0123 | 4.93 | [+0.0365, +0.0848] | **yes** |
| `dpo_balanced_b0.1_lr5e-5@checkpoint_100` | 0.2587 | +0.0306 | 0.0127 | 2.41 | [+0.0057, +0.0556] | **yes** |
| `dpo_balanced_b0.1_lr5e-5@checkpoint_final` | 0.2587 | +0.0306 | 0.0132 | 2.33 | [+0.0048, +0.0564] | **yes** |
| `dpo_b0.1_lr5e-5@checkpoint_final` | 0.2494 | +0.0213 | 0.0149 | 1.42 | [-0.0080, +0.0505] | no |
| `dpo_b0.1_lr5e-6@checkpoint_100` | 0.2462 | +0.0181 | 0.0094 | 1.93 | [-0.0003, +0.0365] | no |
| `rft_lr1e-5@checkpoint_50` | 0.2450 | +0.0169 | 0.0094 | 1.80 | [-0.0015, +0.0352] | no |
| `dpo_b0.5_lr1e-5@checkpoint_final` | 0.2412 | +0.0131 | 0.0104 | 1.26 | [-0.0072, +0.0335] | no |

Ten comparisons were run against this base. Bonferroni at alpha=0.05 needs |z| > 2.81; the top two clear it comfortably, the two balanced checkpoints do not.

## The question the project exists to answer

**Best DPO (0.2925) vs RFT (0.2450), paired: +0.0475 +- 0.0137, z = 3.46, 95% CI [+0.0206, +0.0744].**

DPO's use of negative samples does buy something over training on positives alone — but only in a corner of the hyperparameter space the dev-tier sweep ranked last. At lr 1e-5 the best beta was 0.1 and RFT matched every DPO run; at lr 5e-5 the ordering inverts and higher beta wins.

## Length separates the winners from the losers perfectly

| checkpoint | tokens | vs base 149 | stub_args | token limit | unparseable | significant |
|---|---|---|---|---|---|---|
| `dpo_b0.5_lr5e-5@checkpoint_100` | 217 | +68 | 74.8% | 3.4% | 0.3% | **yes** |
| `dpo_b0.3_lr5e-5@checkpoint_final` | 224 | +75 | 80.5% | 4.6% | 0.6% | **yes** |
| `dpo_balanced_b0.1_lr5e-5@checkpoint_100` | 238 | +89 | 82.0% | 6.5% | 0.6% | **yes** |
| `dpo_balanced_b0.1_lr5e-5@checkpoint_final` | 237 | +88 | 82.4% | 6.8% | 0.8% | **yes** |
| `dpo_b0.1_lr5e-5@checkpoint_final` | 182 | +33 | 71.5% | 0.8% | 0.2% | no |
| `dpo_b0.1_lr5e-6@checkpoint_100` | 107 | -42 | 85.1% | 0.5% | 0.4% | no |
| `rft_lr1e-5@checkpoint_50` | 128 | -21 | 79.0% | 0.9% | 0.8% | no |
| `dpo_b0.5_lr1e-5@checkpoint_final` | 117 | -32 | 81.2% | 0.4% | 0.2% | no |

Every checkpoint that generates **longer** than the base model beats it significantly (217-238 tokens). Every checkpoint that generates **at or below** base length does not (107-182 tokens). The split is clean with no exceptions across eight checkpoints.

This is the same axis as the anchor-run collapse (`notes/limitations.md`): the preference corpus is skewed -55 tokens toward shorter chosen answers, runs that follow that skew shorten and gain nothing, and runs that escape it gain. Whether length is the cause or a marker of escaping the corpus's surface statistics is not settled by these eight points.

## Length balancing

Balanced vs unbalanced at the same beta=0.1, lr=5e-5, paired: +0.0094 +- 0.0129, z = 0.73. The balanced corpus is nominally ahead and costs 239 of 1,080 pairs to build, but the difference does not clear noise. It is not the mechanism behind the large effects above — those came from the beta/lr corner, on the unbalanced corpus.

## Caveats

- Checkpoints are saved at steps 50/100/final while evals ran at 45/90/135, so a mid-run peak is evaluated at the nearest saved adapter, up to 10 steps away.
- One seed per configuration. The beta x lr interaction is inferred from single runs at each corner.
- The dev tier that drove the sweep has a paired SE of ~0.025 and could not have detected any of this; that is what made the full tier necessary rather than optional.
