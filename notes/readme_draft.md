# README notes (Phase 6)

Material for the public writeup. Not the README — that is written in Phase 6.
Kept here so the numbers are recorded when they are measured rather than
reconstructed months later.

## Which baseline number is the honest one

**Report Qwen2.5-0.5B-Instruct greedy pass@1 = 0.240 on the full tier.**

An earlier baseline in this repo's history reads **0.355**, and it is in the
git log, so it needs explaining rather than quietly dropping.

That number was measured with MBPP's full asserts in the prompt — the same
asserts the reward function grades against. The model could read the expected
outputs of the tests it was about to be scored on, and completions were
observed echoing them back verbatim. It is a real measurement of a leaky task,
not a measurement of the model.

The prompt now carries only a signature stub derived from the first assert —
function name and arity, no expected outputs (`def is_Word_Present(arg0,
arg1):`). Under that prompt the same model, same harness, same seed scores:

| | asserts visible (leaky) | signature stub (honest) |
|---|---|---|
| full greedy pass@1 | 0.355 | **0.240** |
| full temp 0.8 pass@1 | 0.300 | **0.228** |
| full temp 0.8 pass@8 | 0.545 | **0.460** |
| dev greedy pass@1 | 0.322 | **0.289** |

Every improvement this project reports is measured against the right-hand
column. The gap between the columns — about 11 points of greedy pass@1 — is a
useful incidental result in its own right: it is roughly what test visibility
is worth on MBPP at 0.5B, and it is larger than any training effect this
project is likely to produce. Worth stating, because a reader comparing these
numbers against published MBPP results will find most of them closer to the
left-hand column.

## Limitations evidence

See `notes/limitations.md`.

## Prediction, registered before any training run

Recorded here on the strength of the pair statistics alone, before DPO has
been run even once, so that confirming it later is a prediction rather than a
story told about a number after seeing it.

**`stub_args_rate` will rise above its 80–87% baseline after DPO training.**

The prompt hands the model a signature stub with placeholder parameter names,
`def is_Word_Present(arg0, arg1):`. Copying those names verbatim into the
answer is functionally harmless and stylistically poor, and the untrained
policy already does it in 80–87% of completions (`runs/baseline/*.json`).

The preference corpus prefers it. In `data/pairs_train.jsonl` the chosen side
retains the placeholders **78.7%** of the time against the rejected side's
**61.9%** — a 16.8 point gap pointing the wrong way. Nothing in the reward
ladder rewards `arg0`; the correlation exists because a completion that copies
the signature exactly is also more likely to get the function name right and
therefore to pass. DPO cannot tell those apart. It is asked to increase the
probability of chosen relative to rejected, and placeholder retention is one
of the features that separates them.

So the specific claim: after DPO, `stub_args_rate` on the dev tier rises, and
it rises further at larger β. pass@1 will not register it either way, which is
the point — this is the class of regression an execution-verified reward is
structurally blind to, because the code runs correctly either way.

Falsifiable, and it may well be wrong. The 16.8 point gap is a correlation in
a 1,080-pair corpus, and LoRA at rank 16 may simply not have the capacity to
pick up a stylistic feature at all while it is busy with correctness. If the
rate holds flat, that is worth reporting too: it would mean the surface
statistics of the preference data do not transfer as readily as this reasoning
assumes.

Instrumented in `EvalReport.stub_args_rate`, logged at every checkpoint eval
(PROJECT.md §3b).

## Outcome of the stub_args prediction (Phase 4 sweep)

The prediction registered above was that DPO would push `stub_args_rate` above
its 80.8% dev baseline. Measured across seven runs:

| run | stub_args | vs baseline | dev tokens | dev pass@1 |
|---|---|---|---|---|
| dpo β=0.1 lr=1e-5 | 88.3% | +7.5 | 100 | 0.2306 |
| dpo β=0.05 lr=1e-5 | 88.1% | +7.3 | 80 | 0.2500 |
| dpo β=0.3 lr=1e-5 | 84.2% | +3.4 | 111 | 0.2556 |
| dpo β=0.1 lr=5e-6 | 82.8% | +2.0 | 111 | 0.2528 |
| dpo β=0.5 lr=1e-5 | 82.2% | +1.4 | 120 | 0.2639 |
| dpo β=0.1 lr=5e-5 | 70.6% | **−10.2** | 188 | **0.2806** |
| rft lr=1e-5 | 77.5% | −3.3 | 138 | 0.2556 |

**Confirmed in five of six DPO runs, and reversed in the sixth — which is the
run that scored best.** The direction was right and the mechanism plausible,
but it does not survive contact with the learning rate.

Two things are worth taking from this rather than one.

The prediction's reasoning holds where the policy stays near the base
distribution: at lr 5e-6 and 1e-5, retention rises, and it rises most where β
is smallest — exactly where the KL penalty is weakest and the preference data
has the most influence. That is the predicted effect, and β acts on it in the
predicted direction.

At lr 5e-5 the policy moves far enough that the pattern inverts: retention
falls 10 points, completions get 88 tokens longer, and pass@1 is the highest
in the sweep. Something qualitatively different is happening there, and one
run cannot say what. It is the obvious thing to look at next.

There is also a length signal running underneath all of this. Every run that
shortened its completions relative to the 161-token baseline scored at or near
baseline; the one run that lengthened them scored best. The preference corpus
is skewed −55 tokens toward shorter chosen answers, and the runs that learned
that skew did not benefit from it. That is the length pathology §2c predicted,
appearing as predicted, in the direction predicted — and it is the argument
for the balanced-corpus ablation being run rather than assumed.

**None of these pass@1 differences clear noise at 90 problems** (see
`runs/sweep_summary.md`). The stub_args and length numbers are far outside
noise; the capability numbers are not.

## The gate is passed, and the dev tier could never have shown it

Phase 4's sweep ran seven configurations on the dev tier and concluded that
nothing cleared noise. That conclusion was correct about the dev tier and wrong
about the project. Two things were happening at once.

**The dev tier is underpowered for this effect.** 90 problems gives a paired
standard error of ~0.025 against baseline. The full tier (200 problems x 8
samples) gives ~0.013. The strongest dev result in the whole sweep was z = 1.33;
the same class of effect on the full tier reaches z = 4.9.

**The sweep searched the wrong corner.** It moved one axis at a time from
β=0.1, lr=1e-5, exactly as §4 prescribes, and lr=5e-5 was reached only at the
end and only at β=0.1. Running the β axis *at* lr=5e-5 — the tie-break — found
the configurations that matter. The β ordering inverts with learning rate: at
lr 1e-5 the best β was 0.1 and RFT matched every DPO run, and at lr 5e-5 the
best β is 0.5.

| full tier, 200 x 8 | pass@1 | vs base | z |
|---|---|---|---|
| base | 0.2281 | — | — |
| RFT | 0.2450 | +0.0169 | 1.80 |
| DPO β=0.1 lr=1e-5 family | 0.241–0.249 | +0.013…+0.021 | 1.3–1.9 |
| **DPO β=0.3 lr=5e-5** | 0.2888 | +0.0606 | **4.93** |
| **DPO β=0.5 lr=5e-5** | 0.2925 | +0.0644 | **4.89** |

**DPO vs RFT, paired on the same 200 problems: +0.0475 ± 0.0137, z = 3.46.**
That is the project's research question, and the answer at 0.5B on MBPP is that
the negative samples do buy something — roughly 4.8 points of pass@1 over
training on positives alone — but only at a (β, lr) setting that a
one-axis-at-a-time sweep starting from β=0.1, lr=1e-5 does not reach.

The honest framing for the writeup is that both the negative result and the
positive one were produced by the same harness, and the difference between them
was eval power and one unexplored corner. A reader should take the methodology
lesson as seriously as the number: a 90-problem dev tier cannot adjudicate a
3-point effect, and a sweep that moves one axis at a time can rank the winning
region last.

### The length correlation

Across 17 evaluated checkpoints, sorting by generated length separates
significant from non-significant results with zero interleaving: all 7 below 183
tokens fail, all 10 above 184 succeed. The threshold is ~183 tokens, not the
base model's 149 — an RFT seed at 152 tokens and a DPO run at 182 are both
longer than base and both non-significant, so "longer than base" is not the
line.

The preference corpus is skewed −55 tokens toward shorter chosen answers. Runs
that followed that skew shortened and gained nothing; runs that escaped it
gained. Whether length is the cause or a marker of escaping the corpus's
surface statistics is not settled by eight points, and saying which would need
a controlled length intervention rather than an observation.

Length balancing itself was tested directly and is *not* the mechanism:
balanced vs unbalanced at matched β and lr is +0.0094 ± 0.0129 (z = 0.73). The
large effects come from the β/lr corner, on the unbalanced corpus.
