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
