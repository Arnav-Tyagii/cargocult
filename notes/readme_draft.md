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
