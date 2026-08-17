# Limitations, with evidence

Material for the README's limitations section. Each entry is a real artefact
from a real run, kept verbatim so the writeup quotes evidence rather than
recollection.

## "Passes the asserts" is not "correct"

MBPP ships three asserts per problem. Three asserts do not pin down a
function, and the reward signal cannot tell a correct solution from one that
happens to satisfy them. This is the clearest example the project produced,
from the n=16 preference set (task_id 897, samples 13 and 2, sampled under the
pre-signature-stub prompt).

The task: *"Write a python function to check whether the word is present in a
given sentence or not."* Its asserts:

```python
assert is_Word_Present("machine learning","machine") == True
assert is_Word_Present("easy","fun") == False
assert is_Word_Present("python language","code") == False
```

**Chosen** — reward 1.0, passed 3/3:

```python
def is_Word_Present(sentence, word):
    return word in sentence

# Test the function
print(is_Word_Present("machine learning", "machine"))  # Output: True
print(is_Word_Present("easy", "fun"))  # Output: False
print(is_Word_Present("python language", "code"))  # Output: False
```

**Rejected** — reward 0.3, passed 0/3:

```python
def is_word_present(sentence, word):
    # Split the sentence into words
    words = sentence.split()
    # Check if the word is present in the list of words
    if word in words:
        return True
    else:
        return False
```

The chosen completion is a substring test. It answers `True` for
`is_Word_Present("concatenate", "cat")`, which is wrong, and it passes all
three asserts because none of them probe that case.

The rejected completion splits the sentence into words and checks membership —
the behaviour the task actually describes. It scores 0.3 because it defines
`is_word_present` where the tests call `is_Word_Present`.

So the pair teaches the policy to prefer the substring hack over the correct
algorithm, on the strength of a capital letter. Both labels are honest — the
rejected one genuinely fails to execute — but the preference is backwards with
respect to the task.

**Two things follow.**

1. Name mismatches were 11.1% of all runnable failures in that run. That is
   why the prompt now carries a signature stub (`def is_Word_Present(arg0,
   arg1):`): it removes the naming lottery without handing the model the
   asserts it is scored on. This example is retained because the underlying
   problem — weak tests admitting wrong solutions — is not fixed by the stub.
2. This is the reward-hacking surface §8 predicts for GRPO, visible already in
   offline data. MBPP's `challenge_test_list` is carried through `data.py`
   unused, and is the natural held-out check against it.

## A 0.5B model does not reliably follow a one-line instruction

The prompt was changed to carry a signature stub — `def is_Word_Present(arg0,
arg1):` — under the hypothesis that most `NameError` failures were the model
guessing an identifier it had never been told. The stub was expected to
eliminate most of that class. Measured over 5,984 completions before and
after:

| | asserts in prompt | signature stub |
|---|---|---|
| pass rate | 26.1% | 18.3% |
| NameError share of runnable failures | 11.1% | 8.1% |
| completions failing on the target name | 223 | 185 |
| pairable problems (of 374) | 226 | 180 |

The hypothesis was mostly wrong, and the breakdown says why. Of the 185
completions that still fail on the target name *while being shown that exact
name in the prompt*, 89% define a genuinely different one — `is_prime` for
`prime_num`, `longest_chain_length` for `max_chain_length`, `bell_number` for
`bell_Number`. Only 9% are a casing slip and 2% define no function at all.

So the failure was never primarily an information problem. The model is told
the name and writes the name it considers natural instead. At 0.5B,
instruction-following is itself the bottleneck, which is worth stating plainly
in a writeup about improving a 0.5B model with preference optimization: some
of what looks like a reasoning failure is a formatting failure, and some of
what looks like a formatting failure does not respond to being told.

The stub was kept anyway, for a different reason than the one it was adopted
for: putting the graded asserts in the prompt leaks the reward function, and
completions were observed echoing them back verbatim. The pass rate difference
between the two rows above is therefore not a regression — the earlier, higher
number was measured on a task where the model could read the answers.

## Why the anchor run collapsed after step 90

`dpo_b0.1_lr1e-5` scored 0.2639 on dev at step 90 and 0.2306 at step 135 —
below where it started — under a cosine schedule decaying to zero. A schedule
that anneals to nothing should not usually undo its own progress, so the run
is worth reading closely. The logs say it was over-optimising a length prior,
and that ties it to the project's main finding rather than being a separate
curiosity.

Between the two evals, averaged over 15-step windows:

| | steps 75-90 | steps 120-135 | change |
|---|---|---|---|
| `logp_rejected` | −118.8 | −138.6 | **−19.8** |
| `logp_chosen` | −79.6 | −74.8 | +4.8 |
| implicit reward margin | 0.378 | 0.606 | +0.23 |
| training loss | 0.592 | 0.538 | −0.05 |
| learning rate | 3.7e-6 | 1.1e-7 | annealing |

And what the model actually generated on dev:

| eval | dev pass@1 | mean tokens | unparseable |
|---|---|---|---|
| baseline | 0.2472 | 161 | 0.8% |
| step 45 | 0.2611 | 109 | 0.6% |
| step 90 | 0.2639 | 106 | 0.6% |
| step 135 | 0.2306 | 100 | **1.7%** |

**It is not likelihood displacement.** `logp_chosen` *rose* through the
collapse. The pathology §4 anticipated — both levels falling while the gap
grows — did not happen in any run of the sweep.

What happened instead: `logp_rejected` kept falling, hard, right to the end of
the schedule. The margin was still growing at step 135 and the training loss
was still improving. The optimiser was doing exactly what the objective asked
of it, and the objective was wrong.

The rejected side of this corpus is systematically *longer* — the pre-balance
skew is −55 tokens — so driving `logp_rejected` down is, in large part,
driving down the probability of long output. The model's generations shorten
monotonically: 161 tokens at baseline, 109, 106, 100. By step 135 the
truncation reaches the point of breaking syntax, and unparseable output nearly
triples from 0.6% to 1.7%. That is where the pass@1 went.

The cosine schedule does not protect against this because the damage is not
caused by large steps. It is caused by many small steps all pointing the same
direction, and annealing the learning rate slows the walk without changing its
heading.

Two things corroborate the reading. `dpo_b0.5_lr1e-5`, with five times the KL
penalty, holds its generation length flat (116/122/120 tokens) and does not
collapse (0.2583/0.2556/0.2639) — constraining divergence from the base policy
constrains the length drift with it. And `dpo_b0.1_lr5e-5`, the one run whose
generations got *longer* (143/186/188), scored best in the entire sweep. Every
run that shortened toward the corpus's length prior did worse than the run that
moved away from it.

The corpus statistic that caused this is a documented property of the data
(§2c), which is why length balancing exists as an ablation rather than an
assumption.
