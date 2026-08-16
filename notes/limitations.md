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
