"""MBPP loading, the task_id splits, and prompt/response formatting.

The split is by task_id and is fixed for the whole project (PROJECT.md §2).
It is asserted disjoint and complete at import time rather than trusted,
because the one bug that cannot be recovered from later is training on the
problems that produced the headline number.

MBPP's own HuggingFace splits happen to use these same boundaries, but the
ranges are applied here explicitly so that a change in the dataset's split
layout shows up as a count mismatch instead of silently reshuffling what
"test" means halfway through the project.
"""

from __future__ import annotations

import ast
import random
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

DATASET = "google-research-datasets/mbpp"
CONFIG = "full"

# Inclusive task_id ranges. fewshot 1-10 · test 11-510 · dev 511-600 · train 601-974
SPLITS: dict[str, tuple[int, int]] = {
    "fewshot": (1, 10),
    "test": (11, 510),
    "dev": (511, 600),
    "train": (601, 974),
}

EXPECTED_COUNTS = {"fewshot": 10, "test": 500, "dev": 90, "train": 374}


def _assert_splits_are_a_partition() -> None:
    """No task_id in two splits, and every id from 1..974 in exactly one."""
    owner: dict[int, str] = {}
    for name, (lo, hi) in SPLITS.items():
        assert lo <= hi, f"{name}: empty range {lo}..{hi}"
        assert hi - lo + 1 == EXPECTED_COUNTS[name], (
            f"{name}: range {lo}..{hi} holds {hi - lo + 1} ids, "
            f"expected {EXPECTED_COUNTS[name]}"
        )
        for task_id in range(lo, hi + 1):
            assert task_id not in owner, (
                f"task_id {task_id} is in both '{owner[task_id]}' and '{name}' — "
                "the splits overlap"
            )
            owner[task_id] = name
    assert set(owner) == set(range(1, 975)), "splits do not cover task_id 1..974"


_assert_splits_are_a_partition()


@dataclass
class Problem:
    """One MBPP problem. Field names follow the dataset's own."""

    task_id: int
    text: str  # the natural-language task description
    code: str  # the reference solution, used by RFT/DPO for nothing — see note
    test_list: list[str]  # the 3 asserts that define correctness
    test_setup_code: str = ""  # rarely non-empty; runs before the asserts
    challenge_test_list: list[str] = field(default_factory=list)

    # Note on `code`: the reference solution is deliberately *not* used as a
    # training target anywhere in this project. The signal is execution, not
    # imitation. It is carried because §6's contamination check needs it.

    @property
    def split(self) -> str:
        return split_of(self.task_id)


def split_of(task_id: int) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= task_id <= hi:
            return name
    raise KeyError(f"task_id {task_id} is outside MBPP")


@lru_cache(maxsize=1)
def _all_rows() -> tuple[dict, ...]:
    """Every MBPP row, from every HuggingFace split, keyed only by task_id."""
    from datasets import load_dataset  # imported lazily: heavy, and not always needed

    dataset = load_dataset(DATASET, CONFIG)
    rows = [dict(row) for split in dataset.values() for row in split]
    rows.sort(key=lambda row: row["task_id"])
    return tuple(rows)


def load_problems(split: str) -> list[Problem]:
    """All problems in one of the splits above, ordered by task_id."""
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    lo, hi = SPLITS[split]
    problems = [
        Problem(
            task_id=row["task_id"],
            text=row["text"],
            code=row["code"],
            test_list=list(row["test_list"]),
            test_setup_code=row.get("test_setup_code") or "",
            challenge_test_list=list(row.get("challenge_test_list") or []),
        )
        for row in _all_rows()
        if lo <= row["task_id"] <= hi
    ]
    expected = EXPECTED_COUNTS[split]
    assert len(problems) == expected, (
        f"{split}: loaded {len(problems)} problems, expected {expected}. "
        "The dataset changed under the split ranges."
    )
    return problems


def subsample(problems: Sequence[Problem], n: int, seed: int) -> list[Problem]:
    """A deterministic n-problem subset, returned in task_id order.

    Used to cut the 500-problem test split down to an eval tier. Seeded
    separately from generation so that every checkpoint is scored on the same
    problems — a subset that moved between runs would make two checkpoints
    incomparable, which is the whole point of the exercise.
    """
    problems = sorted(problems, key=lambda p: p.task_id)
    if n >= len(problems):
        return list(problems)
    chosen = random.Random(seed).sample(range(len(problems)), n)
    return [problems[i] for i in sorted(chosen)]


# --- prompts -----------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Write a Python function for the following task.\n\n"
    "Task: {text}\n\n"
    "Your solution must define this function:\n{stub}\n\n"
    "Reply with a single Python code block containing the complete solution."
)

# A signature stub, not the asserts. MBPP descriptions do not state the
# function name, so with nothing the model is graded on guessing an
# identifier — 11.1% of runnable failures in the first generation run were a
# NameError from exactly that. But putting the *full* asserts in the prompt,
# as the standard MBPP protocol does, hands the model the reward function's
# own test cases, and completions were observed echoing them back. The stub
# is the minimum that makes the task well-posed: name and arity, no expected
# outputs, generic argument names so nothing about the semantics leaks.

# Calls that wrap the function under test rather than being it.
_ASSERT_WRAPPERS = frozenset({
    "abs", "all", "any", "bool", "dict", "float", "frozenset", "int", "isinstance",
    "iter", "len", "list", "max", "min", "next", "repr", "round", "set", "sorted",
    "str", "sum", "tuple", "type", "zip", "map", "filter", "range", "print",
})


def signature_stub(problem: "Problem") -> str:
    """`def name(arg0, arg1):`, derived from the problem's first assert.

    The asserts are the only place the required identifier appears, so it is
    parsed out of them rather than guessed. Wrapper calls are skipped: MBPP
    writes things like `assert math.isclose(func(3), 4.5)` and
    `assert set(func(x)) == {1, 2}`, where the outermost call is not the
    function being asked for.
    """
    for test in problem.test_list:
        try:
            with warnings.catch_warnings():
                # Test strings carry regex literals; the tokenizer warns about
                # "\d" and friends and that noise is not ours to emit.
                warnings.simplefilter("ignore")
                tree = ast.parse(test)
        except SyntaxError:
            continue
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        # Prefer a call that is not a wrapper, but fall back to allowing them:
        # MBPP task 126 asks for a function actually named `sum`, and excluding
        # it on principle would leave that problem with no signature at all.
        named = [c for c in calls if c.func.id not in _ASSERT_WRAPPERS] or calls
        if not named:
            continue
        # Source order, so the left-most call wins when several qualify.
        call = min(named, key=lambda c: (c.lineno, c.col_offset))
        args = ", ".join(f"arg{i}" for i in range(len(call.args)))
        return f"def {call.func.id}({args}):"
    return ""


# The contamination probe's prompt: signature, no task description (§6). A
# solution produced from this cannot have come from reading the task.
NO_DESCRIPTION_TEMPLATE = (
    "Write the body of this Python function:\n{stub}\n\n"
    "Reply with a single Python code block containing the complete solution."
)


def format_prompt(
    problem: Problem, tokenizer, fewshot: Sequence[Problem] = (),
    include_description: bool = True,
) -> str:
    """The exact string handed to the model, chat template already applied.

    Falls back to plain concatenation when the tokenizer has no chat template
    — PythonGPT, the 54.6M from-scratch baseline, is not an instruct model and
    has none.
    """
    turns: list[tuple[str, str]] = []
    for example in fewshot:
        turns.append(("user", _user_message(example)))
        turns.append(("assistant", as_code_block(example.code)))
    turns.append(("user", _user_message(problem, include_description)))

    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": role, "content": content} for role, content in turns]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n\n".join(content for _, content in turns) + "\n\n"


def _user_message(problem: Problem, include_description: bool = True) -> str:
    if not include_description:
        return NO_DESCRIPTION_TEMPLATE.format(stub=signature_stub(problem))
    return PROMPT_TEMPLATE.format(
        text=problem.text.strip(), stub=signature_stub(problem)
    )


def as_code_block(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(completion: str) -> str:
    """Pull the runnable Python out of a model response.

    Small instruct models wrap code in a markdown fence and surround it with
    prose that does not parse. Everything here is deliberately literal: no
    repairing of broken code, no stripping of stray prose outside a fence.
    A completion that cannot be extracted is a completion that scores 0.0,
    and the reward ladder is supposed to see that.

    Handles the unterminated fence, which is common: a completion that hit
    the token limit mid-function has an opening ``` and no closing one.
    """
    matches = _FENCE.findall(completion)
    if not matches:
        return completion.strip()
    # Prefer the first block explicitly tagged as Python; otherwise the first
    # block at all. Later blocks are usually example usage or test output.
    for language, body in matches:
        if language.lower() in ("python", "py", "python3"):
            return body.strip()
    return matches[0][1].strip()


_STUB_ARG = re.compile(r"^arg\d+$")


def retains_stub_arg_names(code: str) -> bool:
    """True if the code keeps the prompt's placeholder parameter names.

    The signature stub says `def f(arg0, arg1):` and the model frequently
    copies those names straight through instead of naming its parameters. That
    is functionally harmless and stylistically poor, which makes it exactly the
    kind of degradation pass@1 cannot see: tracked so that a training run that
    increases it is caught (PROJECT.md §3b).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return any(
        _STUB_ARG.match(arg.arg)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for arg in node.args.args
    )
