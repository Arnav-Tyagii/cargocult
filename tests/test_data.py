"""Tests for MBPP loading, the task_id splits, and prompt formatting.

The split tests are the important ones. Everything else in this project is
recoverable; evaluating on problems that were trained on is not, because the
damage is invisible in the numbers.
"""

import pytest

from src import data
from src.data import Problem

# The split from PROJECT.md §2, repeated here as a literal so that changing
# src.data alone cannot change what the project means by "test".
SPEC_SPLITS = {
    "fewshot": (1, 10),
    "test": (11, 510),
    "dev": (511, 600),
    "train": (601, 974),
}


@pytest.fixture(scope="module")
def dev_problems():
    try:
        return data.load_problems("dev")
    except Exception as exc:  # no network and nothing cached
        pytest.skip(f"MBPP unavailable: {exc}")


def problem(**kwargs) -> Problem:
    base = dict(
        task_id=511,
        text="Write a function that adds two numbers.",
        code="def add(a, b):\n    return a + b",
        test_list=["assert add(1, 2) == 3"],
    )
    return Problem(**{**base, **kwargs})


# --- splits ------------------------------------------------------------------


def test_split_ranges_match_the_spec():
    assert data.SPLITS == SPEC_SPLITS


def test_splits_are_a_partition_of_1_to_974():
    seen = set()
    for lo, hi in data.SPLITS.values():
        ids = set(range(lo, hi + 1))
        assert not (seen & ids), "splits overlap"
        seen |= ids
    assert seen == set(range(1, 975))


def test_the_partition_check_actually_catches_an_overlap(monkeypatch):
    """The guard has to fail loudly, not pass vacuously."""
    monkeypatch.setattr(data, "SPLITS", {"a": (1, 5), "b": (4, 8)})
    monkeypatch.setattr(data, "EXPECTED_COUNTS", {"a": 5, "b": 5})
    with pytest.raises(AssertionError, match="overlap"):
        data._assert_splits_are_a_partition()


def test_the_partition_check_catches_a_gap(monkeypatch):
    monkeypatch.setattr(data, "SPLITS", {"a": (1, 5), "b": (7, 11)})
    monkeypatch.setattr(data, "EXPECTED_COUNTS", {"a": 5, "b": 5})
    with pytest.raises(AssertionError, match="cover"):
        data._assert_splits_are_a_partition()


@pytest.mark.parametrize(
    "task_id,expected",
    [
        (1, "fewshot"), (10, "fewshot"),
        (11, "test"), (510, "test"),
        (511, "dev"), (600, "dev"),
        (601, "train"), (974, "train"),
    ],
)
def test_split_boundaries(task_id, expected):
    assert data.split_of(task_id) == expected


@pytest.mark.parametrize("task_id", [0, 975, -1])
def test_task_ids_outside_mbpp_raise(task_id):
    with pytest.raises(KeyError):
        data.split_of(task_id)


def test_unknown_split_name_raises():
    with pytest.raises(KeyError):
        data.load_problems("validation")  # MBPP's name for it, not ours


# --- loading -----------------------------------------------------------------


def test_dev_split_loads_the_expected_problems(dev_problems):
    assert len(dev_problems) == 90
    assert [p.task_id for p in dev_problems] == list(range(511, 601))
    assert all(p.split == "dev" for p in dev_problems)


def test_every_problem_has_asserts(dev_problems):
    # The reward ladder divides by n_tests; a problem with none would score at
    # the "ran" rung forever and never be solvable.
    assert all(len(p.test_list) >= 1 for p in dev_problems)


def test_reference_solutions_pass_their_own_tests(dev_problems):
    """A sanity check on the loader, the sandbox and MBPP at once."""
    from src.sandbox import run_tests

    for candidate in dev_problems[:5]:
        result = run_tests(
            candidate.code, candidate.test_list, setup_code=candidate.test_setup_code
        )
        assert result.n_passed == result.n_tests, (
            f"task {candidate.task_id} reference solution failed: {result.stderr_tail}"
        )


# --- subsampling -------------------------------------------------------------


def test_subsample_is_deterministic_and_ordered():
    problems = [problem(task_id=i) for i in range(11, 511)]
    first = data.subsample(problems, 200, seed=0)
    second = data.subsample(problems, 200, seed=0)

    assert [p.task_id for p in first] == [p.task_id for p in second]
    assert len(first) == 200
    assert [p.task_id for p in first] == sorted(p.task_id for p in first)


def test_subsample_is_stable_under_input_order():
    """Two checkpoints must be scored on the same problems, whatever order
    the caller happened to hand them over in."""
    problems = [problem(task_id=i) for i in range(11, 511)]
    shuffled = list(reversed(problems))
    assert [p.task_id for p in data.subsample(problems, 50, 0)] == [
        p.task_id for p in data.subsample(shuffled, 50, 0)
    ]


def test_subsample_different_seeds_differ():
    problems = [problem(task_id=i) for i in range(11, 511)]
    assert [p.task_id for p in data.subsample(problems, 200, 0)] != [
        p.task_id for p in data.subsample(problems, 200, 1)
    ]


def test_subsample_returns_everything_when_n_is_too_large():
    problems = [problem(task_id=i) for i in range(11, 21)]
    assert len(data.subsample(problems, 999, 0)) == 10


# --- prompts -----------------------------------------------------------------


class PlainTokenizer:
    """A tokenizer with no chat template, like the 54.6M baseline model."""

    chat_template = None


class ChatTokenizer:
    chat_template = "exists"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        rendered = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return rendered + "<assistant>"


def test_prompt_contains_the_task_and_the_tests():
    # Without the asserts the model has to guess the function name, which is
    # not what we mean to be measuring.
    prompt = data.format_prompt(problem(), ChatTokenizer())
    assert "adds two numbers" in prompt
    assert "assert add(1, 2) == 3" in prompt


def test_chat_template_is_applied_with_a_generation_prompt():
    prompt = data.format_prompt(problem(), ChatTokenizer())
    assert prompt.endswith("<assistant>")
    assert prompt.startswith("<user>")


def test_plain_tokenizer_falls_back_to_concatenation():
    prompt = data.format_prompt(problem(), PlainTokenizer())
    assert "<user>" not in prompt
    assert "adds two numbers" in prompt


def test_fewshot_examples_become_alternating_turns():
    example = problem(task_id=1, text="Reverse a string.", code="def rev(s): return s[::-1]")
    prompt = data.format_prompt(problem(), ChatTokenizer(), fewshot=[example])
    assert prompt.count("<user>") == 2
    assert "<assistant>```python\ndef rev(s): return s[::-1]\n```</assistant>" in prompt


# --- extracting code out of a response ---------------------------------------


def test_extract_fenced_block():
    assert data.extract_code("Sure!\n```python\ndef f():\n    return 1\n```\nDone.") == (
        "def f():\n    return 1"
    )


def test_extract_prefers_the_python_tagged_block():
    completion = "```\n>>> f()\n1\n```\nand the code:\n```python\ndef f():\n    return 1\n```"
    assert data.extract_code(completion) == "def f():\n    return 1"


def test_extract_takes_the_first_block_when_none_are_tagged():
    assert data.extract_code("```\ndef f(): pass\n```\n```\ndef g(): pass\n```") == (
        "def f(): pass"
    )


def test_extract_handles_an_unterminated_fence():
    # What a completion that hit the token limit mid-function looks like.
    assert data.extract_code("```python\ndef f():\n    return 1") == (
        "def f():\n    return 1"
    )


def test_extract_leaves_unfenced_text_alone():
    # Prose in, prose out: it fails to parse and the reward ladder scores 0.0.
    assert data.extract_code("  I cannot solve this.  ") == "I cannot solve this."


def test_extract_does_not_repair_broken_code():
    assert data.extract_code("```python\ndef f(:\n```") == "def f(:"
