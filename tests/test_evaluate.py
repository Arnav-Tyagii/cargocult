"""Tests for the pass@k harness.

The estimator gets the most attention here. Every number this project reports
passes through it, a bug in it is silent, and the wrong-but-plausible version
("did any of k samples pass") is the one people actually write — so it is
checked against exact combinatorics and against brute-force enumeration of
every k-subset, not just against itself.

Nothing here needs a GPU or a real model. Sampling itself lives in
generate.py and is covered by test_generate.py; what is tested here is
everything evaluate.py does around it.
"""

import itertools
import json
import math
from types import SimpleNamespace

import pytest
import torch

from src import evaluate as ev
from src.data import Problem
from src.evaluate import EvalReport, Generation, evaluate, pass_at_k
from src.reward import R_RUNS, R_TESTS_CEIL
from src.sandbox import ExecResult

SOLUTION = "def add(a, b):\n    return a + b"
WRONG = "def add(a, b):\n    return a - b"


def problem(task_id=511) -> Problem:
    return Problem(
        task_id=task_id,
        text="Add two numbers.",
        code=SOLUTION,
        test_list=["assert add(1, 2) == 3", "assert add(2, 2) == 4"],
    )


def generation(text: str, n_tokens: int = 20, hit_token_limit: bool = False):
    return Generation(text=f"```python\n{text}\n```", n_tokens=n_tokens,
                      hit_token_limit=hit_token_limit)


class FakeTokenizer:
    """No chat template, so format_prompt takes the plain-text path."""

    chat_template = None


class FakeModel(torch.nn.Module):
    def __init__(self, name="fake/model"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4))
        self.config = SimpleNamespace(name_or_path=name)


# --- the estimator -----------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 13))
def test_matches_the_exact_combinatorial_identity(n):
    """1 - C(n-c, k) / C(n, k), computed the slow obvious way."""
    for c in range(n + 1):
        for k in range(1, n + 1):
            expected = 1.0 - (
                math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0.0
            )
            assert pass_at_k(n, c, k) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("n,c,k", [(8, 3, 2), (6, 2, 3), (5, 1, 4), (4, 2, 2)])
def test_matches_brute_force_over_every_k_subset(n, c, k):
    correct = set(range(c))
    subsets = list(itertools.combinations(range(n), k))
    fraction_with_a_pass = sum(1 for s in subsets if correct & set(s)) / len(subsets)
    assert pass_at_k(n, c, k) == pytest.approx(fraction_with_a_pass)


def test_is_not_the_biased_any_of_k_estimator():
    # One lucky sample out of eight is pass@1 = 0.125, not 1.0. This is the
    # entire reason the combinatorial form exists.
    assert pass_at_k(8, 1, 1) == pytest.approx(0.125)
    assert pass_at_k(8, 1, 8) == 1.0


@pytest.mark.parametrize("n,c,k,expected", [(4, 0, 1, 0.0), (4, 4, 1, 1.0), (4, 4, 4, 1.0)])
def test_estimator_endpoints(n, c, k, expected):
    assert pass_at_k(n, c, k) == expected


def test_estimator_is_monotonic_in_c():
    values = [pass_at_k(8, c, 2) for c in range(9)]
    assert values == sorted(values)


@pytest.mark.parametrize("args", [(4, 0, 8), (4, 0, 0), (4, 5, 1), (4, -1, 1)])
def test_estimator_rejects_impossible_arguments(args):
    with pytest.raises(ValueError):
        pass_at_k(*args)


# --- tiers -------------------------------------------------------------------


def test_tier_specs_match_the_plan():
    assert ev.TIERS["dev"].split == "dev"
    assert (ev.TIERS["dev"].n_problems, ev.TIERS["dev"].n_samples) == (90, 4)
    assert ev.TIERS["full"].split == "test"
    assert (ev.TIERS["full"].n_problems, ev.TIERS["full"].n_samples) == (200, 8)


def test_dev_tier_never_touches_test_problems():
    """The one mistake that cannot be undone later."""
    try:
        dev = ev.tier_problems("dev")
        full = ev.tier_problems("full")
    except Exception as exc:
        pytest.skip(f"MBPP unavailable: {exc}")
    assert {p.task_id for p in dev}.isdisjoint({p.task_id for p in full})
    assert all(511 <= p.task_id <= 600 for p in dev)
    assert all(11 <= p.task_id <= 510 for p in full)


def test_tier_problem_sets_are_stable_across_calls():
    try:
        first = [p.task_id for p in ev.tier_problems("full")]
        second = [p.task_id for p in ev.tier_problems("full")]
    except Exception as exc:
        pytest.skip(f"MBPP unavailable: {exc}")
    assert first == second and len(first) == 200


# --- checkpoint identity -----------------------------------------------------


def test_checkpoint_hash_is_stable():
    model = FakeModel()
    assert ev.checkpoint_hash(model) == ev.checkpoint_hash(model)


def test_checkpoint_hash_changes_when_a_weight_changes():
    model = FakeModel()
    before = ev.checkpoint_hash(model)
    with torch.no_grad():
        model.weight[0] += 1e-3
    assert ev.checkpoint_hash(model) != before


def test_checkpoint_hash_changes_with_the_base_model():
    assert ev.checkpoint_hash(FakeModel("a")) != ev.checkpoint_hash(FakeModel("b"))


def test_checkpoint_hash_survives_a_fully_frozen_model():
    model = FakeModel()
    unfrozen = ev.checkpoint_hash(model)
    model.requires_grad_(False)
    # Freezing for inference must not collapse every checkpoint to one key.
    assert ev.checkpoint_hash(model) == unfrozen


# --- the eval cache's view of a completion -----------------------------------


def test_generation_projects_a_completion():
    """Token accounting itself is generate.py's; this is only the projection."""
    from src.generate import Completion

    gen = ev.Generation.of(
        Completion(text="code", token_ids=[5, 6, 7], logprobs=[-0.1, -0.2, -0.3],
                   hit_token_limit=False)
    )
    assert gen == ev.Generation(text="code", n_tokens=3, hit_token_limit=False)


# --- the generation cache ----------------------------------------------------


@pytest.fixture
def cached(tmp_path):
    """A written-out cache plus everything needed to read it back."""
    problems = [problem(511), problem(512)]
    prompts = ["prompt a", "prompt b"]
    generations = [[generation(SOLUTION)], [generation(WRONG)]]
    path = tmp_path / "cache.json"
    ev._save_cache(path, problems, prompts, generations,
                   meta={"n_samples": 1, "n_truncated_prompts": 3})
    return SimpleNamespace(path=path, problems=problems, prompts=prompts)


def test_cache_round_trips(cached):
    loaded, meta = ev._load_cache(cached.path, cached.problems, cached.prompts, 1)
    assert loaded[0][0].text == f"```python\n{SOLUTION}\n```"
    assert meta["n_truncated_prompts"] == 3


def test_a_rescore_still_reports_what_the_generations_cost(tmp_path):
    """Re-scoring must not rewrite a run's report to claim it was free.

    Observed for real: rerunning the baseline command to check reproducibility
    overwrote the committed report with 0 min generate, 0 MB vram.
    """
    problems = [problem(511)]
    tokenizer, model = FakeTokenizer(), FakeModel()
    prompts = [ev.format_prompt(p, tokenizer) for p in problems]
    path = ev._cache_path(tmp_path, ev.checkpoint_hash(model), "dev", 0, 0.8, 384)
    ev._save_cache(path, problems, prompts, [[generation(SOLUTION)]],
                   meta={"n_samples": 1, "generation_seconds": 739.3,
                         "peak_vram_mb": 2115.2, "n_truncated_prompts": 5})

    report = evaluate(model, tokenizer, problems, k=1, n_samples=1,
                      temperature=0.8, tier="dev", cache_dir=tmp_path, n_workers=1)

    assert report.cache_hit
    assert report.generation_seconds == 739.3
    assert report.peak_vram_mb == 2115.2
    assert report.n_truncated_prompts == 5


def test_cache_misses_when_the_sampling_knobs_are_not_recorded(tmp_path):
    """An old cache predates top_k/repetition_penalty being pinned.

    Those do not appear in the filename but they change what was sampled, so
    reusing such a cache would score one distribution and report another.
    """
    problems, prompts = [problem(511)], ["p"]
    path = tmp_path / "old.json"
    ev._save_cache(path, problems, prompts, [[generation(SOLUTION)]],
                   meta={"n_samples": 1})
    blob = json.loads(path.read_text(encoding="utf-8"))
    del blob["meta"]["top_k"]
    path.write_text(json.dumps(blob), encoding="utf-8")
    assert ev._load_cache(path, problems, prompts, 1) is None


def test_cache_misses_when_a_sampling_knob_differs(cached, monkeypatch):
    monkeypatch.setattr(ev, "SAMPLING_REPETITION_PENALTY", 1.1)
    assert ev._load_cache(cached.path, cached.problems, cached.prompts, 1) is None


def test_cache_misses_when_the_prompt_changes(cached):
    # A template edit must invalidate, or old generations get scored as new.
    assert ev._load_cache(cached.path, cached.problems, ["prompt a", "edited"], 1) is None


def test_cache_misses_when_the_problem_set_changes(cached):
    other = [problem(511), problem(999)]
    assert ev._load_cache(cached.path, other, cached.prompts, 1) is None


def test_cache_misses_when_more_samples_are_wanted(cached):
    assert ev._load_cache(cached.path, cached.problems, cached.prompts, 8) is None


def test_cache_hit_truncates_to_the_requested_samples(tmp_path):
    problems = [problem(511)]
    prompts = ["p"]
    path = tmp_path / "c.json"
    ev._save_cache(path, problems, prompts,
                   [[generation(SOLUTION), generation(WRONG), generation(WRONG)]],
                   meta={"n_samples": 3})
    loaded, _ = ev._load_cache(path, problems, prompts, 2)
    assert len(loaded[0]) == 2


def test_cache_misses_on_a_missing_file(tmp_path):
    assert ev._load_cache(tmp_path / "nope.json", [problem()], ["p"], 1) is None


def test_sampling_parameters_are_in_the_cache_filename():
    """Greedy and temperature-0.8 baselines share a checkpoint, tier and seed."""
    greedy = ev._cache_path("d", "abc", "full", 0, 0.0, 384)
    sampled = ev._cache_path("d", "abc", "full", 0, 0.8, 384)
    assert greedy != sampled
    for part in ("abc", "full", "seed0"):
        assert part in greedy.name


# --- end to end, from a seeded cache -----------------------------------------


def test_evaluate_scores_a_cached_run(tmp_path):
    """Real sandbox, real reward ladder, no GPU: 1 of 2 samples correct."""
    problems = [problem(511), problem(512)]
    tokenizer = FakeTokenizer()
    model = FakeModel()
    prompts = [ev.format_prompt(p, tokenizer) for p in problems]
    path = ev._cache_path(tmp_path, ev.checkpoint_hash(model), "dev", 0, 0.8, 384)
    ev._save_cache(
        path, problems, prompts,
        [[generation(SOLUTION), generation(WRONG)]] * 2,
        meta={"n_samples": 2},
    )

    report = evaluate(model, tokenizer, problems, k=1, n_samples=2,
                      temperature=0.8, tier="dev", cache_dir=tmp_path, n_workers=2)

    assert report.cache_hit and report.generation_seconds == 0.0
    assert report.pass_at_k == 0.5  # one of two samples passes, per problem
    assert report.pass_at_k_all["2"] == 1.0
    assert report.solved_any == 1.0
    assert report.outcome_counts == {"pass": 2, "wrong": 2}
    # The wrong solution runs but passes 0 of 2 asserts, so it sits on the
    # "ran" rung of the ladder rather than anywhere in the test band.
    assert report.mean_reward == pytest.approx((R_TESTS_CEIL + R_RUNS) / 2)
    assert [p.task_id for p in report.problems] == [511, 512]
    assert report.problems[0].n_correct == 1


def test_report_json_round_trip(tmp_path):
    problems = [problem(511)]
    tokenizer, model = FakeTokenizer(), FakeModel()
    prompts = [ev.format_prompt(p, tokenizer) for p in problems]
    path = ev._cache_path(tmp_path, ev.checkpoint_hash(model), "dev", 0, 0.8, 384)
    ev._save_cache(path, problems, prompts, [[generation(SOLUTION)]],
                   meta={"n_samples": 1})
    report = evaluate(model, tokenizer, problems, k=1, n_samples=1,
                      temperature=0.8, tier="dev", cache_dir=tmp_path, n_workers=1)

    out = report.save(tmp_path / "runs" / "eval.json")
    assert EvalReport.load(out) == report

    # Per-problem breakdown, not just the headline number.
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["problems"][0]["task_id"] == 511
    assert blob["problems"][0]["outcomes"] == ["pass"]
    assert "pass_at_k" in blob and "outcome_counts" in blob


# --- aggregation edge cases --------------------------------------------------


def build_report(results, **kwargs):
    """_build_report over synthetic ExecResults, bypassing the sandbox."""
    problems = [problem(511 + i) for i in range(len(results))]
    generations = [[generation("x") for _ in row] for row in results]
    defaults = dict(
        tier="dev", k=1, n_samples=len(results[0]), temperature=0.8, top_p=0.95,
        seed=0, max_new_tokens=384, model_name="m", checkpoint="h", n_truncated=0,
        generation_seconds=0.0, execution_seconds=0.0, peak_vram_mb=0.0,
        cache_path="", cache_hit=True,
    )
    return ev._build_report(problems=problems, generations=generations,
                            exec_results=results, **{**defaults, **kwargs})


def result(n_tests=2, n_passed=2, parsed=True, ran=True, timed_out=False):
    return ExecResult(parsed=parsed, ran=ran, n_tests=n_tests, n_passed=n_passed,
                      timed_out=timed_out, stderr_tail="", wall_time=0.1)


def test_a_problem_with_no_tests_is_never_counted_as_solved():
    # Partial credit is a fraction of asserts; zero asserts is not a pass.
    report = build_report([[result(n_tests=0, n_passed=0)]])
    assert report.problems[0].n_correct == 0
    assert report.pass_at_k == 0.0


def test_partial_passes_do_not_count_as_correct():
    report = build_report([[result(n_tests=3, n_passed=2)]])
    assert report.problems[0].n_correct == 0
    assert report.outcome_counts == {"partial": 1}


def test_solved_any_is_at_least_pass_at_k():
    """The biased number is reported next to the unbiased one as a check."""
    report = build_report([[result(), result(n_passed=0), result(n_passed=0),
                            result(n_passed=0)]], k=1, n_samples=4)
    assert report.pass_at_k == pytest.approx(0.25)
    assert report.solved_any == 1.0


def test_failure_rates_are_tracked():
    report = build_report([[result(parsed=False, ran=False, n_passed=0),
                            result(ran=False, n_passed=0, timed_out=True)]],
                          k=1, n_samples=2)
    assert report.unparseable_rate == 0.5
    assert report.timeout_rate == 0.5
    assert report.outcome_counts == {"timeout": 1, "unparseable": 1}
