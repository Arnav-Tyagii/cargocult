"""pass@k evaluation harness.

Every eval is 400-1,600 fresh generations. That was expected to make
evaluation the expensive half of the project; measured, a full tier is ~14
minutes (PROJECT.md §7), so the tiers below now buy iteration latency rather
than GPU budget. Three things drive this module's design.

**The estimator is unbiased.** pass@k is *not* "did any of k samples pass".
Drawing n samples and reporting the fraction of problems where any of the
first k passed is biased upward and the bias grows with n. The combinatorial
estimator below is the HumanEval one: given c correct out of n, the expected
value over a random k-subset.

**Two tiers.** `dev` (90 problems x 4 samples, ~3 min) for every iteration,
`full` (200 x 8, ~14 min) for final candidates only. Three minutes beats
fourteen while a knob is being turned; the full tier stays the only honest
number for a candidate.

**Generations are cached.** Keyed by (checkpoint_hash, tier, seed) — plus the
sampling parameters, which have to be in the key or greedy and temperature-0.8
baselines on the same checkpoint would overwrite each other. Re-scoring after
a change to the reward ladder or the sandbox then costs CPU only, no GPU.

WINDOWS NOTE
------------
Execution runs in a ProcessPoolExecutor, which uses spawn on Windows and
re-imports the caller's __main__ module in every worker. Guard entrypoints
with `if __name__ == "__main__":` or the whole script re-runs per worker.
Note that the guard stops the *body* from re-running but not module-level
imports: a script that does `import torch` at the top pays that import in
every worker (~6s each, once). Keeping heavy imports inside main() avoids it.
Against a 20-90 minute eval it is noise either way.

FROZEN: see PROJECT.md §8. The positional signature of `evaluate` is fixed so
that the GRPO extension scores checkpoints through the same harness; the
extra knobs are keyword-only with defaults, which leaves existing call sites
valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm

from src.data import (
    Problem,
    extract_code,
    format_prompt,
    load_problems,
    retains_stub_arg_names,
    subsample,
)
from src.generate import MAX_SEQ, Completion, sample_completions
from src.reward import compute_reward, reward_tier
from src.sandbox import DEFAULT_TIMEOUT, ExecResult, run_tests

DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 0

# What generate.sample_completions pins, restated here so the report can record
# the distribution it actually sampled from rather than the one it meant to.
SAMPLING_TOP_K = None
SAMPLING_REPETITION_PENALTY = 1.0

# Which problems a tier evaluates is fixed by this seed, not by the sampling
# seed: two checkpoints must be scored on the same problems to be comparable.
SUBSET_SEED = 0

CACHE_DIR = Path("runs/cache/generations")


@dataclass(frozen=True)
class Tier:
    name: str
    split: str
    n_problems: int
    n_samples: int


TIERS: dict[str, Tier] = {
    # The dev split is 90 problems, not the round 100 in the plan; it is used
    # whole rather than padded from elsewhere, because the alternative is
    # taking iteration decisions on test problems.
    "dev": Tier("dev", split="dev", n_problems=90, n_samples=4),
    "full": Tier("full", split="test", n_problems=200, n_samples=8),
}


def tier_problems(tier: str) -> list[Problem]:
    """The problem set for a tier. Deterministic, and identical across runs."""
    spec = TIERS[tier]
    return subsample(load_problems(spec.split), spec.n_problems, SUBSET_SEED)


# --- the estimator -----------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: 1 - C(n-c, k) / C(n, k).

    The probability that a uniformly random k-subset of the n samples contains
    no correct sample, subtracted from 1. Computed as a running product rather
    than with binomials so it cannot overflow and stays exact for small n.

        n: samples drawn, c: how many were correct, k: budget being reported.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n < k:
        raise ValueError(f"cannot estimate pass@{k} from {n} samples")
    if c < 0 or c > n:
        raise ValueError(f"c={c} out of range for n={n}")
    if n - c < k:
        return 1.0  # fewer than k failures exist, so every k-subset has a pass
    none_correct = 1.0
    for i in range(n - c + 1, n + 1):
        none_correct *= 1.0 - k / i
    return 1.0 - none_correct


# --- checkpoint identity -----------------------------------------------------

# Above this many trainable parameters, hashing every byte costs more than the
# eval saves. A LoRA adapter (~10M) is hashed exactly; a full base model is
# fingerprinted from a strided sample instead.
MAX_EXACT_HASH_PARAMS = 50_000_000


def checkpoint_hash(model) -> str:
    """Identity of the weights being evaluated, for the generation cache key.

    Exact for adapters, which is the case that matters: every checkpoint in
    the sweep differs only in its LoRA weights, and a stale cache hit there
    would mean silently reporting one checkpoint's numbers for another.

    For a model with more trainable parameters than MAX_EXACT_HASH_PARAMS the
    hash falls back to a strided sample plus the model identity. That is
    probabilistic — it would catch a merged or retrained base model with very
    high probability, but it is not a proof of difference.
    """
    digest = hashlib.blake2b(digest_size=8)
    config = getattr(model, "config", None)
    digest.update(str(getattr(config, "name_or_path", "")).encode())
    digest.update(type(model).__name__.encode())

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not params:  # everything frozen for inference; fall back to all weights
        params = list(model.named_parameters())
    params.sort(key=lambda item: item[0])

    exact = sum(p.numel() for _, p in params) <= MAX_EXACT_HASH_PARAMS
    digest.update(b"exact" if exact else b"sampled")
    for name, param in params:
        digest.update(name.encode())
        flat = param.detach().flatten()
        if not exact:
            stride = max(1, flat.numel() // 64)
            flat = flat[::stride][:64]
        digest.update(flat.float().cpu().numpy().tobytes())
    return digest.hexdigest()


# --- generation --------------------------------------------------------------


@dataclass
class Generation:
    """What the eval cache stores per sample.

    Thinner than `generate.Completion` on purpose. Sampling produces token ids
    and per-token logprobs because GRPO will need them (§8); evaluation reads
    neither, and keeping them would multiply every eval cache on disk to carry
    data nothing here opens. Generation happens in exactly one place —
    `generate.sample_completions` — and this is only the projection of its
    output that pass@k needs.
    """

    text: str
    n_tokens: int
    hit_token_limit: bool

    @classmethod
    def of(cls, completion: Completion) -> "Generation":
        return cls(
            text=completion.text,
            n_tokens=len(completion.token_ids),
            hit_token_limit=completion.hit_token_limit,
        )


def _count_truncated(tokenizer, prompts: Sequence[str], max_new_tokens: int) -> int:
    """Prompts that will not fit and get cut from the left.

    sample_completions warns about these; the count is recomputed here because
    the report records it and a warning is not a return value.
    """
    budget = MAX_SEQ - max_new_tokens
    return sum(1 for p in prompts if len(tokenizer(p).input_ids) > budget)


# --- generation cache --------------------------------------------------------


def _cache_path(
    cache_dir: Path,
    checkpoint: str,
    tier: str,
    seed: int,
    temperature: float,
    max_new_tokens: int,
) -> Path:
    name = (
        f"{checkpoint}_{tier}_seed{seed}"
        f"_temp{temperature:g}_max{max_new_tokens}.json"
    )
    return Path(cache_dir) / name


def _prompt_hash(prompts: Sequence[str]) -> str:
    """Covers both the prompt template and which problems are in the tier."""
    digest = hashlib.blake2b(digest_size=8)
    for prompt in prompts:
        digest.update(prompt.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_cache(
    path: Path, problems: Sequence[Problem], prompts: Sequence[str], n_samples: int
) -> tuple[list[list[Generation]], dict] | None:
    """Cached generations and their meta, or None if the key has moved."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    meta = blob.get("meta", {})
    if meta.get("prompt_hash") != _prompt_hash(prompts):
        return None  # template edited, or the problem set changed
    # Not in the filename, but they change what was sampled, so a cache
    # written under different values is a different dataset. Absent means an
    # older cache from before these were pinned: treat as a miss and redo it.
    if meta.get("top_k", "absent") != SAMPLING_TOP_K:
        return None
    if meta.get("repetition_penalty", "absent") != SAMPLING_REPETITION_PENALTY:
        return None
    if meta.get("n_samples", 0) < n_samples:
        return None
    records = blob.get("problems", [])
    if [r["task_id"] for r in records] != [p.task_id for p in problems]:
        return None
    # A cache holding more samples than asked for is still a hit: the extras
    # are iid draws and the first n are a valid sample.
    generations = [
        [Generation(**gen) for gen in record["completions"][:n_samples]]
        for record in records
    ]
    return generations, meta


def _save_cache(path: Path, problems, prompts, generations, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "meta": {
            **meta,
            "prompt_hash": _prompt_hash(prompts),
            "top_k": SAMPLING_TOP_K,
            "repetition_penalty": SAMPLING_REPETITION_PENALTY,
        },
        "problems": [
            {
                "task_id": problem.task_id,
                "prompt": prompt,
                "completions": [asdict(gen) for gen in gens],
            }
            for problem, prompt, gens in zip(problems, prompts, generations)
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob), encoding="utf-8")
    os.replace(tmp, path)


# --- execution ---------------------------------------------------------------


def _execute(
    problems: Sequence[Problem],
    generations: Sequence[Sequence[Generation]],
    timeout: float,
    n_workers: int,
) -> list[list[ExecResult]]:
    """Score every completion in the sandbox, in parallel across processes."""
    results: list[list[ExecResult | None]] = [[None] * len(g) for g in generations]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        pending = {}
        for problem_index, (problem, gens) in enumerate(zip(problems, generations)):
            for sample_index, gen in enumerate(gens):
                future = pool.submit(
                    run_tests,
                    extract_code(gen.text),
                    problem.test_list,
                    setup_code=problem.test_setup_code,
                    timeout=timeout,
                )
                pending[future] = (problem_index, sample_index)
        for future in tqdm(
            as_completed(pending), total=len(pending), desc="execute", unit="sample"
        ):
            problem_index, sample_index = pending[future]
            results[problem_index][sample_index] = future.result()
    return results  # type: ignore[return-value]


# --- report ------------------------------------------------------------------


@dataclass
class ProblemReport:
    task_id: int
    n_samples: int
    n_correct: int
    pass_at_k: float
    mean_reward: float
    best_reward: float
    rewards: list[float]
    outcomes: list[str]  # reward_tier per sample, for the failure taxonomy
    n_tokens: list[int]


@dataclass
class EvalReport:
    tier: str
    k: int
    n_problems: int
    n_samples: int
    temperature: float
    top_p: float
    seed: int
    max_new_tokens: int
    model_name: str
    checkpoint_hash: str
    # Recorded because they were once wrong and nothing said so: a checkpoint's
    # own generation_config used to leak top_k and repetition_penalty into
    # sampling, so a report naming only temperature and top_p was describing a
    # distribution it had not sampled from. generate.py fixes these; the report
    # states them so any future drift is visible in the artifact.
    top_k: int | None
    repetition_penalty: float

    pass_at_k: float
    pass_at_k_all: dict[str, float]
    mean_reward: float
    solved_any: float

    mean_completion_tokens: float
    token_limit_rate: float
    timeout_rate: float
    unparseable_rate: float
    # Share of completions that copy the prompt's placeholder parameter names
    # into the answer. Tracked per checkpoint because it is degradation pass@1
    # is blind to: a run can hold its score while its output gets worse to read.
    stub_args_rate: float
    outcome_counts: dict[str, int]
    n_truncated_prompts: int

    generation_seconds: float
    execution_seconds: float
    peak_vram_mb: float
    cache_path: str
    cache_hit: bool
    created: str
    notes: str = ""
    problems: list[ProblemReport] = field(default_factory=list)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "EvalReport":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        problems = [ProblemReport(**p) for p in blob.pop("problems", [])]
        return cls(**blob, problems=problems)

    def summary(self) -> str:
        # ASCII only: this line gets redirected into log files on a Windows
        # console that is not always UTF-8.
        return (
            f"{self.tier} tier | pass@{self.k}={self.pass_at_k:.3f} | "
            f"mean reward={self.mean_reward:.3f} | "
            f"{self.n_problems} problems x {self.n_samples} samples | "
            f"{self.mean_completion_tokens:.0f} tokens/completion"
        )


# --- entry point -------------------------------------------------------------


def evaluate(
    model,
    tokenizer,
    problems: Sequence[Problem],
    k: int,
    n_samples: int,
    temperature: float,
    tier: str,
    *,
    seed: int = DEFAULT_SEED,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 8,
    fewshot: Sequence[Problem] = (),
    cache_dir=CACHE_DIR,
    n_workers: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    notes: str = "",
) -> EvalReport:
    """Sample n_samples completions per problem, execute them, report pass@k.

    FROZEN: see PROJECT.md §8 — the positional arguments are the extension's
    call signature. Extra knobs are keyword-only with defaults.

    Args:
        problems: what to evaluate. `tier_problems(tier)` builds the standard
            set; passing a different list is allowed and gets recorded.
        k: the k in pass@k for the headline number. Must be <= n_samples.
        temperature: 0 means greedy, in which case n_samples > 1 buys nothing.
        tier: "dev" or "full". Names the run and keys the cache; it does not
            override `problems` or `n_samples`.
        batch_size: sequences per forward pass, not prompts — VRAM scales with
            this, and each prompt expands to n_samples sequences.

    Returns:
        EvalReport, also serialisable to JSON with the per-problem breakdown.
    """
    problems = list(problems)
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
    if k > n_samples:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n_samples}")
    if temperature == 0 and n_samples > 1:
        raise ValueError("greedy decoding with n_samples > 1 samples the same text")

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) // 2)

    prompts = [format_prompt(p, tokenizer, fewshot) for p in problems]
    checkpoint = checkpoint_hash(model)
    path = _cache_path(Path(cache_dir), checkpoint, tier, seed, temperature, max_new_tokens)

    cached = _load_cache(path, problems, prompts, n_samples)
    cache_hit = cached is not None
    generation_seconds = 0.0
    peak_vram_mb = 0.0
    n_truncated = 0
    generations: list[list[Generation]] = []
    if cached is not None:
        # Everything below describes the generations, not the pass that is
        # scoring them, so it is carried forward from the run that produced
        # them. Otherwise re-scoring a cached run rewrites its own report to
        # claim it cost no GPU time, and §5's compute-cost comparison is lost.
        generations, meta = cached
        n_truncated = meta.get("n_truncated_prompts", 0)
        generation_seconds = meta.get("generation_seconds", 0.0)
        peak_vram_mb = meta.get("peak_vram_mb", 0.0)

    if not cache_hit:
        on_cuda = torch.cuda.is_available()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        n_truncated = _count_truncated(tokenizer, prompts, max_new_tokens)
        generations = [
            [Generation.of(c) for c in row]
            for row in sample_completions(
                model, tokenizer, prompts, n_samples, temperature, top_p,
                max_new_tokens, batch_size=batch_size, seed=seed,
            )
        ]
        generation_seconds = time.perf_counter() - started
        if on_cuda:
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        _save_cache(
            path, problems, prompts, generations,
            meta={
                "checkpoint_hash": checkpoint,
                "tier": tier,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "n_samples": n_samples,
                "max_new_tokens": max_new_tokens,
                "model_name": str(getattr(getattr(model, "config", None), "name_or_path", "")),
                "n_truncated_prompts": n_truncated,
                "generation_seconds": generation_seconds,
                "peak_vram_mb": peak_vram_mb,
                "created": _now(),
            },
        )

    started = time.perf_counter()
    exec_results = _execute(problems, generations, timeout, n_workers)
    execution_seconds = time.perf_counter() - started

    return _build_report(
        problems=problems,
        generations=generations,
        exec_results=exec_results,
        tier=tier,
        k=k,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_new_tokens=max_new_tokens,
        model_name=str(getattr(getattr(model, "config", None), "name_or_path", "")),
        checkpoint=checkpoint,
        n_truncated=n_truncated,
        generation_seconds=generation_seconds,
        execution_seconds=execution_seconds,
        peak_vram_mb=peak_vram_mb,
        cache_path=str(path),
        cache_hit=cache_hit,
        notes=notes,
    )


def _build_report(
    *,
    problems,
    generations,
    exec_results,
    tier,
    k,
    n_samples,
    temperature,
    top_p,
    seed,
    max_new_tokens,
    model_name,
    checkpoint,
    n_truncated,
    generation_seconds,
    execution_seconds,
    peak_vram_mb,
    cache_path,
    cache_hit,
    notes="",
) -> EvalReport:
    ks = sorted({k} | {kk for kk in (1, 2, 4, 8, 16) if kk <= n_samples})
    per_problem: list[ProblemReport] = []
    outcome_counts: dict[str, int] = {}
    all_rewards: list[float] = []
    all_tokens: list[int] = []
    n_token_limited = n_timed_out = n_unparseable = n_stub_args = 0

    for problem, gens, results in zip(problems, generations, exec_results):
        rewards = [compute_reward(r, g.hit_token_limit) for r, g in zip(results, gens)]
        outcomes = [reward_tier(r) for r in results]
        # "Correct" is all asserts passing. A problem shipping zero tests
        # cannot be solved, only run, and must not count as a pass.
        n_correct = sum(1 for r in results if r.n_tests > 0 and r.n_passed == r.n_tests)

        for outcome in outcomes:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        all_rewards.extend(rewards)
        all_tokens.extend(g.n_tokens for g in gens)
        n_token_limited += sum(1 for g in gens if g.hit_token_limit)
        n_timed_out += sum(1 for r in results if r.timed_out)
        n_unparseable += sum(1 for r in results if not r.parsed)
        n_stub_args += sum(1 for g in gens if retains_stub_arg_names(extract_code(g.text)))

        per_problem.append(
            ProblemReport(
                task_id=problem.task_id,
                n_samples=len(gens),
                n_correct=n_correct,
                pass_at_k=pass_at_k(len(gens), n_correct, k),
                mean_reward=_mean(rewards),
                best_reward=max(rewards, default=0.0),
                rewards=[round(r, 4) for r in rewards],
                outcomes=outcomes,
                n_tokens=[g.n_tokens for g in gens],
            )
        )

    n_total = max(1, len(all_rewards))
    return EvalReport(
        tier=tier,
        k=k,
        n_problems=len(problems),
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
        checkpoint_hash=checkpoint,
        top_k=SAMPLING_TOP_K,
        repetition_penalty=SAMPLING_REPETITION_PENALTY,
        pass_at_k=_mean([p.pass_at_k for p in per_problem]),
        pass_at_k_all={
            str(kk): _mean(
                [pass_at_k(p.n_samples, p.n_correct, kk) for p in per_problem]
            )
            for kk in ks
        },
        mean_reward=_mean(all_rewards),
        # Reported next to pass@k as a sanity check, never instead of it: this
        # is the biased "any of n passed" number, and it should sit above
        # pass@k for k < n. If it does not, something is wrong upstream.
        solved_any=_mean([1.0 if p.n_correct > 0 else 0.0 for p in per_problem]),
        mean_completion_tokens=_mean([float(t) for t in all_tokens]),
        token_limit_rate=n_token_limited / n_total,
        timeout_rate=n_timed_out / n_total,
        unparseable_rate=n_unparseable / n_total,
        stub_args_rate=n_stub_args / n_total,
        outcome_counts=dict(sorted(outcome_counts.items())),
        n_truncated_prompts=n_truncated,
        generation_seconds=generation_seconds,
        execution_seconds=execution_seconds,
        peak_vram_mb=peak_vram_mb,
        cache_path=cache_path,
        cache_hit=cache_hit,
        created=_now(),
        notes=notes,
        problems=per_problem,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
