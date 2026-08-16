"""Build preference pairs from scored completions.

`chosen` is a completion that passes every assert. `rejected` is one that does
not. Everything interesting here is in what gets thrown away, so every filter
counts its drops and the counts go in `pairs_stats.json` — each one is a
finding for the writeup, not just hygiene (PROJECT.md §2c).

THE FOUR FILTERS
----------------
**Reward margin.** A pair whose two sides scored almost the same teaches
almost nothing. With `chosen` restricted to full passes the smallest real gap
is 1.00 against a 0.77 two-of-three partial, so the threshold is low by
design; it exists to catch the degenerate case, not to prune hard contrasts.

**Near-duplicate.** If chosen and rejected differ by a token or two, the
gradient is noise pointing in an arbitrary direction. Distance is computed
over token ids rather than characters — that is what the model actually
emitted, and a one-character rename should not read as a large edit.

**Length balance, symmetric — and an ablation, not the default.** DPO will
happily learn "longer is better" from a corpus where chosen is longer. The
measured skew here runs the other way: passing completions are ~77 tokens
*shorter*, which is the more dangerous direction, because a policy that learns
to truncate looks like it is improving until someone reads the output.

Balancing costs 475 of 1,356 pairs, and it spends them on exactly the most
informative contrasts — a terse correct answer against a rambling broken one.
So both sets are emitted and the default is the *unbalanced* one, following
§4's rule for the other pathologies: instrument, do not prevent. Training logs
mean completion length beside pass@1, so if the policy starts truncating it is
visible in the curve rather than assumed away in the data. The balanced set is
the ablation that answers whether it mattered.

**Cap per problem.** Six. An easy problem can otherwise contribute dozens of
pairs and drown out the problems the model is actually failing.

WHAT LIMITS THE PAIR COUNT
--------------------------
Not any of the filters: the base model's pass rate. A problem yields pairs
only if it produced at least one passing and one failing sample, and at 0.5B
scale a large minority of MBPP problems never pass at all. That ceiling is a
property of the policy, not of this file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence

from src.data import retains_stub_arg_names

DEFAULT_COMPLETIONS = Path("data/completions.jsonl")
DEFAULT_PAIRS = Path("data/pairs_train.jsonl")
DEFAULT_PAIRS_BALANCED = Path("data/pairs_train_balanced.jsonl")
DEFAULT_RFT = Path("data/rft_train.jsonl")
DEFAULT_STATS = Path("data/pairs_stats.json")

CAP_PER_PROBLEM = 6
MIN_REWARD_MARGIN = 0.10
MIN_EDIT_DISTANCE = 0.15
LENGTH_TOLERANCE_TOKENS = 5.0


@dataclass
class Pair:
    task_id: int
    prompt: str
    chosen: str
    rejected: str
    chosen_token_ids: list[int]
    rejected_token_ids: list[int]
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    chosen_tokens: int
    rejected_tokens: int
    edit_distance: float
    chosen_sample_index: int
    rejected_sample_index: int
    chosen_hit_token_limit: bool = False
    rejected_hit_token_limit: bool = False
    chosen_stub_args: bool = False
    rejected_stub_args: bool = False

    @property
    def length_delta(self) -> int:
        """Positive when chosen is the longer side. Sign is load-bearing."""
        return self.chosen_tokens - self.rejected_tokens


def solved(completion: dict) -> bool:
    """Every assert passed. A problem shipping no asserts cannot be solved."""
    result = completion["exec"]
    return result["n_tests"] > 0 and result["n_passed"] == result["n_tests"]


def load_completions(path: Path) -> dict[int, list[dict]]:
    by_problem: dict[int, list[dict]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_problem.setdefault(row["task_id"], []).append(row)
    for rows in by_problem.values():
        rows.sort(key=lambda r: r["sample_index"])
    return by_problem


def edit_distance(left: Sequence[int], right: Sequence[int]) -> float:
    """Normalised dissimilarity in [0, 1] over token ids.

    SequenceMatcher rather than true Levenshtein: it is stdlib, it is
    implemented in C, and at ~150 tokens a quadratic Python edit distance over
    every candidate pair would dominate the runtime of this whole phase. The
    ratio is a matching-block similarity, not an edit metric, so this is a
    proxy — good enough to separate "one token differs" from "different
    program", which is the only distinction the filter needs to make.
    """
    if not left and not right:
        return 0.0
    return 1.0 - SequenceMatcher(None, left, right, autojunk=False).ratio()


@dataclass
class Stats:
    n_problems: int = 0
    n_completions: int = 0
    n_problems_with_pass: int = 0
    n_problems_with_fail: int = 0
    n_problems_pairable: int = 0
    n_candidates: int = 0
    dropped_reward_margin: int = 0
    dropped_near_duplicate: int = 0
    dropped_by_cap: int = 0
    dropped_length_balance: int = 0
    n_pairs: int = 0
    n_pairs_balanced: int = 0
    n_problems_contributing: int = 0
    pairs_per_problem: dict[str, int] = field(default_factory=dict)
    length_skew_pre_balance: float = 0.0
    length_skew_post_balance: float = 0.0
    chosen_tokens_mean: float = 0.0
    rejected_tokens_mean: float = 0.0
    chosen_reward_mean: float = 0.0
    rejected_reward_mean: float = 0.0
    reward_margin_mean: float = 0.0
    edit_distance_mean: float = 0.0
    rejected_at_token_limit: int = 0
    chosen_at_token_limit: int = 0
    stub_args_rate_chosen: float = 0.0
    stub_args_rate_rejected: float = 0.0
    rft_examples: int = 0
    rft_problems: int = 0
    rft_duplicates_dropped: int = 0
    config: dict = field(default_factory=dict)


def build_pairs(
    by_problem: dict[int, list[dict]],
    prompts: dict[int, str],
    *,
    cap: int = CAP_PER_PROBLEM,
    min_reward_margin: float = MIN_REWARD_MARGIN,
    min_edit_distance: float = MIN_EDIT_DISTANCE,
    length_tolerance: float = LENGTH_TOLERANCE_TOKENS,
) -> tuple[list[Pair], list[Pair], Stats]:
    """Returns (unbalanced, balanced, stats). The unbalanced set is the default
    training corpus; the balanced one is §5's length ablation."""
    stats = Stats(
        n_problems=len(by_problem),
        n_completions=sum(len(v) for v in by_problem.values()),
        config={
            "cap_per_problem": cap,
            "min_reward_margin": min_reward_margin,
            "min_edit_distance": min_edit_distance,
            "length_tolerance_tokens": length_tolerance,
        },
    )

    pairs: list[Pair] = []
    for task_id, rows in sorted(by_problem.items()):
        passing = [r for r in rows if solved(r)]
        failing = [r for r in rows if not solved(r)]
        stats.n_problems_with_pass += bool(passing)
        stats.n_problems_with_fail += bool(failing)
        if not passing or not failing:
            continue
        stats.n_problems_pairable += 1

        candidates: list[Pair] = []
        for chosen in passing:
            for rejected in failing:
                stats.n_candidates += 1
                margin = chosen["reward"] - rejected["reward"]
                if margin < min_reward_margin:
                    stats.dropped_reward_margin += 1
                    continue
                distance = edit_distance(chosen["token_ids"], rejected["token_ids"])
                if distance < min_edit_distance:
                    stats.dropped_near_duplicate += 1
                    continue
                candidates.append(
                    Pair(
                        task_id=task_id,
                        prompt=prompts[task_id],
                        chosen=chosen["text"],
                        rejected=rejected["text"],
                        chosen_token_ids=chosen["token_ids"],
                        rejected_token_ids=rejected["token_ids"],
                        chosen_reward=chosen["reward"],
                        rejected_reward=rejected["reward"],
                        reward_margin=round(margin, 4),
                        chosen_tokens=len(chosen["token_ids"]),
                        rejected_tokens=len(rejected["token_ids"]),
                        edit_distance=round(distance, 4),
                        chosen_sample_index=chosen["sample_index"],
                        rejected_sample_index=rejected["sample_index"],
                        chosen_hit_token_limit=chosen["hit_token_limit"],
                        rejected_hit_token_limit=rejected["hit_token_limit"],
                        chosen_stub_args=retains_stub_arg_names(chosen["code"]),
                        rejected_stub_args=retains_stub_arg_names(rejected["code"]),
                    )
                )

        # Clearest contrasts first. Ties break on sample index so the selection
        # is reproducible rather than dependent on dict ordering.
        candidates.sort(
            key=lambda p: (
                -p.reward_margin,
                -p.edit_distance,
                p.chosen_sample_index,
                p.rejected_sample_index,
            )
        )
        if len(candidates) > cap:
            stats.dropped_by_cap += len(candidates) - cap
        pairs.extend(candidates[:cap])

    stats.length_skew_pre_balance = _mean([p.length_delta for p in pairs])
    balanced, dropped = balance_lengths(pairs, length_tolerance)
    stats.dropped_length_balance = dropped
    stats.length_skew_post_balance = _mean([p.length_delta for p in balanced])
    stats.n_pairs_balanced = len(balanced)

    stats.n_pairs = len(pairs)
    per_problem = Counter(p.task_id for p in pairs)
    stats.n_problems_contributing = len(per_problem)
    stats.pairs_per_problem = {
        str(k): v for k, v in sorted(Counter(per_problem.values()).items())
    }
    stats.chosen_tokens_mean = _mean([p.chosen_tokens for p in pairs])
    stats.rejected_tokens_mean = _mean([p.rejected_tokens for p in pairs])
    stats.chosen_reward_mean = _mean([p.chosen_reward for p in pairs])
    stats.rejected_reward_mean = _mean([p.rejected_reward for p in pairs])
    stats.reward_margin_mean = _mean([p.reward_margin for p in pairs])
    stats.edit_distance_mean = _mean([p.edit_distance for p in pairs])
    # A rejected sample cut off at the budget is a different kind of negative
    # from one that finished and was wrong: it teaches "do not ramble" as much
    # as "do not be incorrect", and it is the shape most likely to be driving
    # the length skew.
    stats.rejected_at_token_limit = sum(1 for p in pairs if p.rejected_hit_token_limit)
    stats.chosen_at_token_limit = sum(1 for p in pairs if p.chosen_hit_token_limit)
    # Stylistic degradation pass@1 cannot see: the model copying the prompt's
    # placeholder parameter names straight into its answer.
    stats.stub_args_rate_chosen = _mean([float(p.chosen_stub_args) for p in pairs])
    stats.stub_args_rate_rejected = _mean([float(p.rejected_stub_args) for p in pairs])
    return pairs, balanced, stats


def balance_lengths(pairs: list[Pair], tolerance: float) -> tuple[list[Pair], int]:
    """Drop the most skewed pairs until the mean length delta is near zero.

    Symmetric: it does not care which side is longer, only that the corpus
    does not systematically reward one. The cost is real and worth stating —
    the most extreme pairs are often the most informative ones (a short
    correct answer against a long broken one), so this trades signal for the
    absence of a length shortcut.
    """
    if not pairs:
        return pairs, 0
    remaining = sorted(pairs, key=lambda p: p.length_delta)
    dropped = 0
    while remaining and abs(_mean([p.length_delta for p in remaining])) > tolerance:
        if _mean([p.length_delta for p in remaining]) < 0:
            remaining.pop(0)  # chosen too short on average
        else:
            remaining.pop()  # chosen too long on average
        dropped += 1
    remaining.sort(key=lambda p: (p.task_id, p.chosen_sample_index, p.rejected_sample_index))
    return remaining, dropped


def build_rft(
    by_problem: dict[int, list[dict]], prompts: dict[int, str]
) -> tuple[list[dict], int]:
    """Passing completions only, for the rejection-sampling baseline.

    Exact duplicates within a problem are dropped: a model that emits the same
    solution five times would otherwise be trained on it five times, which is
    a reweighting nobody chose.
    """
    examples: list[dict] = []
    duplicates = 0
    for task_id, rows in sorted(by_problem.items()):
        seen: set[str] = set()
        for row in rows:
            if not solved(row):
                continue
            if row["text"] in seen:
                duplicates += 1
                continue
            seen.add(row["text"])
            examples.append({
                "task_id": task_id,
                "prompt": prompts[task_id],
                "completion": row["text"],
                "token_ids": row["token_ids"],
                "reward": row["reward"],
                "sample_index": row["sample_index"],
                # Needed by the trainers: length has to be reported separately
                # for normally-terminated completions, because a truncated one
                # is long for a reason that has nothing to do with the policy's
                # verbosity (PROJECT.md §3b).
                "hit_token_limit": row["hit_token_limit"],
            })
    return examples, duplicates


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def build_prompts(split: str, model: str) -> dict[int, str]:
    """Prompts are not stored per completion; they are re-derived here.

    Deterministic given the same tokenizer and template, and identical to what
    generation used — 16 copies of the same string per problem is not worth
    putting on disk.
    """
    from transformers import AutoTokenizer

    from src.data import format_prompt, load_problems

    tokenizer = AutoTokenizer.from_pretrained(model)
    return {p.task_id: format_prompt(p, tokenizer) for p in load_problems(split)}


def format_preview(pair: Pair, index: int, width: int = 78) -> str:
    from src.data import extract_code

    def block(label: str, text: str) -> str:
        code = extract_code(text) or "(nothing extractable)"
        body = "\n".join("    " + line for line in code.splitlines()[:14])
        return f"  {label}\n{body}"

    return "\n".join([
        "=" * width,
        f"pair {index}  task_id {pair.task_id}  "
        f"margin {pair.reward_margin:.2f}  edit {pair.edit_distance:.2f}  "
        f"len {pair.chosen_tokens} vs {pair.rejected_tokens} "
        f"({pair.length_delta:+d})",
        "-" * width,
        block(f"CHOSEN   reward {pair.chosen_reward:.2f}  "
              f"(sample {pair.chosen_sample_index})", pair.chosen),
        "-" * width,
        block(f"REJECTED reward {pair.rejected_reward:.2f}  "
              f"(sample {pair.rejected_sample_index})", pair.rejected),
    ])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", default=DEFAULT_COMPLETIONS, type=Path)
    parser.add_argument("--pairs-out", default=DEFAULT_PAIRS, type=Path)
    parser.add_argument("--balanced-out", default=DEFAULT_PAIRS_BALANCED, type=Path)
    parser.add_argument("--rft-out", default=DEFAULT_RFT, type=Path)
    parser.add_argument("--stats-out", default=DEFAULT_STATS, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cap", default=CAP_PER_PROBLEM, type=int)
    parser.add_argument("--min-reward-margin", default=MIN_REWARD_MARGIN, type=float)
    parser.add_argument("--min-edit-distance", default=MIN_EDIT_DISTANCE, type=float)
    parser.add_argument("--length-tolerance", default=LENGTH_TOLERANCE_TOKENS, type=float)
    parser.add_argument("--preview", default=0, type=int,
                        help="print N pairs and write nothing")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    by_problem = load_completions(args.completions)
    prompts = build_prompts(args.split, args.model)

    pairs, balanced, stats = build_pairs(
        by_problem, prompts,
        cap=args.cap,
        min_reward_margin=args.min_reward_margin,
        min_edit_distance=args.min_edit_distance,
        length_tolerance=args.length_tolerance,
    )
    rft, duplicates = build_rft(by_problem, prompts)
    stats.rft_examples = len(rft)
    stats.rft_problems = len({e["task_id"] for e in rft})
    stats.rft_duplicates_dropped = duplicates

    if args.preview:
        step = max(1, len(pairs) // args.preview)
        for i, pair in enumerate(pairs[::step][: args.preview], start=1):
            print(format_preview(pair, i))
        print("=" * 78)
        print(json.dumps(asdict(stats), indent=2))
        print("\npreview only, nothing written")
        return 0

    _write_jsonl(args.pairs_out, (asdict(p) for p in pairs))
    _write_jsonl(args.balanced_out, (asdict(p) for p in balanced))
    _write_jsonl(args.rft_out, rft)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")

    print(f"pairs      {stats.n_pairs} from {stats.n_problems_contributing} problems "
          f"(default, unbalanced)")
    print(f"balanced   {stats.n_pairs_balanced} pairs (ablation), "
          f"skew {stats.length_skew_pre_balance:+.1f} -> "
          f"{stats.length_skew_post_balance:+.1f} tokens")
    print(f"rft        {stats.rft_examples} examples from {stats.rft_problems} problems")
    print(f"written    {args.pairs_out}, {args.balanced_out}, {args.rft_out}, "
          f"{args.stats_out}")
    return 0


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
