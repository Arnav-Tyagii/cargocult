"""Contamination probe: which test problems can the model solve blind?

    python scripts/contamination.py

MBPP is public and pre-dates Qwen2.5's training cut-off, so some of it is in the
pretraining data. This asks the cheap version of the question: strip the natural
language description, leave only the signature the tests require, and sample.
Anything solved from that was not solved by reading the task.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
It is an **upper bound on memorization**, not a measurement of it, because the
signature is not information-free. MBPP function names frequently paraphrase the
task — `is_not_prime`, `max_chain_length`, `remove_Occ` — so a model can
sometimes infer the problem from the identifier alone without ever having seen
it. A flagged problem is therefore "solvable without the prose", which contains
genuine memorization and name-inference in unknown proportion.

That makes it the conservative direction for the purpose it serves. Removing
these problems removes more than contamination, so a training effect that
survives on the remainder survives a stricter test than contamination alone
demands.

Sampling matches the evaluation harness exactly — n=8, temperature 0.8,
top_p 0.95 — so "solved" means the same thing in both places. A problem counts
as flagged if any of the 8 blind samples passes all of its asserts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = REPO / "runs" / "contamination.json"


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data import extract_code, format_prompt
    from src.evaluate import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TOP_P, tier_problems
    from src.reward import reward_tier
    from src.sandbox import DEFAULT_TIMEOUT, run_tests

    device = "cuda" if torch.cuda.is_available() else "cpu"
    problems = tier_problems("full")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()

    prompts = [format_prompt(p, tokenizer, include_description=False) for p in problems]
    print(f"probing {len(problems)} test problems with the description withheld")
    print(f"example prompt tail: ...{prompts[0][-90:]!r}")

    from src.generate import sample_completions

    started = time.perf_counter()
    rows = sample_completions(
        model, tokenizer, prompts, 8, 0.8, DEFAULT_TOP_P, DEFAULT_MAX_NEW_TOKENS,
        batch_size=64, seed=0,
    )
    generate_minutes = (time.perf_counter() - started) / 60

    jobs = []
    for i, (problem, completions) in enumerate(zip(problems, rows)):
        for j, completion in enumerate(completions):
            jobs.append((i, j, extract_code(completion.text), problem))

    results: dict[tuple[int, int], object] = {}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2)) as pool:
        futures = {
            pool.submit(run_tests, code, problem.test_list,
                        setup_code=problem.test_setup_code, timeout=DEFAULT_TIMEOUT): (i, j)
            for i, j, code, problem in jobs
        }
        for future in futures:
            results[futures[future]] = future.result()

    flagged, detail = [], []
    for i, problem in enumerate(problems):
        outcomes = [results[(i, j)] for j in range(len(rows[i]))]
        n_solved = sum(
            1 for r in outcomes if r.n_tests > 0 and r.n_passed == r.n_tests
        )
        if n_solved:
            flagged.append(problem.task_id)
        detail.append({
            "task_id": problem.task_id,
            "n_solved_blind": n_solved,
            "n_samples": len(outcomes),
            "outcomes": [reward_tier(r) for r in outcomes],
        })

    rate = len(flagged) / len(problems)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "model": MODEL,
        "tier": "full",
        "n_problems": len(problems),
        "n_samples": 8,
        "temperature": 0.8,
        "top_p": DEFAULT_TOP_P,
        "prompt": "signature stub only, natural-language description withheld",
        "note": ("Upper bound on memorization: MBPP function names often paraphrase "
                 "the task, so this conflates memorization with name-inference. "
                 "Removing these problems is therefore stricter than removing "
                 "contamination alone."),
        "flagged_rate": round(rate, 4),
        "n_flagged": len(flagged),
        "flagged_task_ids": flagged,
        "generate_minutes": round(generate_minutes, 1),
        "problems": detail,
    }, indent=2), encoding="utf-8")

    print(f"\nflagged {len(flagged)}/{len(problems)} = {rate:.1%} solvable blind")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
