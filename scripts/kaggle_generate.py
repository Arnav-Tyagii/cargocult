"""Sample the preference dataset: every training problem, n completions each.

    python scripts/kaggle_generate.py                  # 374 problems x 8
    python scripts/kaggle_generate.py --limit 4 --n 2  # smoke test
    python scripts/kaggle_generate.py --resume         # continue a dead run

    # extend an existing dataset to 16 samples/problem, then merge the files:
    python scripts/kaggle_generate.py --sample-offset 8 --out data/completions_9to16.jsonl

Generate once, train many times — this file produces the artifact that Phase 3
and Phase 4 consume, so its provenance matters more than its speed. Every
completion is written with the log-probabilities it was sampled under, its
sandbox result and its reward, and the run's configuration goes in a sidecar.

Named for Kaggle and written to run there (self-contained, model pulled from
the hub, output under /kaggle/working when that exists), but §7 measured the
local run at under an hour, so it runs locally by default.

WHY THERE IS NO GENERATE/EXECUTE OVERLAP
----------------------------------------
The original design called for a ProcessPoolExecutor consuming a queue that the
GPU loop feeds, on the grounds that serialising sandbox scoring after generation
"adds hours". Measured on the Phase 1 baselines, scoring 1,600 completions
took 36 seconds across 10 workers: the 6s timeout is not the typical case,
only 4 of 1,600 hit it, and the mean execution is ~0.1s. Serial scoring of
3,000 completions costs about a minute, so the queue is not worth its own
failure modes. Chunking below gives most of the overlap for free anyway — the
CPU scores chunk N while nothing else needs the GPU.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
KAGGLE_WORKING = Path("/kaggle/working")


def default_out() -> Path:
    if KAGGLE_WORKING.exists():
        return KAGGLE_WORKING / "completions.jsonl"
    return Path("data/completions.jsonl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--split", default="train", help="MBPP split to sample")
    parser.add_argument("--n", default=8, type=int, help="completions per problem")
    parser.add_argument("--temperature", default=1.0, type=float)
    # 1.0, not 0.95: at temperature 1.0 with no nucleus cutoff the stored
    # logprobs *are* the distribution the tokens were drawn from, which is the
    # one thing GRPO needs them for (§8). Any top_p < 1 leaves pi_old
    # describing a distribution that never produced these tokens.
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--max-new-tokens", default=384, type=int)
    parser.add_argument("--batch-size", default=64, type=int,
                        help="sequences per forward pass; VRAM scales with this")
    parser.add_argument("--chunk", default=32, type=int,
                        help="problems per write; smaller loses less to a crash")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--n-workers", default=None, type=int)
    parser.add_argument("--timeout", default=None, type=float)
    parser.add_argument("--sample-offset", default=0, type=int,
                        help="number the samples from here, and draw a disjoint "
                             "set: use 8 to add samples 9-16 to an existing run")
    parser.add_argument("--resume", action="store_true",
                        help="skip task_ids already present in the output file")
    return parser.parse_args(argv)


def already_done(path: Path) -> set[int]:
    """task_ids with at least one completion on disk."""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["task_id"])
                except (ValueError, KeyError):
                    pass  # a torn last line from a killed run
    return done


def main(argv=None) -> int:
    args = parse_args(argv)

    # Imported here: on Windows every sandbox worker re-imports this module.
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data import extract_code, format_prompt, load_problems
    from src.generate import sample_completions
    from src.reward import compute_reward, reward_tier
    from src.sandbox import DEFAULT_TIMEOUT, run_tests

    out = args.out or default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    timeout = args.timeout or DEFAULT_TIMEOUT
    n_workers = args.n_workers or max(1, (os.cpu_count() or 4) // 2)

    problems = load_problems(args.split)
    if args.limit:
        problems = problems[: args.limit]
    if args.resume:
        done = already_done(out)
        problems = [p for p in problems if p.task_id not in done]
        print(f"resuming: {len(done)} problems already written, {len(problems)} to go")
    elif out.exists():
        out.unlink()  # a partial file from a previous run is not a valid dataset

    print(f"model      {args.model}")
    print(f"split      {args.split} ({len(problems)} problems x {args.n} samples)")
    print(f"sampling   temperature {args.temperature}, top_p {args.top_p}, "
          f"max_new_tokens {args.max_new_tokens}, seed {args.seed}")
    print(f"device     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"out        {out}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generate_seconds = execute_seconds = 0.0
    n_written = 0
    outcomes: dict[str, int] = {}

    with out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(problems), args.chunk):
            chunk = problems[start : start + args.chunk]
            prompts = [format_prompt(p, tokenizer) for p in chunk]

            mark = time.perf_counter()
            rows = sample_completions(
                model, tokenizer, prompts, args.n, args.temperature, args.top_p,
                args.max_new_tokens, batch_size=args.batch_size,
                # Offset per chunk so a resumed run does not redraw what it
                # already has, and per sample-offset so extending a dataset
                # draws new completions rather than a copy of the first pass.
                seed=args.seed + args.sample_offset * 10_000 + start,
            )
            generate_seconds += time.perf_counter() - mark

            mark = time.perf_counter()
            codes = [
                (i, j, extract_code(c.text))
                for i, row in enumerate(rows)
                for j, c in enumerate(row)
            ]
            results: dict[tuple[int, int], object] = {}
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        run_tests, code, chunk[i].test_list,
                        setup_code=chunk[i].test_setup_code, timeout=timeout,
                    ): (i, j)
                    for i, j, code in codes
                }
                for future in futures:
                    results[futures[future]] = future.result()
            execute_seconds += time.perf_counter() - mark

            for i, problem in enumerate(chunk):
                for j, completion in enumerate(rows[i]):
                    result = results[(i, j)]
                    tier = reward_tier(result)
                    outcomes[tier] = outcomes.get(tier, 0) + 1
                    handle.write(json.dumps({
                        "task_id": problem.task_id,
                        "sample_index": args.sample_offset + j,
                        "text": completion.text,
                        "code": extract_code(completion.text),
                        "token_ids": completion.token_ids,
                        "logprobs": completion.logprobs,
                        "hit_token_limit": completion.hit_token_limit,
                        "reward": round(compute_reward(result, completion.hit_token_limit), 4),
                        "outcome": tier,
                        "exec": {
                            "parsed": result.parsed,
                            "ran": result.ran,
                            "n_tests": result.n_tests,
                            "n_passed": result.n_passed,
                            "timed_out": result.timed_out,
                            "stderr_tail": result.stderr_tail,
                            "wall_time": round(result.wall_time, 3),
                        },
                    }) + "\n")
                    n_written += 1
            handle.flush()
            done_problems = start + len(chunk)
            elapsed = time.perf_counter() - started
            print(f"  {done_problems}/{len(problems)} problems, {n_written} completions, "
                  f"{elapsed / 60:.1f} min elapsed, {outcomes}", flush=True)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available() else 0.0
    )
    meta = {
        "model": args.model,
        "split": args.split,
        "n_problems": len(problems),
        "n_samples": args.n,
        "sample_offset": args.sample_offset,
        "n_completions": n_written,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": None,
        "repetition_penalty": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timeout": timeout,
        "outcomes": dict(sorted(outcomes.items())),
        "generate_seconds": round(generate_seconds, 1),
        "execute_seconds": round(execute_seconds, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }
    meta_path = out.with_name(out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nwrote      {n_written} completions to {out}")
    print(f"outcomes   {meta['outcomes']}")
    print(f"pass rate  {outcomes.get('pass', 0) / max(1, n_written):.1%} of completions")
    print(f"cost       {generate_seconds / 60:.1f} min generate, "
          f"{execute_seconds / 60:.1f} min execute, {peak_vram_mb:.0f} MB peak vram")
    print(f"meta       {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
