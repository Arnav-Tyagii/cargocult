"""Record the untrained baselines. One command, reproducible from a seed.

    python scripts/baseline.py                     # full tier, both baselines
    python scripts/baseline.py --tier dev          # the ~20 min iteration tier
    python scripts/baseline.py --limit 8 --n-samples 2   # smoke test

Two configurations are recorded for the policy model, per PROJECT.md §1d:
greedy, and temperature 0.8 reported at pass@1 and pass@8. The third baseline
(PythonGPT, the 54.6M from-scratch model, expected ~0) runs through the same
command with --model pointing at that checkpoint.

Nothing here is trained. The point is a number that later work has to beat,
produced by exactly the harness that will score that later work.
"""

import argparse
import sys
from pathlib import Path

# Run from anywhere: this script lives one level below the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tier", default="full", choices=["dev", "full"])
    parser.add_argument("--out", default="runs/baseline", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-samples", default=8, type=int,
                        help="samples per problem for the temperature run")
    parser.add_argument("--temperature", default=0.8, type=float)
    # 64 sequences measured at 136 seq/min and 2.0 GB peak on the 3050 Ti,
    # against 14 seq/min at batch 8 — generation here is memory-bandwidth
    # bound, so small batches pay for the weight reads and get nothing back.
    # Well inside the 3.0 GB target; drop it if a longer-prompt tier OOMs.
    parser.add_argument("--batch-size", default=64, type=int,
                        help="sequences per forward pass; VRAM scales with this")
    parser.add_argument("--max-new-tokens", default=384, type=int)
    parser.add_argument("--n-workers", default=None, type=int,
                        help="sandbox processes; defaults to half the cores")
    parser.add_argument("--limit", default=None, type=int,
                        help="evaluate only the first N problems (debugging)")
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--notes", default="",
                        help="free text recorded in each report, for provenance")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Imported here, not at module level: on Windows every sandbox worker
    # re-imports this module, and torch costs ~6s each time it does.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluate import evaluate, tier_problems

    problems = tier_problems(args.tier)
    if args.limit:
        problems = problems[: args.limit]

    print(f"model      {args.model}")
    print(f"tier       {args.tier} ({len(problems)} problems)")
    print(f"device     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    shared = dict(
        tier=args.tier,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        n_workers=args.n_workers,
        notes=args.notes,
    )
    runs = []
    if not args.skip_greedy:
        # Greedy is one sample by definition: pass@1 with no sampling noise.
        runs.append(("greedy", dict(k=1, n_samples=1, temperature=0.0)))
    runs.append(
        (
            f"temp{args.temperature:g}",
            dict(k=1, n_samples=args.n_samples, temperature=args.temperature),
        )
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for tag, config in runs:
        print(f"\n=== {tag} ===")
        report = evaluate(model, tokenizer, problems, **config, **shared)
        path = report.save(args.out / f"{args.tier}_{tag}.json")
        print(report.summary())
        print(f"pass@k     {report.pass_at_k_all}")
        print(f"outcomes   {report.outcome_counts}")
        print(f"tokens     mean {report.mean_completion_tokens:.0f}, "
              f"{report.token_limit_rate:.1%} hit the limit")
        print(f"cost       {report.generation_seconds / 60:.1f} min generate, "
              f"{report.execution_seconds / 60:.1f} min execute, "
              f"{report.peak_vram_mb:.0f} MB peak vram"
              f"{' (cached)' if report.cache_hit else ''}")
        print(f"written    {path}")
    return 0


if __name__ == "__main__":
    # Required on Windows: the sandbox pool spawns, and each worker imports
    # this module. Without the guard the whole eval would restart per worker.
    raise SystemExit(main())
