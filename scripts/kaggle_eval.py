"""Full-tier evaluation of saved adapter checkpoints.

    python scripts/kaggle_eval.py --checkpoints runs/dpo_.../checkpoint_final ...
    python scripts/kaggle_eval.py --tier dev --checkpoints ...

Takes a list of adapter directories and emits one EvalReport each (PROJECT.md
§3, §5). The base model is included by default as the control every checkpoint
is compared against; its report comes from the generation cache when the
settings match, so it costs execution time only.

Named for Kaggle and written to run there, but §7 measured the full tier at
14-27 minutes locally, so it runs locally by default.

WHY THE FULL TIER
-----------------
The dev tier is 90 problems and its paired standard error against baseline is
~0.025. The whole Phase 4 sweep landed inside +-0.033 of baseline, so every
result was inside the noise. The full tier is 200 problems x 8 samples, which
roughly halves that standard error — enough to separate a 3-point effect from
nothing, which is exactly the question the sweep could not answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUT = REPO / "runs" / "fulltier"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="*", default=[], type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tier", default="full", choices=["dev", "full"])
    parser.add_argument("--n-samples", default=8, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--no-base", action="store_true",
                        help="skip the untrained control")
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


def label_for(checkpoint: Path) -> str:
    """`<run tag>@<checkpoint>` — unique and readable in a results table."""
    return f"{checkpoint.parent.name}@{checkpoint.name}"


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluate import evaluate, tier_problems

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    problems = tier_problems(args.tier)

    targets: list[tuple[str, Path | None]] = []
    if not args.no_base:
        targets.append(("base", None))
    targets += [(label_for(Path(c)), Path(c)) for c in args.checkpoints]

    print(f"tier       {args.tier} ({len(problems)} problems x {args.n_samples})")
    print(f"targets    {[name for name, _ in targets]}")

    results = {}
    for name, checkpoint in targets:
        print(f"\n=== {name} ===", flush=True)
        # A fresh base per checkpoint. Loading is ~15s against a 15-minute
        # eval, and it removes any question of adapter state leaking between
        # evaluations — which would be indistinguishable from a real effect.
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        if checkpoint is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(checkpoint))
        model.to(device).eval()

        report = evaluate(
            model, tokenizer, problems, k=args.k, n_samples=args.n_samples,
            temperature=args.temperature, tier=args.tier, seed=args.seed,
            batch_size=args.batch_size,
            notes=args.notes or f"{args.tier}-tier eval of {name}",
        )
        path = report.save(args.out / f"{args.tier}_{name.replace('/', '_')}.json")
        results[name] = report
        print(report.summary())
        print(f"stub_args {report.stub_args_rate:.1%} | "
              f"unparseable {report.unparseable_rate:.1%} | "
              f"token limit {report.token_limit_rate:.1%}")
        print(f"written    {path}", flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    for name, report in results.items():
        print(f"{name:<40} pass@1 {report.pass_at_k_all['1']:.4f}  "
              f"tokens {report.mean_completion_tokens:>4.0f}  "
              f"stub {report.stub_args_rate:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
