"""DPO training loop. The loss itself is in losses.py [OWNER WRITES].

    python -m src.train_dpo --dry-run
    python -m src.train_dpo --beta 0.1 --lr 1e-5 --tag beta0.1

Batch size 1-2 pairs with gradient accumulation to an effective 8, gradient
checkpointing on, adapter-only checkpoints (PROJECT.md §3b).

WHAT THE STEP LOG IS FOR
------------------------
Every field in the loss's metrics dict is logged, plus grad norm, LR, peak
VRAM and completion lengths. The two that matter most are the ones a loss
curve hides:

  - `logp_chosen` and `logp_rejected` as absolute levels. DPO constrains only
    their difference, so both can fall together while the loss improves. That
    is likelihood displacement, and it is invisible in loss, margin and
    reward_accuracy alike.
  - completion length, reported twice — over everything, and over normally
    terminated completions only. The pair corpus is skewed -55 tokens toward
    shorter chosen answers, so a policy that learns to truncate would be
    rewarded for it; separating the two stops a change in token-limit hits
    from being read as a change in verbosity.

One forward pass covers chosen and rejected together: they are concatenated
into a batch of 2N sequences, so the policy sees each pair once, not twice.
"""

from __future__ import annotations

import argparse
import json
import random
import math
from pathlib import Path

import torch

DEFAULT_PAIRS = Path("data/pairs_train.jsonl")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS, type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tag", default="dpo")
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=1, type=int, help="pairs per forward")
    parser.add_argument("--grad-accum", default=8, type=int)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--warmup-steps", default=10, type=int)
    parser.add_argument("--checkpoint-every", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--vram-ceiling-mb", default=3072.0, type=float,
                        help="PROJECT.md §2 target; --dry-run asserts against it")
    parser.add_argument("--run-dir", default=None, type=Path,
                        help="exact output directory; default runs/<timestamp>_<tag>")
    parser.add_argument("--eval-every", default=0, type=int,
                        help="dev-tier eval every N steps; 0 disables, final eval always runs")
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--no-final-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="20 steps on 50 pairs, asserting VRAM and that loss falls")
    return parser.parse_args(argv)


def load_pairs(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)

    from src.losses import dpo_loss, reference_logprobs, sequence_logprobs
    from src.utils.logging import JsonlLogger, create_run_dir, write_config
    from src.utils.training import (
        build_batch,
        cosine_schedule,
        length_stats,
        load_policy,
        evaluate_checkpoint,
        peak_vram_mb,
        save_adapter,
        write_run_summary,
        trainable_parameters,
    )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = load_pairs(args.pairs)
    # Shuffled by seed, not left in file order. Without this a "seed" only
    # varies LoRA initialisation and dropout, and the reported spread across
    # seeds understates real run-to-run variance — which matters most for the
    # one number the project publishes.
    random.Random(args.seed).shuffle(pairs)
    if args.dry_run:
        pairs = pairs[:50]
    n_steps = max(1, len(pairs) * args.epochs // (args.batch_size * args.grad_accum))
    if args.dry_run:
        # §3b asks for 20 steps on 50 pairs, which is ~3 passes over them at an
        # effective batch of 8. Cycling deliberately: judging "did the loss
        # fall" across 20 disjoint mini-batches would be judging it across 20
        # different datasets, and pair-to-pair variance here is larger than any
        # 20-step trend.
        n_steps = 20

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(f"{args.tag}_dry" if args.dry_run else args.tag)
    write_config(run_dir, {**vars(args), "n_steps": n_steps, "n_pairs": len(pairs),
                           "device": device})

    model, tokenizer = load_policy(
        args.model, device=device,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    # Warmup cannot outlast the run, or the LR never reaches its set value and
    # a short run looks like it cannot learn.
    warmup = min(args.warmup_steps, max(1, n_steps // 5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_schedule(n_steps, warmup)
    )

    n_trainable = sum(p.numel() for p in params)
    print(f"run        {run_dir}")
    print(f"pairs      {len(pairs)} | steps {n_steps} | effective batch "
          f"{args.batch_size * args.grad_accum} | warmup {warmup}")
    print(f"trainable  {n_trainable/1e6:.1f}M of {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    print(f"beta {args.beta} | lr {args.lr} | device {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger = JsonlLogger(run_dir / "log.jsonl")
    evals: list = []

    def _eval(at_step: int):
        report = evaluate_checkpoint(
            model, tokenizer, run_dir, at_step,
            seed=args.seed, batch_size=args.eval_batch_size,
        )
        logger.log(step=at_step, event="dev_eval",
                   dev_pass_at_1=report.pass_at_k,
                   dev_mean_tokens=report.mean_completion_tokens,
                   dev_stub_args_rate=report.stub_args_rate,
                   dev_token_limit_rate=report.token_limit_rate,
                   dev_unparseable_rate=report.unparseable_rate)
        print(f"  [eval @{at_step}] dev pass@1 {report.pass_at_k:.4f} "
              f"stub_args {report.stub_args_rate:.1%} "
              f"tokens {report.mean_completion_tokens:.0f}", flush=True)
        return report

    losses_seen: list[float] = []
    step_rows: list[dict] = []
    cursor = 0
    model.train()

    try:
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            lengths_chosen, lengths_rejected = [], []
            limit_chosen, limit_rejected = [], []

            for _ in range(args.grad_accum):
                batch = [pairs[(cursor + i) % len(pairs)] for i in range(args.batch_size)]
                cursor += args.batch_size

                # Chosen and rejected in one forward pass, chosen first.
                prompts = [p["prompt"] for p in batch] * 2
                completions = (
                    [p["chosen_token_ids"] for p in batch]
                    + [p["rejected_token_ids"] for p in batch]
                )
                input_ids, attention_mask, labels, prompt_lens = build_batch(
                    tokenizer, prompts, completions, device
                )

                policy = sequence_logprobs(
                    model, input_ids, labels, prompt_lens, attention_mask=attention_mask
                )
                reference = reference_logprobs(
                    model, input_ids, labels, prompt_lens, attention_mask=attention_mask
                )
                half = len(batch)
                loss, metrics = dpo_loss(
                    policy[:half], policy[half:],
                    reference[:half], reference[half:],
                    beta=args.beta,
                )
                (loss / args.grad_accum).backward()

                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value / args.grad_accum
                lengths_chosen += [len(p["chosen_token_ids"]) for p in batch]
                lengths_rejected += [len(p["rejected_token_ids"]) for p in batch]
                limit_chosen += [p["chosen_hit_token_limit"] for p in batch]
                limit_rejected += [p["rejected_hit_token_limit"] for p in batch]

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            chosen_len = length_stats(lengths_chosen, limit_chosen)
            rejected_len = length_stats(lengths_rejected, limit_rejected)
            record = logger.log(
                step=step,
                **{k: round(v, 6) for k, v in totals.items()},
                grad_norm=round(float(grad_norm), 4),
                lr=scheduler.get_last_lr()[0],
                peak_vram_mb=peak_vram_mb(),
                len_chosen=chosen_len["mean"],
                len_chosen_terminated=chosen_len["mean_terminated"],
                len_rejected=rejected_len["mean"],
                len_rejected_terminated=rejected_len["mean_terminated"],
                n_token_limited=chosen_len["n_token_limited"] + rejected_len["n_token_limited"],
            )
            losses_seen.append(totals["loss"])
            step_rows.append(record)

            if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
                print(f"  step {step:>4} loss {totals['loss']:.4f} "
                      f"margin {totals['reward_margin']:+.4f} "
                      f"logp_c {totals['logp_chosen']:.1f} "
                      f"logp_r {totals['logp_rejected']:.1f} "
                      f"|g| {record['grad_norm']:.2f} vram {record['peak_vram_mb']:.0f}MB",
                      flush=True)

            if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
                save_adapter(model, run_dir / f"checkpoint_{step + 1}")
            if args.eval_every and (step + 1) % args.eval_every == 0:
                evals.append((step + 1, _eval(step + 1)))

        if not args.dry_run:
            save_adapter(model, run_dir / "checkpoint_final")
            if not args.no_final_eval and not (
                evals and evals[-1][0] == n_steps
            ):
                evals.append((n_steps, _eval(n_steps)))
    finally:
        logger.close()

    # Max over the per-step readings, not the live counter: evaluate() resets
    # peak stats, so reading it at the end reports only the last eval's peak
    # and silently hides what training actually cost.
    vram = max([r.get("peak_vram_mb", 0.0) for r in step_rows] or [peak_vram_mb()])
    # Thirds rather than single steps: one mini-batch of one pair is noisier
    # than the trend being tested for, so a two-step comparison would pass or
    # fail on which pairs happened to land at the ends.
    third = max(1, len(losses_seen) // 3)
    first, last = _mean(losses_seen[:third]), _mean(losses_seen[-third:])
    print(f"\nloss       {first:.4f} -> {last:.4f} (first third vs last third)")
    print(f"peak vram  {vram:.0f} MB")
    print(f"log        {run_dir / 'log.jsonl'}")
    summary = write_run_summary(
        run_dir, "dpo",
        {**vars(args), "n_steps": n_steps, "pairs": len(pairs)},
        step_rows, evals,
    )
    print(f"summary    {summary}")

    if args.dry_run:
        _assert_dry_run(vram, args.vram_ceiling_mb, first, last, losses_seen)
        print("dry run OK: under the VRAM ceiling and the loss decreased")
    return 0


def _assert_dry_run(vram, ceiling, first, last, losses_seen) -> None:
    """§3b: the dry run asserts the VRAM ceiling and that loss decreases."""
    assert vram <= ceiling, f"peak VRAM {vram:.0f} MB exceeds the {ceiling:.0f} MB ceiling"
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
    # A loss pinned at exactly -log(0.5) means the reference equals the policy,
    # i.e. the adapter was never actually disabled or never actually trained.
    assert not all(
        abs(value - math.log(2)) < 1e-6 for value in losses_seen
    ), "loss sat at -log(0.5) throughout: policy and reference are identical"


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
