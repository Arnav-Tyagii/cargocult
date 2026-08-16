"""Rejection-sampling fine-tuning: plain SFT on the completions that passed.

    python -m src.train_rft --dry-run
    python -m src.train_rft --lr 1e-5 --tag rft

The baseline DPO has to beat. Same policy, same LoRA configuration, same
schedule, same logging schema as train_dpo.py — everything shared lives in
utils/training.py so the two arms cannot drift apart by accident, because a
difference that crept in through a copy-paste would read as a result.

ANSWER-ONLY LOSS MASKING
------------------------
Cross-entropy is summed over completion tokens only, never the prompt, using
the same `prompt_lens` masking the DPO side uses. Training on the prompt would
teach the model to reproduce MBPP task descriptions, which nothing asks of it,
and would dominate the gradient — prompts are ~84 tokens against completions
of ~150.

Loss is normalised per token rather than per sequence. That is the standard
SFT objective, and it is also the honest one here: a per-sequence sum would
weight a 300-token solution twice as heavily as a 150-token one for no reason
connected to correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

DEFAULT_EXAMPLES = Path("data/rft_train.jsonl")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=DEFAULT_EXAMPLES, type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tag", default="rft")
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--grad-accum", default=4, type=int,
                        help="effective batch 8, matching the DPO arm")
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--warmup-steps", default=10, type=int)
    parser.add_argument("--checkpoint-every", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--vram-ceiling-mb", default=3072.0, type=float)
    parser.add_argument("--run-dir", default=None, type=Path,
                        help="exact output directory; default runs/<timestamp>_<tag>")
    parser.add_argument("--eval-every", default=0, type=int,
                        help="dev-tier eval every N steps; 0 disables, final eval always runs")
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--no-final-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="20 steps on 50 examples, asserting VRAM and that loss falls")
    return parser.parse_args(argv)


def load_examples(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)

    from src.losses import sequence_logprobs
    from src.utils.logging import JsonlLogger, create_run_dir, write_config
    from src.utils.training import (
        build_batch,
        completion_token_count,
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

    examples = load_examples(args.examples)
    if args.dry_run:
        examples = examples[:50]
    n_steps = max(1, len(examples) * args.epochs // (args.batch_size * args.grad_accum))
    if args.dry_run:
        n_steps = 20  # cycles the 50 examples; see the note in train_dpo.py

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(f"{args.tag}_dry" if args.dry_run else args.tag)
    write_config(run_dir, {**vars(args), "n_steps": n_steps,
                           "n_examples": len(examples), "device": device})

    model, tokenizer = load_policy(
        args.model, device=device,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    warmup = min(args.warmup_steps, max(1, n_steps // 5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_schedule(n_steps, warmup)
    )

    print(f"run        {run_dir}")
    print(f"examples   {len(examples)} | steps {n_steps} | effective batch "
          f"{args.batch_size * args.grad_accum} | warmup {warmup}")
    print(f"trainable  {sum(p.numel() for p in params)/1e6:.1f}M")
    print(f"lr {args.lr} | device {device}")

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
            step_loss = 0.0
            lengths, limits = [], []

            for _ in range(args.grad_accum):
                batch = [
                    examples[(cursor + i) % len(examples)] for i in range(args.batch_size)
                ]
                cursor += args.batch_size

                input_ids, attention_mask, labels, prompt_lens = build_batch(
                    tokenizer,
                    [e["prompt"] for e in batch],
                    [e["token_ids"] for e in batch],
                    device,
                )
                logprobs = sequence_logprobs(
                    model, input_ids, labels, prompt_lens, attention_mask=attention_mask
                )
                n_tokens = completion_token_count(labels, prompt_lens)
                # Negative mean log-likelihood per completion token.
                loss = -logprobs.sum() / max(1, n_tokens)
                (loss / args.grad_accum).backward()

                step_loss += loss.item() / args.grad_accum
                lengths += [len(e["token_ids"]) for e in batch]
                limits += [e["hit_token_limit"] for e in batch]

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            stats = length_stats(lengths, limits)
            record = logger.log(
                step=step,
                loss=round(step_loss, 6),
                perplexity=round(float(torch.tensor(step_loss).exp()), 4),
                grad_norm=round(float(grad_norm), 4),
                lr=scheduler.get_last_lr()[0],
                peak_vram_mb=peak_vram_mb(),
                len_completion=stats["mean"],
                len_completion_terminated=stats["mean_terminated"],
                n_token_limited=stats["n_token_limited"],
            )
            losses_seen.append(step_loss)
            step_rows.append(record)

            if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
                print(f"  step {step:>4} loss {step_loss:.4f} "
                      f"ppl {record['perplexity']:.2f} "
                      f"|g| {record['grad_norm']:.2f} "
                      f"len {record['len_completion']:.0f} "
                      f"vram {record['peak_vram_mb']:.0f}MB", flush=True)

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
    third = max(1, len(losses_seen) // 3)
    first = sum(losses_seen[:third]) / third
    last = sum(losses_seen[-third:]) / third
    print(f"\nloss       {first:.4f} -> {last:.4f} (first third vs last third)")
    print(f"peak vram  {vram:.0f} MB")
    print(f"log        {run_dir / 'log.jsonl'}")
    summary = write_run_summary(
        run_dir, "rft",
        {**vars(args), "n_steps": n_steps, "examples": len(examples)},
        step_rows, evals,
    )
    print(f"summary    {summary}")

    if args.dry_run:
        assert vram <= args.vram_ceiling_mb, (
            f"peak VRAM {vram:.0f} MB exceeds the {args.vram_ceiling_mb:.0f} MB ceiling"
        )
        assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
        print("dry run OK: under the VRAM ceiling and the loss decreased")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
