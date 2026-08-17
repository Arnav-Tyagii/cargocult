"""Setup shared by both trainers.

DPO and RFT are a controlled comparison, so anything that differs between them
by accident invalidates the result. Model loading, the LoRA configuration,
batch construction and the learning-rate schedule live here precisely so they
cannot drift apart — a copy-pasted LoRA rank that got edited on one side only
would look like a finding.
"""

from __future__ import annotations

import math

import torch

from src.losses import IGNORE_INDEX

MAX_SEQ = 768  # PROJECT.md §2

# PROJECT.md §2, fixed. ~8.8M trainable on Qwen2.5-0.5B.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_policy(
    model_name: str,
    *,
    device: str,
    dtype=torch.bfloat16,
    gradient_checkpointing: bool = True,
    init_adapter=None,
):
    """The policy, with a LoRA adapter. Returns (model, tokenizer).

    `init_adapter` continues from existing adapter weights instead of a fresh
    one — §5's sequential recipe, DPO on top of the RFT checkpoint. The
    reference model is still the base weights with the adapter disabled, which
    is the right reference for that recipe: DPO's constraint should be against
    the policy it is improving on, and here that policy *is* the adapter.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if init_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGETS,
                task_type="CAUSAL_LM",
            ),
        )

    # Adapter weights in fp32 even though the base is bf16: Adam's moments on
    # bf16 parameters lose updates smaller than ~1e-3 relative, which at
    # LR 1e-5 is most of them.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Without this the checkpointed blocks receive no grad-requiring input
        # (the embedding is frozen) and backward silently does nothing.
        model.enable_input_require_grads()
        model.config.use_cache = False

    model.to(device)
    return model, tokenizer


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def build_batch(tokenizer, prompts, completions, device, max_seq: int = MAX_SEQ):
    """Tokenized prompt+completion, right-padded.

    The completion arrives as the token ids the sampler actually emitted, not
    as text to re-encode. Decoding and re-encoding is not the identity — it
    drops the trailing EOS and can resegment whitespace — and any drift there
    would mean training on tokens the policy never produced.

    Returns (input_ids, attention_mask, labels, prompt_lens), all on `device`.
    """
    rows = []
    for prompt, completion_ids in zip(prompts, completions):
        prompt_ids = tokenizer(prompt).input_ids
        ids = (list(prompt_ids) + list(completion_ids))[:max_seq]
        rows.append((ids, min(len(prompt_ids), len(ids))))

    width = max(len(ids) for ids, _ in rows)
    pad_id = tokenizer.pad_token_id or 0
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)

    for i, (ids, _) in enumerate(rows):
        span = torch.tensor(ids, dtype=torch.long)
        input_ids[i, : len(ids)] = span
        attention_mask[i, : len(ids)] = 1
        labels[i, : len(ids)] = span

    prompt_lens = torch.tensor([pl for _, pl in rows], dtype=torch.long)
    return (
        input_ids.to(device),
        attention_mask.to(device),
        labels.to(device),
        prompt_lens.to(device),
    )


def completion_token_count(labels: torch.Tensor, prompt_lens: torch.Tensor) -> int:
    """How many tokens the loss is actually summed over. Same masking as
    `sequence_logprobs`, so a mismatch here would misnormalise the SFT loss."""
    targets = labels[:, 1:]
    positions = torch.arange(1, labels.shape[1], device=labels.device)
    mask = (positions.unsqueeze(0) >= prompt_lens.unsqueeze(1)) & (targets != IGNORE_INDEX)
    return int(mask.sum().item())


def cosine_schedule(total_steps: int, warmup_steps: int):
    """Linear warmup then cosine decay, as a LambdaLR multiplier."""

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return multiplier


def length_stats(lengths, hit_token_limit) -> dict[str, float]:
    """Mean completion length overall and among normally-terminated ones.

    Reported separately because they answer different questions. 11.6% of the
    rejected side of the pair corpus was cut off at the token budget, and a
    truncated completion is long for a reason that has nothing to do with the
    policy's verbosity. Mixing them means a run that learns to ramble and a run
    that merely hits the cap more often look identical (PROJECT.md §3b).
    """
    lengths = list(lengths)
    terminated = [n for n, hit in zip(lengths, hit_token_limit) if not hit]
    return {
        "mean": _mean(lengths),
        "mean_terminated": _mean(terminated),
        "n_token_limited": len(lengths) - len(terminated),
    }


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 2) if values else 0.0


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def save_adapter(model, path) -> None:
    """Adapter only — the base weights never change and are 1 GB (§3b)."""
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))


# --- checkpoint evaluation ---------------------------------------------------

# Dev tier, temperature 0.8, n=4 — the same configuration as
# runs/baseline/dev_temp0.8.json, which is what a checkpoint is compared to.
DEV_BASELINE_PASS_AT_1 = 0.2472
FULL_BASELINE_PASS_AT_1 = 0.2281


def evaluate_checkpoint(model, tokenizer, run_dir, step, *, seed=0, batch_size=32,
                        n_samples=4, temperature=0.8):
    """Dev-tier eval of the live training model, adapter and all.

    Evaluating in-process avoids reloading a 1 GB base model per checkpoint,
    but training has left the model in a state generation cannot use: the KV
    cache is off and gradient checkpointing is on. Both are flipped for the
    duration and restored in a finally, because leaving use_cache on would
    quietly make the next training step allocate a cache it never reads.
    """
    from pathlib import Path

    from src.evaluate import evaluate, tier_problems

    was_training = model.training
    checkpointing = getattr(model, "is_gradient_checkpointing", False)
    model.eval()
    model.config.use_cache = True
    if checkpointing:
        model.gradient_checkpointing_disable()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    try:
        report = evaluate(
            model, tokenizer, tier_problems("dev"), k=1, n_samples=n_samples,
            temperature=temperature, tier="dev", seed=seed, batch_size=batch_size,
            notes=f"checkpoint eval at step {step}",
        )
    finally:
        model.config.use_cache = False
        if checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        if was_training:
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report.save(Path(run_dir) / f"eval_step{step}.json")
    return report


def write_run_summary(run_dir, kind: str, config: dict, rows: list[dict],
                      evals: list) -> "Path":
    """runs/<tag>/summary.md — what this run did, in the terms §4 asks for."""
    from pathlib import Path

    run_dir = Path(run_dir)
    third = max(1, len(rows) // 3)
    losses = [r["loss"] for r in rows if "loss" in r]
    first = sum(losses[:third]) / third if losses else float("nan")
    last = sum(losses[-third:]) / third if losses else float("nan")
    final = rows[-1] if rows else {}

    lines = [
        f"# {run_dir.name}",
        "",
        f"`{kind}` · {len(rows)} steps · effective batch "
        f"{config.get('batch_size', '?')}x{config.get('grad_accum', '?')} · "
        f"peak {max((r.get('peak_vram_mb', 0) for r in rows), default=0):.0f} MB",
        "",
        "## Configuration",
        "",
        "| key | value |",
        "|---|---|",
    ]
    for key in ("beta", "lr", "epochs", "batch_size", "grad_accum", "seed",
                "pairs", "examples", "n_steps"):
        if key in config and config[key] is not None:
            lines.append(f"| {key} | `{config[key]}` |")

    lines += [
        "",
        "## Training",
        "",
        f"- loss {first:.4f} -> {last:.4f} (first third vs last third)",
    ]
    for key, label in (
        ("reward_margin", "implicit reward margin"),
        ("reward_accuracy", "reward accuracy"),
        ("logp_chosen", "logp chosen (absolute level)"),
        ("logp_rejected", "logp rejected (absolute level)"),
        ("grad_norm", "grad norm"),
    ):
        if key in final:
            lines.append(f"- final {label}: {final[key]:.4f}")

    if "logp_chosen" in final and rows:
        starts = [r for r in rows if "logp_chosen" in r]
        if starts:
            drift = final["logp_chosen"] - starts[0]["logp_chosen"]
            lines.append(
                f"- logp_chosen moved {drift:+.1f} nats over the run — "
                "negative here is likelihood displacement, and the loss curve "
                "will not show it"
            )

    lines += ["", "## Length", ""]
    for key in ("len_chosen", "len_chosen_terminated", "len_rejected",
                "len_rejected_terminated", "len_completion",
                "len_completion_terminated", "n_token_limited"):
        if key in final:
            lines.append(f"- {key}: {final[key]}")
    lines.append(
        "- `*_terminated` excludes completions cut off at the token budget; "
        "they are long for a reason unrelated to verbosity"
    )

    lines += ["", "## Dev-tier evals", "",
              f"Baseline dev pass@1 is **{DEV_BASELINE_PASS_AT_1:.4f}** "
              f"(runs/baseline/dev_temp0.8.json). The full-tier baseline is "
              f"{FULL_BASELINE_PASS_AT_1:.4f}; these evals are dev-tier.",
              "",
              "| step | pass@1 | vs baseline | mean tokens | terminated | stub_args | unparseable |",
              "|---|---|---|---|---|---|---|"]
    for step, report in evals:
        delta = report.pass_at_k - DEV_BASELINE_PASS_AT_1
        lines.append(
            f"| {step} | {report.pass_at_k:.4f} | {delta:+.4f} | "
            f"{report.mean_completion_tokens:.0f} | "
            f"{1 - report.token_limit_rate:.1%} | "
            f"{report.stub_args_rate:.1%} | {report.unparseable_rate:.1%} |"
        )
    if not evals:
        lines.append("| — | no checkpoint evals were run | | | | | |")

    path = run_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
