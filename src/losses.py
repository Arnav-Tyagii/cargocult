"""DPO loss and the log-probability machinery it sits on.

The loss itself is [OWNER WRITES] — see the marked section at the bottom and
PROJECT.md §3a. What is here is the plumbing underneath it: turning a batch of
prompt+completion sequences into one summed log-probability per sequence, and
getting the reference model's version of the same number without loading a
second model.

WHY THE MASKING IS THE DANGEROUS PART
------------------------------------
`sequence_logprobs` has to score the completion and nothing else. Include the
prompt tokens and every sequence gains a large, nearly constant term that has
nothing to do with the policy's answer; because it is *nearly* constant it
does not obviously break anything, it just quietly shrinks the margin between
chosen and rejected and makes the loss look better behaved than it is. This is
the single most common place to introduce a silent bug in a DPO
implementation, which is why it is tested against a hand-computed example
rather than against itself.

THE OFF-BY-ONE
--------------
A causal LM's logits at position t predict the token at position t+1. So the
log-probability of the token sitting at position t comes from the logits at
t-1, and the token at position 0 has no predictor at all. Everything below is
shifted accordingly, and the test suite checks the shift by feeding logits
whose softmax is known exactly.

MEMORY
------
`log_softmax` over the full vocabulary would allocate a second [batch, seq,
151936] tensor and, with autograd on, keep it for the backward pass. At 768
tokens that is ~470 MB per copy in bf16 on a 4 GB card. Using the identity

    log p(y) = logit[y] - logsumexp(logits)

gathers one value per position and reduces the rest, so the only [batch, seq,
vocab] tensor in play is the one the model already produced.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100  # the HuggingFace convention for "not a target"


def sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prompt_lens: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Summed log-probability of each sequence's completion tokens.

    Args:
        model: the policy. Called once; gradients flow unless the caller is
            inside `torch.no_grad()`.
        input_ids: [batch, seq] of prompt followed by completion, padded.
        labels: [batch, seq], normally equal to input_ids, with IGNORE_INDEX
            at padding positions. Prompt positions may be left as-is —
            `prompt_lens` masks them either way.
        prompt_lens: [batch], number of leading tokens that are prompt. Tokens
            at positions >= prompt_lens[i] are scored; earlier ones are not.
        attention_mask: [batch, seq], passed through to the model. Required
            whenever the batch is padded, or the padding will attend.

    Returns:
        [batch] of summed log-probabilities. Not averaged by length: DPO
        compares two completions of different lengths and a per-token mean
        would silently reintroduce a length preference (PROJECT.md §2c).
    """
    if input_ids.shape != labels.shape:
        raise ValueError(f"input_ids {tuple(input_ids.shape)} != labels {tuple(labels.shape)}")
    if prompt_lens.shape[0] != input_ids.shape[0]:
        raise ValueError("prompt_lens must have one entry per sequence")

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Drop the final position: it predicts a token beyond the sequence.
    logits = logits[:, :-1, :]
    targets = labels[:, 1:]

    # Column j of `targets` is the token at original position j+1.
    positions = torch.arange(1, input_ids.shape[1], device=input_ids.device)
    is_completion = positions.unsqueeze(0) >= prompt_lens.unsqueeze(1).to(positions.device)
    mask = is_completion & (targets != IGNORE_INDEX)

    # clamp: IGNORE_INDEX is not a valid index to gather with. Those positions
    # are masked out immediately afterwards, so the value gathered is unused.
    safe = targets.clamp_min(0).unsqueeze(-1)
    chosen_logits = logits.gather(dim=-1, index=safe).squeeze(-1)
    token_logprobs = chosen_logits - logits.logsumexp(dim=-1)

    return (token_logprobs * mask).sum(dim=-1)


def reference_logprobs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prompt_lens: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The same quantity under the reference policy: the adapter switched off.

    The reference model in this project is not a second set of weights, it is
    these weights with the LoRA adapter disabled (PROJECT.md §2). That is what
    makes DPO fit in 4 GB.

    Two failure modes are guarded here because both are silent. If the adapter
    is not actually disabled, the reference equals the policy, every log-ratio
    is zero, the loss sits at exactly -log(0.5) and the model trains against
    itself. If the adapter is not re-enabled afterwards, the *policy* forward
    pass that follows also runs on base weights and no gradient reaches the
    adapter at all.
    """
    if not hasattr(model, "disable_adapter"):
        raise TypeError(
            "reference_logprobs needs a PEFT model; got "
            f"{type(model).__name__}, which has no disable_adapter()"
        )
    with torch.no_grad(), model.disable_adapter():
        flags = _adapter_flags(model)
        if not flags or not all(flags):
            raise RuntimeError(
                "disable_adapter() did not disable anything — the reference "
                "would be identical to the policy and the loss would be a "
                "constant -log(0.5)"
            )
        logprobs = sequence_logprobs(
            model, input_ids, labels, prompt_lens, attention_mask=attention_mask
        )
    if _adapter_is_disabled(model):
        raise RuntimeError(
            "the adapter is still disabled after disable_adapter() exited; "
            "the policy pass would run on base weights and receive no gradient"
        )
    return logprobs.detach()


def _adapter_flags(model) -> list[bool]:
    """Every LoRA layer's disabled/enabled flag.

    The `isinstance(..., bool)` filter is not defensive padding. On a peft
    tuner layer `disable_adapters` is a bool property, but it is *also* the
    name of a method on transformers' PeftAdapterMixin, which every model
    module inherits. A plain truthiness test therefore sees a bound method —
    always truthy — and reports every model as disabled, including one whose
    adapter is fully live. Only genuine booleans are flags.
    """
    flags = []
    for module in model.modules():
        flag = getattr(module, "disable_adapters", None)
        if isinstance(flag, bool):
            flags.append(flag)
    return flags


def _adapter_is_disabled(model) -> bool:
    """True if any LoRA layer is currently switched off."""
    return any(_adapter_flags(model))


# --- the DPO loss ------------------------------------------------------------
#
# [OWNER WRITES] — PROJECT.md §3a. Deliberately left empty.
#
# `tests/test_losses.py` already contains its contract, currently skipped:
# define `dpo_loss` here and those tests activate. What they expect is a
# function taking the four summed log-probabilities (policy and reference, for
# chosen and rejected) plus beta, and returning a scalar loss such that:
#
#   - a policy identical to the reference gives exactly -log(0.5)
#   - the loss falls toward 0 as the policy's margin over the reference grows
#   - the loss grows large when that margin inverts
#   - gradients reach only the LoRA parameters
