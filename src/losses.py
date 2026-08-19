"""DPO loss and the log-probability machinery it sits on.

The loss itself is at the bottom. Everything above it is the plumbing it sits
on: turning a batch of prompt+completion sequences into one summed
log-probability per sequence, and getting the reference model's version of the
same number without loading a second model.

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
import torch.nn.functional as F

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
        would silently reintroduce a length preference.
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
    these weights with the LoRA adapter disabled. That is what makes DPO fit
    in 4 GB.

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
def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # [batch], from sequence_logprobs
    policy_rejected_logps: torch.Tensor,  # [batch]
    ref_chosen_logps: torch.Tensor,       # [batch], no_grad
    ref_rejected_logps: torch.Tensor,     # [batch], no_grad
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO loss (Rafailov et al., 2023), plus the metrics §3b logs.

    The objective is the log-sigmoid of the margin between how much more the
    policy prefers `chosen` over `rejected` than the reference does:

        L = -log σ( β · [ (πθ_c - ref_c) - (πθ_l - ref_l) ] )

    Written below in the regrouped form (πθ_c - πθ_l) - (ref_c - ref_l),
    which is algebraically identical but subtracts same-scale quantities
    first — the raw logps are large negatives (~-200 for a 200-token
    completion) and differencing them early keeps the magnitudes small.

    NOTE ON WHAT THIS DOES *NOT* CONSTRAIN
    --------------------------------------
    Only the *gap* appears in the loss. Nothing anchors the absolute level of
    either term, so the optimizer is free to push both chosen and rejected
    log-probs down as long as rejected falls faster. That is likelihood
    displacement, and it is why the returned metrics include the raw
    chosen/rejected logps and not just the margin — the pathology is
    invisible in the loss curve.
    """
    # Per-example margin. Detached reference terms carry no gradient.
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    # -log σ(β·logits). logsigmoid is used instead of log(sigmoid(x)) because
    # it is numerically stable in both tails; the naive form underflows to
    # -inf once β·logits drops below about -30.
    loss = -F.logsigmoid(beta * logits).mean()

    # Implicit rewards: DPO's derivation shows the optimal policy's reward is
    # β·log(πθ/ref). These are the reward-model outputs DPO never explicitly
    # trains, recovered for logging.
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    metrics = {
        "loss": loss.item(),
        "reward_chosen": chosen_rewards.mean().item(),
        "reward_rejected": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        # Fraction of pairs the policy currently orders correctly. Rises fast
        # and saturates near 1.0 well before pass@1 moves — it measures
        # ranking, not capability. Do not read it as progress.
        "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        # The two levels, for likelihood displacement. Watch these, not the margin.
        "logp_chosen": policy_chosen_logps.mean().item(),
        "logp_rejected": policy_rejected_logps.mean().item(),
    }
    return loss, metrics