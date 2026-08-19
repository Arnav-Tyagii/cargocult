"""Batched sampling, keeping the log-probabilities the sampler already computed.

This is the Phase 2 pair generator, and it is also the rollout sampler for the
planned GRPO extension — same call, later moved inside a training loop. That is
why `Completion` carries more than text.

WHY LOGPROBS ARE STORED
-----------------------
DPO recomputes log-probabilities from the policy, so for Phase 2-4 these are
dead weight. GRPO needs the *sampling-time* values: the importance ratio is
pi_new(token) / pi_old(token), and pi_old is the distribution that actually
produced the token. Recovering it later is impossible once the adapter has
moved, so it is captured now, when it is free.

GETTING THEM WITHOUT 15 GB OF LOGITS
------------------------------------
The obvious route, `generate(output_scores=True)`, keeps one [batch, vocab]
tensor per step for the whole generation. At batch 64, 384 steps and Qwen's
151,936-token vocabulary that is ~15 GB of float32 on a 4 GB card.

Instead a LogitsProcessor sits at the end of the chain and holds exactly one
step of log-probabilities. On the next call the token sampled from them is
visible as the last element of `input_ids`, so its log-probability is gathered
then and the distribution is dropped. Peak cost is one [batch, vocab] tensor
(~39 MB) rather than one per step, and generation does no extra compute.

WHICH DISTRIBUTION THESE ARE
----------------------------
The policy's own: log pi(token) under the unwarped model. Transformers applies
temperature, top-k and top-p *after* any custom logits processor, so the
recorder sees pre-warp scores. Measured across temperature 0.5/1.0/2.0 and
top_p 0.2/1.0, the recorded values match a teacher-forcing recomputation to
5e-6 in every case — warping the sampler does not move them.

That is the semantics worth having. An importance ratio needs pi_new/pi_old
with both sides measured the same way, and pi_new will come from a training
forward pass, which has no warpers in it. Recording the warped behaviour
distribution instead would leave a temperature factor in the denominator that
the numerator never sees.

The caveat, stated plainly: these are *not* the distribution the tokens were
drawn from whenever temperature != 1.0 or top_p < 1.0. §2b samples at
temperature 1.0, where top_p is the only gap.

What this does not excuse: a repetition penalty is a processor, not a warper,
so it lands *before* the recorder and does corrupt these values. Neutralising
it below is what took the teacher-forcing gap from 1.61 nats to 0.086.

Verified against an independent teacher-forcing pass over prompt+completion,
which is how a training step would recompute them:

    float32   max |diff| 0.00001 over 40 tokens
    bfloat16  max |diff| 0.086, mean 0.016

So the recording is exact, and bf16 alone costs up to ~0.09 nats per token —
incremental decoding through a KV cache and a single full forward do not
produce bit-identical logits. That is the noise floor on any later importance
ratio: ~9% on the worst token. Recompute pi_old in fp32 if that ever matters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import torch
from tqdm import tqdm
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

# Fixed for the project. evaluate.py carries its own copy until it is moved
# onto this module; this is the one that should survive, since evaluate will
# import from here and not the other way round.
MAX_SEQ = 768

DEFAULT_BATCH_SIZE = 64  # sequences per forward pass; ~2.1 GB peak on a 3050 Ti


@dataclass
class Completion:
    """One sampled completion. FROZEN: the planned GRPO extension depends on
    these fields.

    token_ids and logprobs are parallel and both stop at the first EOS
    (inclusive) — the EOS is part of what the policy has to learn to emit, so
    it carries a log-probability like any other token.
    """

    text: str
    token_ids: list[int]
    logprobs: list[float]
    hit_token_limit: bool

    def __post_init__(self):
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError(
                f"{len(self.token_ids)} tokens but {len(self.logprobs)} logprobs; "
                "an importance ratio computed from these would be silently misaligned"
            )


class _LogprobRecorder(LogitsProcessor):
    """Records log pi(sampled token) for each step, one step behind."""

    def __init__(self) -> None:
        self.pending: torch.Tensor | None = None  # [batch, vocab] log-probs
        self.steps: list[torch.Tensor] = []  # each [batch]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.pending is not None:
            # The last column of input_ids is the token drawn from `pending`.
            self._record(input_ids[:, -1])
        # log_softmax allocates a new tensor, so a later in-place processor
        # cannot corrupt what we are holding.
        self.pending = torch.log_softmax(scores.float(), dim=-1)
        return scores

    def _record(self, tokens: torch.Tensor) -> None:
        assert self.pending is not None
        self.steps.append(self.pending.gather(1, tokens[:, None]).squeeze(1))
        self.pending = None

    def finish(self, sequences: torch.Tensor) -> torch.Tensor:
        """Close out the final step and return [batch, steps] on the CPU."""
        if self.pending is not None:
            self._record(sequences[:, -1])
        if not self.steps:
            return torch.empty(sequences.shape[0], 0)
        # Kept on the GPU until now: a per-step transfer would sync every token.
        return torch.stack(self.steps, dim=1).float().cpu()


@torch.no_grad()
def sample_completions(
    model,
    tokenizer,
    prompts: Sequence[str],
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int | None = None,
    progress: bool = True,
) -> list[list[Completion]]:
    """Draw n completions for each prompt. FROZEN: the planned GRPO extension
    depends on this signature.

    Args:
        prompts: fully formatted prompts, chat template already applied.
        n: completions per prompt.
        temperature: 0 means greedy, in which case n > 1 is wasted compute.
        top_p: nucleus cutoff. Ignored when greedy.
        max_new_tokens: generation budget. A completion that reaches it
            without emitting EOS is flagged, and the reward ladder docks it.
        batch_size: sequences per forward pass, not prompts — each prompt
            expands to n sequences. VRAM scales with this; throughput scales
            with it hard, because generation is memory-bandwidth-bound.
        seed: seeds torch's global RNG. Results are reproducible only for a
            fixed batch_size, since batching changes how draws consume it.

    Returns:
        One list of n Completions per prompt, in the order given.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if temperature == 0 and n > 1:
        raise ValueError("greedy decoding with n > 1 returns n copies of one text")
    if max_new_tokens >= MAX_SEQ:
        raise ValueError(f"max_new_tokens={max_new_tokens} leaves no room for a prompt")

    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    eos_ids = _eos_token_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else 0

    # Decoder-only batched generation needs left padding. Over-long prompts are
    # truncated from the left too: the tail carries the task and the generation
    # prompt, which matter more than the system header.
    padding_side, truncation_side = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    prompt_budget = MAX_SEQ - max_new_tokens

    # Built from scratch rather than inherited. Qwen2.5 ships top_k=20 and
    # repetition_penalty=1.1 in its generation_config, and generate applies
    # both even when temperature and top_p are passed explicitly. Three things
    # break if a checkpoint's own config is allowed to leak in here:
    #   - the signature claims sampling is (temperature, top_p) and it is not;
    #     top_k=20 alone discards almost all the diversity that §2b's
    #     temperature 1.0 exists to buy
    #   - repetition_penalty makes the sampling distribution a function of the
    #     generated prefix rather than of the policy, so the logprobs below
    #     stop being a pi_old that any training pass could reproduce
    #   - a future checkpoint shipping a different config would silently
    #     change how the dataset was sampled, with nothing in the diff
    # Passing a whole GenerationConfig replaces the model's outright, so this
    # holds for fields nobody thought to override too.
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        pad_token_id=pad_id,
        eos_token_id=eos_ids or None,
        repetition_penalty=1.0,
        use_cache=True,  # KV cache; without it this is quadratic
        return_dict_in_generate=True,
        # top_k=None disables the warper — GenerationConfig's own default is
        # 50, so leaving it out would truncate. Greedy sets the sampling
        # fields to None because it genuinely does not use them; transformers
        # still logs "generation flags are not valid and may be ignored:
        # ['temperature', 'top_p', 'top_k']" once per process on this path,
        # which is correct and can be ignored. Verified: greedy output matches
        # the raw model's argmax on 24/24 tokens including repeats, so no
        # processor is quietly active.
        **(
            dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=None)
            if temperature > 0
            else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        ),
    )

    per_batch = max(1, batch_size // n)
    completions: list[list[Completion]] = []
    n_truncated = 0

    try:
        for start in tqdm(
            range(0, len(prompts), per_batch),
            desc="sample",
            unit="batch",
            disable=not progress,
        ):
            chunk = list(prompts[start : start + per_batch])
            n_truncated += sum(
                1 for p in chunk if len(tokenizer(p).input_ids) > prompt_budget
            )
            batch = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_budget,
            ).to(model.device)

            recorder = _LogprobRecorder()
            out = model.generate(
                **batch,
                generation_config=config,
                logits_processor=LogitsProcessorList([recorder]),
            )
            new_tokens = out.sequences[:, batch["input_ids"].shape[1] :]
            logprobs = recorder.finish(out.sequences)
            assert logprobs.shape[1] == new_tokens.shape[1], (
                f"recorded {logprobs.shape[1]} steps for {new_tokens.shape[1]} tokens"
            )

            for i in range(len(chunk)):
                rows = slice(i * n, (i + 1) * n)
                completions.append(
                    [
                        _to_completion(ids, lps, tokenizer, eos_ids, max_new_tokens)
                        for ids, lps in zip(new_tokens[rows], logprobs[rows])
                    ]
                )
    finally:
        tokenizer.padding_side = padding_side
        tokenizer.truncation_side = truncation_side

    if n_truncated:
        warnings.warn(
            f"{n_truncated} of {len(prompts)} prompts exceeded the "
            f"{prompt_budget}-token budget and were truncated from the left",
            stacklevel=2,
        )
    return completions


def _to_completion(
    token_ids: torch.Tensor,
    logprobs: torch.Tensor,
    tokenizer,
    eos_ids: Sequence[int],
    max_new_tokens: int,
) -> Completion:
    """Trim one row to its real length, dropping the batch's padding."""
    ids = token_ids.tolist()
    length = len(ids)
    hit_token_limit = True
    for position, token in enumerate(ids):
        if token in eos_ids:
            length = position + 1  # keep the EOS itself
            hit_token_limit = False
            break

    ids = ids[:length]
    return Completion(
        text=tokenizer.decode(ids, skip_special_tokens=True),
        token_ids=ids,
        logprobs=[round(lp, 5) for lp in logprobs[:length].tolist()],
        # A row shorter than the budget cannot have been cut off by it: the
        # batch only stops early once every sequence has finished.
        hit_token_limit=hit_token_limit and len(token_ids) >= max_new_tokens,
    )


def _eos_token_ids(model, tokenizer) -> list[int]:
    ids: list[int] = []
    for source in (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if source is None:
            continue
        ids.extend(source if isinstance(source, (list, tuple)) else [source])
    return sorted({int(i) for i in ids})
