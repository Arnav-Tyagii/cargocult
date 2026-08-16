"""Tests for the sampler.

Everything except the last test runs on a 2-layer randomly-initialised Qwen2
with a 43-token vocabulary, on the CPU in float32. That combination matters:
it runs anywhere, including Kaggle and CI with no GPU, and float32 on one
device makes the teacher-forcing cross-check exact rather than approximate,
so the central invariant can be asserted at 1e-5 instead of at bf16's 0.09.

The tokenizer is a real PreTrainedTokenizerFast built in-process, not a stub.
Left padding is the thing most likely to break silently here, and a fake
tokenizer would fake exactly the behaviour under test.
"""

import warnings

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from src.generate import (
    MAX_SEQ,
    Completion,
    _LogprobRecorder,
    _to_completion,
    sample_completions,
)

UNK, PAD, EOS = 0, 1, 2
N_WORDS = 40


@pytest.fixture(scope="module")
def tokenizer():
    vocab = {"<unk>": UNK, "<pad>": PAD, "<eos>": EOS}
    vocab.update({f"t{i}": i + 3 for i in range(N_WORDS)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"
    )


@pytest.fixture(scope="module")
def model(tokenizer):
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        eos_token_id=EOS,
        pad_token_id=PAD,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config).eval()


# Deliberately different lengths: batching these forces left padding.
PROMPTS = ["t1 t2 t3 t4 t5 t6", "t7 t8", "t9 t10 t11 t12"]


def teacher_forcing_logprobs(model, tokenizer, prompt, token_ids):
    """Recompute log pi(token) with no padding and no KV cache."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    completion = torch.tensor([token_ids])
    logits = model(torch.cat([prompt_ids, completion], dim=1)).logits[0]
    start = prompt_ids.shape[1] - 1
    window = logits[start : start + completion.shape[1]]
    lp = torch.log_softmax(window.float(), dim=-1)
    return lp.gather(1, completion[0][:, None]).squeeze(1).tolist()


# --- the central invariant ---------------------------------------------------


def test_logprobs_match_an_independent_teacher_forcing_pass(model, tokenizer):
    """Recorded log-probabilities must equal what a training step computes.

    This is the one that matters: it covers the LogitsProcessor gather and
    left-padding alignment at once. If padding shifted a sequence's positions,
    the short prompt's completion would be scored against the wrong context
    and these numbers would diverge.
    """
    rows = sample_completions(
        model, tokenizer, PROMPTS, n=2, temperature=1.0, top_p=1.0,
        max_new_tokens=12, batch_size=6, seed=0, progress=False,
    )
    assert len(rows) == len(PROMPTS) and all(len(row) == 2 for row in rows)

    worst = 0.0
    for prompt, row in zip(PROMPTS, rows):
        for completion in row:
            expected = teacher_forcing_logprobs(
                model, tokenizer, prompt, completion.token_ids
            )
            assert len(expected) == len(completion.logprobs)
            worst = max(
                worst,
                max(abs(a - b) for a, b in zip(completion.logprobs, expected)),
            )
    assert worst < 1e-4, f"max divergence {worst}"


def test_the_shortest_prompt_is_the_one_padding_would_break(model, tokenizer):
    """Same check, isolated to the prompt that receives the most padding."""
    short = min(PROMPTS, key=len)
    batched = sample_completions(
        model, tokenizer, PROMPTS, n=1, temperature=0.0, top_p=1.0,
        max_new_tokens=10, batch_size=4, seed=0, progress=False,
    )[PROMPTS.index(short)][0]
    alone = sample_completions(
        model, tokenizer, [short], n=1, temperature=0.0, top_p=1.0,
        max_new_tokens=10, batch_size=1, seed=0, progress=False,
    )[0][0]
    # Greedy decoding is deterministic, so padding is the only thing that
    # could make these differ.
    assert batched.token_ids == alone.token_ids


# --- the recorder, in isolation ----------------------------------------------


def test_recorder_gathers_the_token_sampled_one_step_earlier():
    """The processor sees a step's scores before the token is drawn from them,
    so it records on the following call, when the draw is visible."""
    recorder = _LogprobRecorder()

    first = torch.tensor([[0.0, 1.0, 2.0]])
    returned = recorder(torch.tensor([[7]]), first)
    assert returned is first, "the processor must not alter the scores"
    assert recorder.steps == [], "nothing is knowable until the draw is visible"

    second = torch.tensor([[3.0, 0.0, 0.0]])
    recorder(torch.tensor([[7, 2]]), second)  # token 2 was drawn from `first`
    recorded = recorder.finish(torch.tensor([[7, 2, 0]]))  # then token 0

    assert recorded.shape == (1, 2)
    assert recorded[0, 0] == pytest.approx(torch.log_softmax(first, -1)[0, 2])
    assert recorded[0, 1] == pytest.approx(torch.log_softmax(second, -1)[0, 0])


def test_recorder_handles_a_single_step():
    recorder = _LogprobRecorder()
    scores = torch.tensor([[0.0, 5.0]])
    recorder(torch.tensor([[9]]), scores)
    recorded = recorder.finish(torch.tensor([[9, 1]]))
    assert recorded.shape == (1, 1)
    assert recorded[0, 0] == pytest.approx(torch.log_softmax(scores, -1)[0, 1])


# --- token accounting --------------------------------------------------------


class CountingTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def completion_from(ids, max_new_tokens, eos=(99,)):
    return _to_completion(
        torch.tensor(ids),
        torch.zeros(len(ids)),
        CountingTokenizer(),
        eos,
        max_new_tokens,
    )


def test_length_stops_at_the_first_eos():
    completion = completion_from([5, 6, 7, 99, 99, 99], max_new_tokens=6)
    assert completion.token_ids == [5, 6, 7, 99]  # the EOS is kept
    assert len(completion.logprobs) == 4
    assert not completion.hit_token_limit


def test_no_eos_at_the_budget_is_a_token_limit_hit():
    completion = completion_from([5, 6, 7, 8], max_new_tokens=4)
    assert completion.hit_token_limit  # this is what costs it 0.1 reward


def test_a_short_row_without_eos_is_not_a_token_limit_hit():
    # The batch only stops early once every sequence has finished, so a row
    # shorter than the budget cannot have been cut off by it.
    assert not completion_from([5, 6], max_new_tokens=384).hit_token_limit


def test_token_limit_is_consistent_with_what_was_generated(model, tokenizer):
    rows = sample_completions(
        model, tokenizer, PROMPTS, n=2, temperature=1.0, top_p=1.0,
        max_new_tokens=8, batch_size=6, seed=3, progress=False,
    )
    for completion in [c for row in rows for c in row]:
        ran_out = len(completion.token_ids) == 8 and EOS not in completion.token_ids
        assert completion.hit_token_limit == ran_out
        assert len(completion.token_ids) == len(completion.logprobs)


# --- which distribution was sampled ------------------------------------------


@pytest.mark.parametrize(
    "temperature,top_p", [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 0.2)]
)
def test_warping_the_sampler_does_not_move_the_recorded_logprobs(
    model, tokenizer, temperature, top_p
):
    """The recorded values are the policy's, not the behaviour distribution's.

    Transformers runs temperature/top-k/top-p after any custom processor, so
    the recorder sees pre-warp scores. That is deliberate: pi_new will later
    come from a training forward pass with no warpers in it, and a ratio whose
    denominator carried a temperature factor its numerator never saw would be
    wrong. The consequence to keep in mind is that these are not the
    distribution the tokens were actually drawn from unless temperature is 1.0
    and top_p is 1.0.
    """
    prompt = PROMPTS[0]
    completion = sample_completions(
        model, tokenizer, [prompt], n=1, temperature=temperature, top_p=top_p,
        max_new_tokens=10, batch_size=1, seed=1, progress=False,
    )[0][0]
    raw = teacher_forcing_logprobs(model, tokenizer, prompt, completion.token_ids)
    assert completion.logprobs == pytest.approx(raw, abs=1e-4)


def test_a_repetition_penalty_would_corrupt_the_recorded_logprobs(model, tokenizer):
    """Guards the other half of the rule: penalties are processors, not
    warpers, so they land before the recorder. This is the failure that cost
    1.61 nats of divergence before generate.py pinned the penalty to 1.0."""
    from transformers import GenerationConfig

    import src.generate as gen

    prompt = PROMPTS[0]
    real = GenerationConfig

    class Penalised(GenerationConfig):
        def __init__(self, **kwargs):
            super().__init__(**{**kwargs, "repetition_penalty": 1.3})

    gen.GenerationConfig = Penalised
    try:
        completion = sample_completions(
            model, tokenizer, [prompt], n=1, temperature=1.0, top_p=1.0,
            max_new_tokens=12, batch_size=1, seed=1, progress=False,
        )[0][0]
    finally:
        gen.GenerationConfig = real
    raw = teacher_forcing_logprobs(model, tokenizer, prompt, completion.token_ids)
    worst = max(abs(a - b) for a, b in zip(completion.logprobs, raw))
    assert worst > 1e-3, (
        "a repetition penalty no longer perturbs the recorded logprobs; if "
        "transformers moved penalties after custom processors, the pinning in "
        "generate.py can be revisited"
    )


def test_greedy_is_the_raw_argmax(model, tokenizer):
    """A leaked repetition_penalty would show up here first."""
    prompt = PROMPTS[0]
    completion = sample_completions(
        model, tokenizer, [prompt], n=1, temperature=0.0, top_p=1.0,
        max_new_tokens=12, batch_size=1, seed=0, progress=False,
    )[0][0]
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full = torch.cat([prompt_ids, torch.tensor([completion.token_ids])], dim=1)
    logits = model(full).logits[0]
    start = prompt_ids.shape[1] - 1
    argmax = logits[start : start + len(completion.token_ids)].argmax(-1).tolist()
    assert argmax == completion.token_ids


# --- guards ------------------------------------------------------------------


def test_completion_rejects_misaligned_logprobs():
    with pytest.raises(ValueError, match="misaligned"):
        Completion(text="x", token_ids=[1, 2], logprobs=[-0.1], hit_token_limit=False)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(n=0, temperature=1.0, top_p=1.0, max_new_tokens=8), "at least 1"),
        (dict(n=2, temperature=0.0, top_p=1.0, max_new_tokens=8), "greedy"),
        (dict(n=1, temperature=1.0, top_p=1.0, max_new_tokens=MAX_SEQ), "no room"),
    ],
)
def test_rejects_impossible_requests(model, tokenizer, kwargs, match):
    with pytest.raises(ValueError, match=match):
        sample_completions(model, tokenizer, PROMPTS[:1], **kwargs, progress=False)


def test_over_long_prompts_warn(model, tokenizer):
    long_prompt = " ".join(["t1"] * (MAX_SEQ - 8 + 5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample_completions(
            model, tokenizer, [long_prompt], n=1, temperature=1.0, top_p=1.0,
            max_new_tokens=8, batch_size=1, progress=False,
        )
    assert any("truncated" in str(w.message) for w in caught)


def test_tokenizer_padding_side_is_restored(model, tokenizer):
    tokenizer.padding_side = "right"
    sample_completions(
        model, tokenizer, PROMPTS[:2], n=1, temperature=1.0, top_p=1.0,
        max_new_tokens=4, batch_size=2, progress=False,
    )
    assert tokenizer.padding_side == "right", "caller's tokenizer was mutated"


# --- the real model ----------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_logprobs_match_teacher_forcing_on_the_real_policy():
    """The same invariant against Qwen2.5-0.5B, in float32 on the GPU.

    The tiny model above cannot catch anything specific to the real
    checkpoint — above all a generation_config that quietly reintroduces
    top_k or a repetition penalty, which is exactly what happened once.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data import format_prompt, load_problems

    name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        policy = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    except Exception as exc:  # offline, or no room for the download
        pytest.skip(f"{name} unavailable: {exc}")
    policy = policy.to("cuda").eval()
    prompt = format_prompt(load_problems("dev")[0], tok)

    completion = sample_completions(
        policy, tok, [prompt], n=1, temperature=1.0, top_p=1.0,
        max_new_tokens=32, batch_size=1, seed=0, progress=False,
    )[0][0]

    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    ids = torch.tensor([completion.token_ids], device="cuda")
    logits = policy(torch.cat([prompt_ids, ids], dim=1)).logits[0]
    start = prompt_ids.shape[1] - 1
    lp = torch.log_softmax(logits[start : start + ids.shape[1]].float(), dim=-1)
    expected = lp.gather(1, ids[0][:, None]).squeeze(1).tolist()

    assert completion.logprobs == pytest.approx(expected, abs=1e-3)
