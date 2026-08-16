"""Tests for the log-probability helpers, and the contract for the DPO loss.

The central test feeds the model logits whose softmax is known exactly, so the
expected sequence log-probability can be worked out by hand rather than by
running the same code twice. That matters more here than anywhere else in the
project: prompt masking and the off-by-one both fail *quietly*, producing
numbers that are wrong but plausible, and a test written against the
implementation's own output would agree with either.

The `dpo_loss` tests at the bottom are skipped until that function exists —
they are its specification (PROJECT.md §3a marks it [OWNER WRITES]).
"""

import math
from types import SimpleNamespace

import pytest
import torch

from src import losses
from src.losses import IGNORE_INDEX, reference_logprobs, sequence_logprobs

peft = pytest.importorskip("peft")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402


class FixedLogits(torch.nn.Module):
    """A model whose logits are whatever the test says they are."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.logits)


def logits_for(probability_rows):
    """Logits whose softmax is exactly the given probabilities.

    softmax(log p) == p, so writing the log of the intended distribution makes
    every expected log-probability a plain log() the test can state literally.
    """
    return torch.log(torch.tensor([probability_rows], dtype=torch.float64))


# Four positions, vocabulary of four. Row t is the distribution used to predict
# the token at position t+1.
ROWS = [
    [0.1, 0.2, 0.3, 0.4],      # predicts position 1
    [0.25, 0.25, 0.25, 0.25],  # predicts position 2
    [0.5, 0.2, 0.2, 0.1],      # predicts position 3
    [0.7, 0.1, 0.1, 0.1],      # predicts position 4, which does not exist
]
# Position 0 is the prompt; the completion is the tokens 2, 0, 1.
TOKENS = [[3, 2, 0, 1]]


# --- the hand-computed example -----------------------------------------------


def test_three_token_completion_matches_a_hand_computed_value():
    """log P(2 | ...) + log P(0 | ...) + log P(1 | ...), worked out by hand.

    Token at position 1 is 2, scored by row 0  -> log 0.3
    Token at position 2 is 0, scored by row 1  -> log 0.25
    Token at position 3 is 1, scored by row 2  -> log 0.2
    Row 3 predicts a fifth token and must not be used at all.
    """
    model = FixedLogits(logits_for(ROWS))
    input_ids = torch.tensor(TOKENS)

    total = sequence_logprobs(
        model, input_ids, input_ids.clone(), torch.tensor([1])
    )

    expected = math.log(0.3) + math.log(0.25) + math.log(0.2)
    assert expected == pytest.approx(math.log(0.015))  # 0.3 * 0.25 * 0.2
    assert total.shape == (1,)
    assert total.item() == pytest.approx(expected, abs=1e-9)


def test_prompt_tokens_are_excluded():
    """With two prompt tokens, position 1 stops being scored.

    This is the failure that matters: including the prompt adds a large,
    nearly constant term to both sides of every pair and shrinks the margin
    without ever looking wrong.
    """
    model = FixedLogits(logits_for(ROWS))
    input_ids = torch.tensor(TOKENS)

    total = sequence_logprobs(model, input_ids, input_ids.clone(), torch.tensor([2]))

    assert total.item() == pytest.approx(math.log(0.25) + math.log(0.2), abs=1e-9)


def test_a_longer_prompt_scores_strictly_less():
    model = FixedLogits(logits_for(ROWS))
    input_ids = torch.tensor(TOKENS)
    totals = [
        sequence_logprobs(model, input_ids, input_ids.clone(), torch.tensor([n])).item()
        for n in (1, 2, 3)
    ]
    assert totals[0] < totals[1] < totals[2] < 0


def test_scoring_the_whole_sequence_still_ignores_position_zero():
    """Nothing predicts the first token, so prompt_lens=0 cannot score it."""
    model = FixedLogits(logits_for(ROWS))
    input_ids = torch.tensor(TOKENS)
    total = sequence_logprobs(model, input_ids, input_ids.clone(), torch.tensor([0]))
    assert total.item() == pytest.approx(
        math.log(0.3) + math.log(0.25) + math.log(0.2), abs=1e-9
    )


def test_ignore_index_masks_padding():
    model = FixedLogits(logits_for(ROWS))
    input_ids = torch.tensor(TOKENS)
    labels = torch.tensor([[3, 2, 0, IGNORE_INDEX]])  # last token is padding

    total = sequence_logprobs(model, input_ids, labels, torch.tensor([1]))

    assert total.item() == pytest.approx(math.log(0.3) + math.log(0.25), abs=1e-9)


def test_padding_positions_do_not_gather_out_of_bounds():
    """IGNORE_INDEX is -100; gathering with it directly would throw."""
    model = FixedLogits(logits_for(ROWS))
    input_ids = torch.tensor(TOKENS)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    assert sequence_logprobs(model, input_ids, labels, torch.tensor([1])).item() == 0.0


def test_each_sequence_gets_its_own_prompt_length():
    rows = torch.log(torch.tensor([ROWS, ROWS], dtype=torch.float64))
    model = FixedLogits(rows)
    input_ids = torch.tensor([[3, 2, 0, 1], [3, 2, 0, 1]])

    totals = sequence_logprobs(model, input_ids, input_ids.clone(), torch.tensor([1, 3]))

    assert totals[0].item() == pytest.approx(
        math.log(0.3) + math.log(0.25) + math.log(0.2), abs=1e-9
    )
    assert totals[1].item() == pytest.approx(math.log(0.2), abs=1e-9)


def test_shape_mismatches_are_rejected():
    model = FixedLogits(logits_for(ROWS))
    with pytest.raises(ValueError, match="labels"):
        sequence_logprobs(
            model, torch.tensor(TOKENS), torch.tensor([[1, 2]]), torch.tensor([1])
        )
    with pytest.raises(ValueError, match="one entry per sequence"):
        sequence_logprobs(
            model, torch.tensor(TOKENS), torch.tensor(TOKENS), torch.tensor([1, 1])
        )


# --- against a real adapter ---------------------------------------------------


@pytest.fixture(scope="module")
def lora_model():
    """A 2-layer Qwen2 with a LoRA adapter, on the CPU.

    lora_B initialises to zero, which makes the adapter an exact identity — so
    it is perturbed here. Otherwise every test comparing policy to reference
    would pass for the wrong reason.
    """
    config = Qwen2Config(
        vocab_size=32, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                        task_type="CAUSAL_LM"),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.05)
    return model.eval()


BATCH = torch.tensor([[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]])
PROMPT_LENS = torch.tensor([2, 2])


def test_reference_differs_from_policy(lora_model):
    """If these are equal the adapter is not doing anything, and DPO's loss
    would sit at a constant -log(0.5) while appearing to run fine."""
    policy = sequence_logprobs(lora_model, BATCH, BATCH.clone(), PROMPT_LENS)
    reference = reference_logprobs(lora_model, BATCH, BATCH.clone(), PROMPT_LENS)
    assert not torch.allclose(policy, reference, atol=1e-6)


def test_reference_is_detached(lora_model):
    reference = reference_logprobs(lora_model, BATCH, BATCH.clone(), PROMPT_LENS)
    assert not reference.requires_grad


def test_adapter_is_re_enabled_afterwards(lora_model):
    reference_logprobs(lora_model, BATCH, BATCH.clone(), PROMPT_LENS)
    assert not losses._adapter_is_disabled(lora_model)
    # And the policy pass that follows must again see the adapter.
    after = sequence_logprobs(lora_model, BATCH, BATCH.clone(), PROMPT_LENS)
    reference = reference_logprobs(lora_model, BATCH, BATCH.clone(), PROMPT_LENS)
    assert not torch.allclose(after, reference, atol=1e-6)


def test_adapter_is_re_enabled_even_if_scoring_raises(lora_model):
    with pytest.raises(ValueError):
        reference_logprobs(lora_model, BATCH, torch.tensor([[1, 2]]), PROMPT_LENS)
    assert not losses._adapter_is_disabled(lora_model)


def test_a_plain_model_is_rejected():
    with pytest.raises(TypeError, match="PEFT"):
        reference_logprobs(
            FixedLogits(logits_for(ROWS)), torch.tensor(TOKENS),
            torch.tensor(TOKENS), torch.tensor([1]),
        )


def test_gradient_reaches_only_the_adapter(lora_model):
    """PROJECT.md §3a: gradient flows only to LoRA params."""
    lora_model.zero_grad(set_to_none=True)
    sequence_logprobs(lora_model, BATCH, BATCH.clone(), PROMPT_LENS).sum().backward()

    with_grad = {n for n, p in lora_model.named_parameters() if p.grad is not None}
    assert with_grad, "nothing received a gradient at all"
    assert all("lora_" in name for name in with_grad), sorted(with_grad)[:5]
    trainable = {n for n, p in lora_model.named_parameters() if p.requires_grad}
    assert with_grad == trainable
    lora_model.zero_grad(set_to_none=True)


# --- the contract for the loss the owner writes -------------------------------

dpo_loss = getattr(losses, "dpo_loss", None)

pytestmark_reason = (
    "src.losses.dpo_loss is not written yet — these tests are its contract "
    "(PROJECT.md §3a marks it [OWNER WRITES]). They activate on definition."
)


@pytest.mark.skipif(dpo_loss is None, reason=pytestmark_reason)
class TestDpoLossContract:
    def test_identical_policy_and_reference_gives_minus_log_half(self):
        """The one number that proves the reference is actually being used.

        If disable_adapter() silently no-ops, every real batch also lands on
        exactly this value — so seeing -log(0.5) in a training log is a red
        flag, not a green one.
        """
        zeros = torch.zeros(4)
        loss, metrics = dpo_loss(zeros, zeros.clone(), zeros.clone(), zeros.clone(), beta=0.1)
        assert loss.item() == pytest.approx(-math.log(0.5), abs=1e-6)
        assert metrics["reward_margin"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["logp_chosen"] == pytest.approx(0.0, abs=1e-9)

    def test_loss_falls_toward_zero_as_the_margin_grows(self):
        reference_chosen = torch.zeros(1)
        reference_rejected = torch.zeros(1)
        previous = None
        for margin in (0.0, 1.0, 5.0, 20.0):
            loss = dpo_loss(
                torch.tensor([margin]), torch.zeros(1),
                reference_chosen, reference_rejected, beta=0.5,
            )[0].item()
            if previous is not None:
                assert loss < previous
            previous = loss
        assert previous == pytest.approx(0.0, abs=1e-3)

    def test_loss_grows_when_the_margin_inverts(self):
        zeros = torch.zeros(1)
        inverted = dpo_loss(
            torch.tensor([-20.0]), zeros.clone(), zeros.clone(), zeros.clone(), beta=0.5
        )[0].item()
        assert math.isfinite(inverted), "logsigmoid underflowed"
        assert inverted > 5.0

    def test_beta_scales_the_margin(self):
        args = (torch.tensor([2.0]), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        assert dpo_loss(*args, beta=0.5)[0].item() < dpo_loss(*args, beta=0.05)[0].item()

    def test_metrics_carry_the_absolute_levels_not_just_the_gap(self):
        """Likelihood displacement is invisible in loss and margin alike."""
        loss, metrics = dpo_loss(
            torch.tensor([-30.0]), torch.tensor([-40.0]),
            torch.tensor([-10.0]), torch.tensor([-20.0]), beta=0.1,
        )
        assert metrics["reward_margin"] == pytest.approx(0.0, abs=1e-6)
        # Both levels collapsed by 20 nats while the margin held at zero.
        assert metrics["logp_chosen"] == pytest.approx(-30.0)
        assert metrics["logp_rejected"] == pytest.approx(-40.0)
        assert set(metrics) >= {
            "loss", "reward_chosen", "reward_rejected", "reward_margin",
            "reward_accuracy", "logp_chosen", "logp_rejected",
        }

    def test_gradient_reaches_the_policy_terms_only(self):
        policy_chosen = torch.tensor([1.0], requires_grad=True)
        policy_rejected = torch.tensor([0.0], requires_grad=True)
        ref_chosen = torch.tensor([0.5], requires_grad=True)
        ref_rejected = torch.tensor([0.5], requires_grad=True)
        loss, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
        loss.backward()
        assert policy_chosen.grad is not None and policy_rejected.grad is not None
        # Raising chosen must lower the loss; raising rejected must raise it.
        assert policy_chosen.grad.item() < 0 < policy_rejected.grad.item()
