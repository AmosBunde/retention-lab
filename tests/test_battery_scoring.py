import math

import pytest
import torch

from retention_lab.battery.protocol import ByteTokenizer, LLResult, TorchCausalLM
from retention_lab.battery.scoring import (
    GreedyItem,
    MCItem,
    group_by_capability,
    score_bits_per_byte,
    score_greedy,
    score_multiple_choice,
)
from retention_lab.models.toy import build_toy_lm
from retention_lab.utils.seeding import torch_generator

TOKENIZER = ByteTokenizer(32)


class PerByteLM:
    """Fake LM: log-likelihood is a fixed per-byte score by first character.

    Choices starting with 'a' score -1.0 per byte, everything else -2.0 per
    byte; greedy is true only for the continuation \"x\".
    """

    def loglikelihood(self, context, continuation):
        per_byte = -1.0 if continuation.startswith("a") else -2.0
        n = len(continuation.encode("utf-8"))
        return LLResult(logprob=per_byte * n, greedy=continuation == "x", n_tokens=n)

    def loglikelihood_tokens(self, tokens):
        return -math.log(32) * (len(tokens) - 1)


class ConstLogitsModel(torch.nn.Module):
    """Uniform next-token distribution regardless of context."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], self.vocab_size)


def test_length_normalization_prefers_per_byte_quality():
    # Total logprob favors "b" (-2 versus -4); per-byte favors "aaaa" (-1 versus -2).
    item = MCItem(context="q: ", choices=("aaaa", "b"), gold=0)
    result = score_multiple_choice(PerByteLM(), "fixture", [item])
    assert result.value == 1.0


def test_tie_breaks_to_lowest_index():
    item = MCItem(context="q: ", choices=("apple", "acorn"), gold=1)
    result = score_multiple_choice(PerByteLM(), "fixture", [item])
    assert result.value == 0.0  # index 0 wins the tie, gold is 1


def test_mc_gold_out_of_range_raises():
    with pytest.raises(ValueError):
        score_multiple_choice(PerByteLM(), "fixture", [MCItem("c", ("a",), 3)])


def test_greedy_scorer_uses_greedy_flag():
    items = [GreedyItem("ctx", "x"), GreedyItem("ctx", "y")]
    result = score_greedy(PerByteLM(), "fixture", items)
    assert result.per_item == (1.0, 0.0)
    assert result.value == 0.5


def test_bpb_uniform_model_is_exact():
    docs = ["hello world", "retention"]
    result = score_bits_per_byte(PerByteLM(), "fixture", docs, TOKENIZER, vocab_size=32)
    n = [len(d.encode("utf-8")) for d in docs]
    expected = (sum(x - 1 for x in n) * math.log(32)) / (math.log(2) * sum(n))
    assert result.value == pytest.approx(expected, abs=1e-12)
    # PerByteLM is exactly the uniform model over 32 tokens, so the chance
    # level must equal the measured value.
    assert result.chance == pytest.approx(result.value, abs=1e-12)


def test_mc_chance_averages_choice_counts():
    from retention_lab.battery.scoring import score_multiple_choice as smc

    items = [
        MCItem("c", ("a", "b"), 0),
        MCItem("c", ("a", "b", "cc", "d"), 0),
    ]
    result = smc(PerByteLM(), "fixture", items)
    assert result.chance == pytest.approx((0.5 + 0.25) / 2)


def test_context_choice_prefers_likelier_context():
    from retention_lab.battery.scoring import ContextChoiceItem, score_context_choice

    class ContextLM(PerByteLM):
        def loglikelihood(self, context, continuation):
            base = super().loglikelihood(context, continuation)
            bonus = 1.0 if context.endswith("good") else 0.0
            return LLResult(base.logprob + bonus, base.greedy, base.n_tokens)

    item = ContextChoiceItem(("prefix good", "prefix bad"), " tail", gold=0)
    result = score_context_choice(ContextLM(), "fixture", [item])
    assert result.value == 1.0
    assert result.chance == 0.5


def test_chunked_token_scoring_counts_each_token_once():
    lm_small = TorchCausalLM(ConstLogitsModel(32), TOKENIZER, max_len=5)
    lm_large = TorchCausalLM(ConstLogitsModel(32), TOKENIZER, max_len=100)
    tokens = TOKENIZER.encode("a deterministic sentence for chunking")
    expected = -(len(tokens) - 1) * math.log(32)
    assert lm_small.loglikelihood_tokens(tokens) == pytest.approx(expected, rel=1e-6)
    assert lm_large.loglikelihood_tokens(tokens) == pytest.approx(expected, rel=1e-6)


def test_torch_lm_is_deterministic_on_toy_model():
    block = {"vocab_size": 32, "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 48}
    model = build_toy_lm(block, torch_generator(5, "model-init"))
    lm = TorchCausalLM(model, TOKENIZER, max_len=48)
    a = lm.loglikelihood("the corpus slice", " is pinned")
    b = lm.loglikelihood("the corpus slice", " is pinned")
    assert a == b
    assert a.n_tokens == len(b" is pinned")


def test_torch_lm_truncates_left_but_keeps_continuation():
    block = {"vocab_size": 32, "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 16}
    model = build_toy_lm(block, torch_generator(5, "model-init"))
    lm = TorchCausalLM(model, TOKENIZER, max_len=16)
    result = lm.loglikelihood("a very long context " * 10, "tail")
    assert result.n_tokens == 4


def test_group_by_capability_requires_all_tasks():
    mc = score_multiple_choice(PerByteLM(), "t1", [MCItem("c", ("a", "b"), 0)])
    grouped = group_by_capability([mc], {"cap": ["t1"]})
    assert grouped["cap"][0].task == "t1"
    with pytest.raises(ValueError):
        group_by_capability([mc], {"cap": ["t1", "t2"]})
