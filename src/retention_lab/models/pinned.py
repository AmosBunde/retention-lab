"""Pinned Hugging Face causal models adapted to the battery protocol.

Loading always demands an explicit revision SHA: there is no default and no
``main``, because a moving revision would silently unfreeze the scoreboard
denominators. The tokenizer adapter exposes exactly what the battery needs
(``encode`` and ``vocab_size``) so the scoring engine stays ignorant of the
transformers API.
"""

from __future__ import annotations

import re

import torch

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class HFTokenizerAdapter:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.vocab_size = int(tokenizer.vocab_size)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)


class HFLogitsAdapter(torch.nn.Module):
    """Expose logits-only forward, the shape the battery protocol expects."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens).logits


def load_pinned_causal_lm(
    repo: str, revision: str, device: str = "cpu", dtype: str = "float32"
) -> tuple[HFLogitsAdapter, HFTokenizerAdapter, int]:
    """Load a causal LM and tokenizer at an exact revision.

    Returns the logits adapter, the tokenizer adapter, and the model's
    maximum position count, which callers use as the scoring context limit.
    """
    if not FULL_SHA.match(revision):
        raise ValueError(
            f"revision must be a full 40-hex commit SHA, got {revision!r}; "
            "moving refs would silently unfreeze the denominators"
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, dtype=getattr(torch, dtype)
    )
    model.to(device)
    model.eval()
    max_len = int(model.config.max_position_embeddings)
    return HFLogitsAdapter(model), HFTokenizerAdapter(tokenizer), max_len
