"""The language-model protocol the battery scores against.

The battery never touches model internals: everything it needs is expressed
as log-likelihoods of continuations given contexts, plus whether each
continuation token would have been the greedy choice. Real students and the
teacher satisfy the protocol through ``TorchCausalLM``; CI satisfies it with
the toy model and the byte tokenizer, so the scoring machinery is exercised
without any downloaded asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn import functional as F


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class ByteTokenizer:
    """Deterministic bytes-to-ids tokenizer for CI-scale models.

    Each UTF-8 byte maps to ``byte % vocab_size``. This is not a linguistic
    tokenizer; it exists so the scoring path runs unchanged on the toy model.
    One token per byte also makes bits-per-byte assertions exact in tests.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]


@dataclass(frozen=True)
class LLResult:
    logprob: float
    greedy: bool
    n_tokens: int


class LM(Protocol):
    def loglikelihood(self, context: str, continuation: str) -> LLResult: ...

    def loglikelihood_tokens(self, tokens: list[int]) -> float: ...


class TorchCausalLM:
    """Wrap a torch causal LM (logits = model(tokens)) and a tokenizer.

    Continuation tokens are computed as ``encode(context + continuation)``
    minus the prefix ``encode(context)``, so tokenizers that merge across the
    boundary are handled the same way the model saw text during training.
    Inputs longer than ``max_len`` are truncated on the left; continuation
    tokens are never truncated.
    """

    def __init__(
        self, model: torch.nn.Module, tokenizer: Tokenizer, max_len: int, device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    @torch.no_grad()
    def _token_logprobs(self, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position log p(token[i] | tokens[:i]) and greedy flags, i >= 1."""
        ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        self.model.eval()
        logits = self.model(ids[:, :-1]).float()
        logprobs = F.log_softmax(logits, dim=-1)[0].cpu()
        targets = ids[0, 1:]
        picked = logprobs[torch.arange(targets.shape[0]), targets]
        greedy = logprobs.argmax(dim=-1) == targets
        return picked, greedy

    def loglikelihood(self, context: str, continuation: str) -> LLResult:
        ctx_tokens = self.tokenizer.encode(context)
        all_tokens = self.tokenizer.encode(context + continuation)
        n_cont = len(all_tokens) - len(ctx_tokens)
        if n_cont <= 0:
            raise ValueError("continuation produced no tokens")
        if len(all_tokens) > self.max_len:
            all_tokens = all_tokens[-self.max_len :]
            if n_cont >= len(all_tokens):
                raise ValueError("continuation alone exceeds the model context")
        picked, greedy = self._token_logprobs(all_tokens)
        return LLResult(
            logprob=float(picked[-n_cont:].sum()),
            greedy=bool(greedy[-n_cont:].all()),
            n_tokens=n_cont,
        )

    def loglikelihood_tokens(self, tokens: list[int]) -> float:
        """Total log-likelihood of tokens[1:] given their prefixes, chunked.

        Documents longer than ``max_len`` are scored in non-overlapping
        chunks with one token of carried context, so every token except the
        very first is conditioned and counted exactly once.
        """
        total = 0.0
        start = 0
        while start < len(tokens) - 1:
            chunk = tokens[start : start + self.max_len]
            picked, _ = self._token_logprobs(chunk)
            total += float(picked.sum())
            start += self.max_len - 1
        return total
