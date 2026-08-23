"""Synthetic corpus for the tiny path.

The tiny path must run without network access, so it trains on generated
token sequences instead of a downloaded corpus. Sequences follow a simple
second-order Markov pattern, which gives a language model a real signal to
learn: the loss on this corpus decreases within a few dozen steps, and the
generator is a pure function of the shared seed.
"""

from __future__ import annotations

import numpy as np
import torch

from retention_lab.utils.seeding import numpy_rng


def synthetic_tokens(seed: int, vocab_size: int, n_tokens: int) -> torch.Tensor:
    """Generate one long deterministic token stream with learnable structure."""
    rng = numpy_rng(seed, "synthetic-corpus")
    # A random but fixed second-order transition table: the next token depends
    # on the previous two. The sharpening factor keeps the conditional entropy
    # near one nat, low enough that a toy model shows clear loss decrease
    # within the smoke budget.
    logits = rng.normal(size=(vocab_size, vocab_size, vocab_size)) * 4.0
    table = np.exp(logits - logits.max(axis=-1, keepdims=True))
    table /= table.sum(axis=-1, keepdims=True)
    out = np.empty(n_tokens, dtype=np.int64)
    out[0], out[1] = rng.integers(0, vocab_size, size=2)
    for i in range(2, n_tokens):
        out[i] = rng.choice(vocab_size, p=table[out[i - 2], out[i - 1]])
    return torch.from_numpy(out)


def pack_batches(
    stream: torch.Tensor, seq_len: int, batch_size: int, seed: int
) -> list[torch.Tensor]:
    """Cut the stream into sequences and batch them in a seed-pinned order."""
    n_seq = (stream.shape[0] - 1) // seq_len
    inputs = stream[: n_seq * seq_len + 1]
    windows = inputs.unfold(0, seq_len + 1, seq_len)
    order = numpy_rng(seed, "packing-order").permutation(n_seq)
    batches = []
    for start in range(0, n_seq - batch_size + 1, batch_size):
        idx = torch.from_numpy(order[start : start + batch_size].copy())
        batches.append(windows[idx])
    return batches
