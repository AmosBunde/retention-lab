"""Deterministic packing and mixture scheduling.

The training stream must be a pure function of (packed sources, mixture
proportions, seed), addressable by index, so that paired experiment arms see
identical batches and a resumed run continues at exactly the sequence it
stopped before. The design is two-phase:

1. **Packing** concatenates documents (with an EOS separator) into blocks of
   ``block_len`` tokens, stored as one array per source. Packing order is
   the document order handed in, which upstream fixes by manifest identity.
2. **Scheduling** decides, for every global block index, which source it
   comes from and which of that source's blocks it is. Counts per source
   follow the mixture proportions by largest remainder over the usable
   total; the interleaving and each source's internal order are seeded
   permutations. Everything is recomputable from the config, so the resume
   cursor is a single integer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from retention_lab.utils.seeding import numpy_rng


def tokenize_and_pack(
    texts: Iterable[str],
    encode_fn: Callable[[str], list[int]],
    block_len: int,
    eos_id: int,
) -> np.ndarray:
    """Concatenate encoded documents, EOS-separated, into (n, block_len)."""
    buffer: list[int] = []
    blocks: list[list[int]] = []
    for text in texts:
        buffer.extend(encode_fn(text))
        buffer.append(eos_id)
        while len(buffer) >= block_len:
            blocks.append(buffer[:block_len])
            buffer = buffer[block_len:]
    if not blocks:
        raise ValueError("sources produced fewer tokens than one block")
    return np.asarray(blocks, dtype=np.int64)


def mixture_counts(available: dict[str, int], proportions: dict[str, float]) -> dict[str, int]:
    """Blocks to draw per source: largest-remainder split of the usable total.

    The usable total is limited by the scarcest source relative to its
    proportion, so no source is asked for more blocks than it has.
    """
    if set(available) != set(proportions):
        raise ValueError(
            f"sources differ: packed={sorted(available)} mixture={sorted(proportions)}"
        )
    if abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError(f"proportions must sum to 1, got {sum(proportions.values())}")
    if any(p <= 0 for p in proportions.values()):
        raise ValueError("every mixture proportion must be positive")
    total = min(int(available[s] / proportions[s]) for s in proportions)
    exact = {s: total * proportions[s] for s in proportions}
    counts = {s: int(exact[s]) for s in proportions}
    remainders = sorted(proportions, key=lambda s: exact[s] - counts[s], reverse=True)
    for source in remainders[: total - sum(counts.values())]:
        counts[source] += 1
    return counts


class MixtureStream:
    """Index-addressable mixed stream over packed sources."""

    def __init__(self, packed: dict[str, np.ndarray], proportions: dict[str, float], seed: int):
        self.packed = packed
        counts = mixture_counts({s: len(a) for s, a in packed.items()}, proportions)
        names = sorted(counts)
        schedule = np.concatenate(
            [np.full(counts[s], i, dtype=np.int32) for i, s in enumerate(names)]
        )
        numpy_rng(seed, "mixture-schedule").shuffle(schedule)
        self.names = names
        self.schedule = schedule
        # Within-source position of each schedule entry: the k-th occurrence
        # of a source reads that source's permuted block number k.
        self.occurrence = np.zeros(len(schedule), dtype=np.int64)
        for i in range(len(names)):
            mask = schedule == i
            self.occurrence[mask] = np.arange(mask.sum())
        self.orders = {
            s: numpy_rng(seed, f"block-order:{s}").permutation(counts[s]) for s in names
        }

    def __len__(self) -> int:
        return len(self.schedule)

    def block(self, index: int) -> np.ndarray:
        name = self.names[self.schedule[index]]
        return self.packed[name][self.orders[name][self.occurrence[index]]]

    def source_of(self, index: int) -> str:
        return self.names[self.schedule[index]]

    def batch(self, step: int, batch_size: int) -> np.ndarray:
        """Batch for optimizer step ``step``; raises past the end of data."""
        start = step * batch_size
        if start + batch_size > len(self.schedule):
            raise IndexError(
                f"step {step} needs blocks up to {start + batch_size}, "
                f"stream has {len(self.schedule)}"
            )
        return np.stack([self.block(i) for i in range(start, start + batch_size)])
