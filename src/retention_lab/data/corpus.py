"""Deterministic document iteration over the pinned corpus shards.

Documents get a global index by their position across shards in manifest
order; that index is the identity used for the held-out split, so the split
is a pure function of the manifest and the stride, independent of machine,
worker count, or read order. Held-out documents serve training-time
validation monitoring only; the frozen battery's bits-per-byte task uses
WikiText-2 by design, so nothing about this corpus can move the frozen
scoreboard.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

TEXT_COLUMN = "text"


def iter_documents(
    shard_paths: list[Path], batch_rows: int = 1024
) -> Iterator[tuple[int, str]]:
    """Yield (global_index, text) across shards in manifest order."""
    index = 0
    for shard in shard_paths:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=[TEXT_COLUMN]):
            for text in batch.column(TEXT_COLUMN).to_pylist():
                yield index, text
                index += 1


def is_heldout(global_index: int, stride: int) -> bool:
    if stride < 2:
        raise ValueError("held-out stride must be at least 2")
    return global_index % stride == 0


def split_documents(
    shard_paths: list[Path], stride: int, want_heldout: bool
) -> Iterator[tuple[int, str]]:
    """Training documents (want_heldout=False) or held-out ones (True)."""
    for index, text in iter_documents(shard_paths):
        if is_heldout(index, stride) == want_heldout:
            yield index, text
