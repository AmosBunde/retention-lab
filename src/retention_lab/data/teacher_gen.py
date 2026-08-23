"""Teacher-generated data with provenance that survives into any mixture.

The pipeline is generic over a ``generate_fn`` so the whole path (prompt
selection, provenance labeling, shard writing) is CPU-testable with a stub;
the real function wraps the pinned teacher's ``generate`` and runs only on
approved GPU time. Prompts are the first ``prompt_tokens`` tokens of
training-split documents chosen by a seeded permutation prefix, so the
prompt set is a pure function of the manifest and this config.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from retention_lab.utils.config import config_hash
from retention_lab.utils.seeding import numpy_rng

SOURCE_LABEL = "teacher_generated"

GenerateFn = Callable[[list[list[int]]], list[list[int]]]
"""Maps a batch of prompt token lists to continuation token lists."""


def select_prompt_indices(n_train_docs: int, n_documents: int, seed: int) -> list[int]:
    """Deterministic choice of which training documents seed prompts."""
    if n_documents > n_train_docs:
        raise ValueError(
            f"asked for {n_documents} prompts from {n_train_docs} training documents"
        )
    rng = numpy_rng(seed, "teachergen-prompts")
    return [int(i) for i in rng.permutation(n_train_docs)[:n_documents]]


def generate_documents(
    prompts: Sequence[tuple[int, list[int]]],
    generate_fn: GenerateFn,
    decode_fn: Callable[[list[int]], str],
    cfg: dict,
    batch_size: int = 32,
) -> Iterator[dict]:
    """Yield labeled document rows: prompt tokens plus teacher continuation."""
    sampling_hash = config_hash(cfg["sampling"])
    teacher = cfg["teacher"]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        continuations = generate_fn([tokens for _, tokens in batch])
        if len(continuations) != len(batch):
            raise ValueError("generate_fn returned a mismatched batch")
        for (doc_index, prompt_tokens), continuation in zip(batch, continuations, strict=True):
            yield {
                "text": decode_fn(list(prompt_tokens) + list(continuation)),
                "source": SOURCE_LABEL,
                "teacher_repo": teacher["repo"],
                "teacher_revision": teacher["revision"],
                "sampling_hash": sampling_hash,
                "prompt_doc_index": doc_index,
                "n_prompt_tokens": len(prompt_tokens),
                "n_generated_tokens": len(continuation),
            }


def write_generated_shard(rows: Sequence[dict], out_path: Path) -> None:
    if not rows:
        raise ValueError("refusing to write an empty generated shard")
    for row in rows:
        if row["source"] != SOURCE_LABEL:
            raise ValueError("every generated row must be labeled teacher_generated")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    pq.write_table(table, out_path)
