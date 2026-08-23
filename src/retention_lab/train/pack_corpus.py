"""Tokenize and pack the pinned corpus once per (corpus, block_len), cached.

The packed array is a pure function of the manifest, the tokenizer, and the
block length, so it is cached on disk beside the assets and reused by every
arm; the cache key embeds the corpus name, the split, and the block length.
Teacher-generated shards pack through the same function, which is what
guarantees the E3 mixture consumes them under identical rules.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from retention_lab.data.assets import corpus_shard_paths
from retention_lab.data.corpus import split_documents
from retention_lab.data.packing import tokenize_and_pack


def packed_cache_path(assets_root: Path, name: str, split: str, block_len: int) -> Path:
    return assets_root / "packed" / f"{name}.{split}.b{block_len}.npy"


def pack_split(
    assets_root: Path,
    corpus_name: str,
    corpus_spec: dict,
    encode_fn: Callable[[str], list[int]],
    eos_id: int,
    block_len: int,
    heldout: bool,
) -> np.ndarray:
    split = "heldout" if heldout else "train"
    cache = packed_cache_path(assets_root, corpus_name, split, block_len)
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    shards = corpus_shard_paths(corpus_spec, assets_root, corpus_name)
    missing = [str(p) for p in shards if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"corpus shards not downloaded (run `make assets` first): {missing}"
        )
    texts = (
        text
        for _, text in split_documents(
            shards, int(corpus_spec["heldout_stride"]), want_heldout=heldout
        )
    )
    packed = tokenize_and_pack(texts, encode_fn, block_len, eos_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npy")
    np.save(tmp, packed)
    tmp.replace(cache)
    return np.load(cache, mmap_mode="r")


def pack_generated(
    assets_root: Path,
    shard_dir: Path,
    encode_fn: Callable[[str], list[int]],
    eos_id: int,
    block_len: int,
) -> np.ndarray:
    """Pack teacher-generated parquet shards (provenance verified upstream)."""
    import pyarrow.parquet as pq

    cache = packed_cache_path(assets_root, "teacher-generated", "train", block_len)
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no generated shards in {shard_dir}")
    texts = []
    for shard in shards:
        table = pq.read_table(shard, columns=["text", "source"])
        sources = set(table.column("source").to_pylist())
        if sources != {"teacher_generated"}:
            raise ValueError(f"{shard}: unexpected source labels {sources}")
        texts.extend(table.column("text").to_pylist())
    packed = tokenize_and_pack(iter(texts), encode_fn, block_len, eos_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npy")
    np.save(tmp, packed)
    tmp.replace(cache)
    return np.load(cache, mmap_mode="r")
