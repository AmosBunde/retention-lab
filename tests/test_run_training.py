import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention_lab.train.pack_corpus import pack_generated, pack_split, packed_cache_path
from retention_lab.train.run_training import count_reclaim


def encode(text):
    return [ord(c) % 199 for c in text]


def corpus_fixture(tmp_path):
    shard_rel = "sample/s0.parquet"
    shard = tmp_path / "corpus" / "fix" / shard_rel
    shard.parent.mkdir(parents=True)
    texts = [f"doc {i} " + "content " * 40 for i in range(30)]
    pq.write_table(pa.table({"text": texts}), shard)
    spec = {"heldout_stride": 5, "files": {shard_rel: {"size": 1, "sha256": "x"}}}
    return spec


def test_pack_split_caches_and_reuses(tmp_path):
    spec = corpus_fixture(tmp_path)
    a = pack_split(tmp_path, "fix", spec, encode, eos_id=0, block_len=64, heldout=False)
    cache = packed_cache_path(tmp_path, "fix", "train", 64)
    assert cache.exists()
    stamp = cache.stat().st_mtime_ns
    b = pack_split(tmp_path, "fix", spec, encode, eos_id=0, block_len=64, heldout=False)
    assert cache.stat().st_mtime_ns == stamp, "second call must hit the cache"
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_pack_split_separates_heldout(tmp_path):
    spec = corpus_fixture(tmp_path)
    train = pack_split(tmp_path, "fix", spec, encode, eos_id=0, block_len=64, heldout=False)
    held = pack_split(tmp_path, "fix", spec, encode, eos_id=0, block_len=64, heldout=True)
    assert len(train) > len(held) > 0


def test_pack_split_demands_downloaded_shards(tmp_path):
    spec = {"heldout_stride": 5, "files": {"missing.parquet": {"size": 1, "sha256": "x"}}}
    with pytest.raises(FileNotFoundError, match="make assets"):
        pack_split(tmp_path, "nope", spec, encode, eos_id=0, block_len=64, heldout=False)


def test_pack_generated_verifies_source_labels(tmp_path):
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    texts = ["generated " + "token " * 30 for _ in range(10)]
    pq.write_table(
        pa.table({"text": texts, "source": ["teacher_generated"] * 10}),
        gen_dir / "shard-000.parquet",
    )
    packed = pack_generated(tmp_path, gen_dir, encode, eos_id=0, block_len=64)
    assert len(packed) > 0

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    pq.write_table(
        pa.table({"text": texts, "source": ["corpus"] * 10}), bad_dir / "shard-000.parquet"
    )
    with pytest.raises(ValueError, match="source labels"):
        pack_generated(tmp_path / "other", bad_dir, encode, eos_id=0, block_len=64)


def test_reclaim_counter_increments_only_on_resume(tmp_path):
    assert count_reclaim(tmp_path, resumed=False) == 0
    assert count_reclaim(tmp_path, resumed=True) == 1
    assert count_reclaim(tmp_path, resumed=True) == 2
    assert count_reclaim(tmp_path, resumed=False) == 2
