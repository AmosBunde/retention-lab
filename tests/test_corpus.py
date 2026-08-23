import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention_lab.data.corpus import is_heldout, iter_documents, split_documents


def write_shard(path, texts):
    pq.write_table(pa.table({"text": texts, "meta": [len(t) for t in texts]}), path)


@pytest.fixture
def shards(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    write_shard(a, [f"doc-{i}" for i in range(7)])
    write_shard(b, [f"doc-{i}" for i in range(7, 12)])
    return [a, b]


def test_global_indices_span_shards_in_manifest_order(shards):
    docs = list(iter_documents(shards, batch_rows=3))
    assert [i for i, _ in docs] == list(range(12))
    assert docs[0][1] == "doc-0"
    assert docs[7][1] == "doc-7"


def test_shard_order_defines_identity(shards):
    forward = list(iter_documents(shards))
    reversed_order = list(iter_documents(list(reversed(shards))))
    assert forward != reversed_order  # manifest order is the identity, not file names


def test_split_is_disjoint_and_exhaustive(shards):
    train = list(split_documents(shards, stride=5, want_heldout=False))
    held = list(split_documents(shards, stride=5, want_heldout=True))
    train_ids = {i for i, _ in train}
    held_ids = {i for i, _ in held}
    assert train_ids.isdisjoint(held_ids)
    assert train_ids | held_ids == set(range(12))
    assert held_ids == {0, 5, 10}


def test_heldout_stride_guard():
    with pytest.raises(ValueError):
        is_heldout(3, stride=1)
