"""Adversarial determinism tests over the full data chain.

Field conditions vary read batch sizes, file names, process boundaries, and
restart positions; none of these may change a single token of the training
stream. Every test here builds the chain end to end from parquet fixtures:
documents, held-out split, packing, mixture schedule, batches.
"""

import hashlib
import subprocess
import sys
import textwrap

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention_lab.data.corpus import iter_documents, split_documents
from retention_lab.data.packing import MixtureStream, tokenize_and_pack


def encode(text):
    return [ord(c) % 256 for c in text]


def write_fixture_shards(tmp_path, names=("s0.parquet", "s1.parquet")):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard_index, name in enumerate(names):
        texts = [f"document {shard_index}-{i} " + "body " * 30 for i in range(25)]
        path = tmp_path / name
        pq.write_table(pa.table({"text": texts}), path)
        paths.append(path)
    return paths


def stream_from(paths, seed=13):
    train = (text for _, text in split_documents(paths, stride=7, want_heldout=False))
    packed = {"corpus": tokenize_and_pack(train, encode, block_len=32, eos_id=0)}
    return MixtureStream(packed, {"corpus": 1.0}, seed=seed)


def stream_digest(stream, n_steps=5, batch_size=4):
    digest = hashlib.sha256()
    for step in range(n_steps):
        digest.update(stream.batch(step, batch_size).tobytes())
    return digest.hexdigest()


def test_read_batch_size_cannot_change_the_documents(tmp_path):
    paths = write_fixture_shards(tmp_path)
    variants = [list(iter_documents(paths, batch_rows=n)) for n in (1, 3, 1000)]
    assert variants[0] == variants[1] == variants[2]


def test_disk_names_do_not_matter_manifest_order_does(tmp_path):
    a = write_fixture_shards(tmp_path / "a", names=("s0.parquet", "s1.parquet"))
    b = write_fixture_shards(tmp_path / "b", names=("zz-renamed.parquet", "aa-renamed.parquet"))
    assert stream_digest(stream_from(a)) == stream_digest(stream_from(b))


def test_restart_mid_epoch_continues_bit_identically(tmp_path):
    paths = write_fixture_shards(tmp_path)
    reference = stream_from(paths)
    upfront = [reference.batch(s, 4).copy() for s in range(6)]
    del reference
    resumed = stream_from(paths)
    for step in range(2, 6):
        assert np.array_equal(resumed.batch(step, 4), upfront[step])


def test_cross_process_stream_is_identical(tmp_path):
    paths = write_fixture_shards(tmp_path)
    here = stream_digest(stream_from(paths))
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from tests.test_data_determinism import stream_digest, stream_from
        paths = [Path({str(paths[0])!r}), Path({str(paths[1])!r})]
        print(stream_digest(stream_from(paths)))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        cwd=".",
    )
    assert out.stdout.strip() == here


def test_injected_unseeded_shuffle_is_caught(tmp_path):
    """Meta-test: the digest actually detects a nondeterministic order."""
    paths = write_fixture_shards(tmp_path)
    base = stream_from(paths)
    tampered = stream_from(paths)
    rng = np.random.default_rng()  # deliberately unseeded
    rng.shuffle(tampered.orders["corpus"])
    with pytest.raises(AssertionError):
        assert stream_digest(base) == stream_digest(tampered)
