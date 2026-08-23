import numpy as np
import pytest

from retention_lab.data.packing import MixtureStream, mixture_counts, tokenize_and_pack

ENC = [ord(c) for c in "x"]


def encode(text):
    return [ord(c) % 256 for c in text]


def make_packed(seed_char, n_blocks, block_len=8):
    texts = [seed_char * (block_len * 2) for _ in range(n_blocks)]
    packed = tokenize_and_pack(texts, encode, block_len, eos_id=0)
    return packed[:n_blocks]


def test_packing_preserves_document_bytes_and_separators():
    packed = tokenize_and_pack(["abc", "de"], encode, block_len=3, eos_id=0)
    flat = packed.reshape(-1).tolist()
    expected = [ord(c) % 256 for c in "abc"] + [0] + [ord(c) % 256 for c in "de"] + [0]
    assert flat == expected[: len(flat)]


def test_packing_requires_at_least_one_block():
    with pytest.raises(ValueError):
        tokenize_and_pack(["a"], encode, block_len=10, eos_id=0)


def test_mixture_counts_largest_remainder_and_scarcity():
    counts = mixture_counts({"a": 70, "b": 30}, {"a": 0.7, "b": 0.3})
    assert counts == {"a": 70, "b": 30}
    # Scarce source limits the usable total: b has only 10 blocks at 30 percent.
    counts = mixture_counts({"a": 1000, "b": 10}, {"a": 0.7, "b": 0.3})
    assert counts["b"] == 10
    assert counts["a"] + counts["b"] == 33


def test_mixture_counts_validation():
    with pytest.raises(ValueError, match="sources differ"):
        mixture_counts({"a": 10}, {"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError, match="sum to 1"):
        mixture_counts({"a": 10}, {"a": 0.9})


def _stream(seed=11):
    packed = {"corpus": make_packed("c", 40), "teacher_generated": make_packed("t", 20)}
    return MixtureStream(packed, {"corpus": 0.7, "teacher_generated": 0.3}, seed=seed)


def test_batches_are_bit_identical_across_instantiations():
    a, b = _stream(), _stream()
    for step in range(3):
        assert np.array_equal(a.batch(step, 4), b.batch(step, 4))


def test_seed_changes_the_stream():
    a, b = _stream(11), _stream(12)
    assert not all(np.array_equal(a.batch(s, 4), b.batch(s, 4)) for s in range(3))


def test_resume_from_cursor_matches_uninterrupted():
    reference = _stream()
    upfront = [reference.batch(s, 4).copy() for s in range(6)]
    resumed = _stream()  # cold restart: rebuild from config, jump to cursor
    for step in range(3, 6):
        assert np.array_equal(resumed.batch(step, 4), upfront[step])


def test_schedule_realizes_proportions_exactly():
    stream = _stream()
    sources = [stream.source_of(i) for i in range(len(stream))]
    n = len(sources)
    assert sources.count("teacher_generated") == pytest.approx(0.3 * n, abs=1)
    assert sources.count("corpus") == pytest.approx(0.7 * n, abs=1)


def test_batch_past_end_raises():
    stream = _stream()
    with pytest.raises(IndexError):
        stream.batch(len(stream) // 4 + 1, 4)
