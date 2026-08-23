import pyarrow.parquet as pq
import pytest

from retention_lab.data.teacher_gen import (
    generate_documents,
    select_prompt_indices,
    write_generated_shard,
)

CFG = {
    "teacher": {"repo": "EleutherAI/pythia-1.4b", "revision": "f" * 40},
    "sampling": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 4},
}


def echo_generate(prompt_batches):
    return [[t + 1 for t in tokens][:4] for tokens in prompt_batches]


def decode(tokens):
    return " ".join(str(t) for t in tokens)


def test_prompt_selection_is_deterministic_and_bounded():
    a = select_prompt_indices(1000, 10, seed=1)
    assert a == select_prompt_indices(1000, 10, seed=1)
    assert a != select_prompt_indices(1000, 10, seed=2)
    with pytest.raises(ValueError):
        select_prompt_indices(5, 10, seed=1)


def test_generated_rows_carry_full_provenance():
    prompts = [(3, [10, 11]), (7, [20, 21, 22])]
    rows = list(generate_documents(prompts, echo_generate, decode, CFG, batch_size=1))
    assert len(rows) == 2
    for row in rows:
        assert row["source"] == "teacher_generated"
        assert row["teacher_revision"] == "f" * 40
        assert len(row["sampling_hash"]) == 64
    assert rows[0]["prompt_doc_index"] == 3
    assert rows[0]["text"].startswith("10 11 ")
    assert rows[0]["n_generated_tokens"] == 2


def test_sampling_hash_tracks_the_sampling_block():
    prompts = [(0, [1, 2])]
    hot = {**CFG, "sampling": {**CFG["sampling"], "temperature": 1.2}}
    row_a = next(iter(generate_documents(prompts, echo_generate, decode, CFG)))
    row_b = next(iter(generate_documents(prompts, echo_generate, decode, hot)))
    assert row_a["sampling_hash"] != row_b["sampling_hash"]


def test_shard_writer_enforces_label_and_roundtrips(tmp_path):
    prompts = [(i, [i, i + 1]) for i in range(5)]
    rows = list(generate_documents(prompts, echo_generate, decode, CFG))
    out = tmp_path / "gen" / "shard-000.parquet"
    write_generated_shard(rows, out)
    table = pq.read_table(out)
    assert set(table.column("source").to_pylist()) == {"teacher_generated"}
    assert table.num_rows == 5

    rows[2] = {**rows[2], "source": "corpus"}
    with pytest.raises(ValueError, match="labeled"):
        write_generated_shard(rows, tmp_path / "gen" / "bad.parquet")


def test_mismatched_generate_batch_is_rejected():
    def broken(prompt_batches):
        return prompt_batches[:-1]

    with pytest.raises(ValueError, match="mismatched"):
        list(generate_documents([(0, [1]), (1, [2])], broken, decode, CFG, batch_size=2))
