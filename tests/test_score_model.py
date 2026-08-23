import json

import pytest

from retention_lab.battery.score_model import ScoreRecord, write_record
from retention_lab.models.pinned import HFTokenizerAdapter, load_pinned_causal_lm


class StubTokenizer:
    vocab_size = 50304

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "battery scoring must not add special tokens"
        return [ord(c) % 50304 for c in text]


def test_moving_revision_is_rejected_before_any_network():
    with pytest.raises(ValueError, match="40-hex"):
        load_pinned_causal_lm("EleutherAI/pythia-1.4b", "main")
    with pytest.raises(ValueError, match="40-hex"):
        load_pinned_causal_lm("EleutherAI/pythia-1.4b", "step143000")


def test_tokenizer_adapter_disables_special_tokens():
    adapter = HFTokenizerAdapter(StubTokenizer())
    assert adapter.vocab_size == 50304
    assert adapter.encode("ab") == [ord("a") % 50304, ord("b") % 50304]


def test_record_roundtrip_preserves_cost_fields(tmp_path):
    record = ScoreRecord(
        run_id="teacher-battery-v1",
        kind="teacher",
        model="EleutherAI/pythia-1.4b",
        revision="0" * 40,
        battery_hash="f" * 64,
        slice="full",
        gpu_hours=1.75,
        cost_usd=1.12,
        instance="example-provider RTX 4090",
        image_tag="sha-abcdef123456",
        wall_seconds=6300.0,
        scores={"recall": []},
    )
    out = tmp_path / "tracker" / "runs" / "teacher.json"
    write_record(record, out)
    loaded = json.loads(out.read_text())
    assert loaded["gpu_hours"] == 1.75
    assert loaded["cost_usd"] == 1.12
    assert loaded["battery_hash"] == "f" * 64
    assert loaded["kind"] == "teacher"
