import json

import pytest

from retention_lab.tracker.results import summarize
from retention_lab.tracker.schema import load_records, validate_record


def scores_block(value_by_cap):
    return {
        cap: [
            {
                "task": f"{cap}-task",
                "metric": "accuracy",
                "value": value,
                "chance": 0.0,
                "n": 4,
                "per_item": [value] * 4,
            }
        ]
        for cap, value in value_by_cap.items()
    }


def scoring_record(run_id, kind, value_by_cap, **extra):
    record = {
        "run_id": run_id,
        "kind": kind,
        "gpu_hours": 1.5,
        "cost_usd": 0.9,
        "instance": "test-rig",
        "image_tag": "sha-test",
        "model": "EleutherAI/pythia-160m",
        "revision": "0" * 40,
        "battery_hash": "f" * 64,
        "slice": "full",
        "wall_seconds": 100.0,
        "scores": scores_block(value_by_cap),
    }
    record.update(extra)
    return record


TRAIN_EXTRA = {"config_hash": "c" * 64, "seed": 1, "tokens_trained": 100, "reclaims": 0}


def test_valid_records_pass():
    assert validate_record(scoring_record("t", "teacher", {"recall": 0.8})) == []
    assert validate_record(scoring_record("c", "control", {"recall": 0.2}, **TRAIN_EXTRA)) == []


def test_missing_cost_and_negative_hours_fail():
    record = scoring_record("t", "teacher", {"recall": 0.8})
    del record["cost_usd"]
    assert any("cost_usd" in p for p in validate_record(record))
    record = scoring_record("t", "teacher", {"recall": 0.8}, gpu_hours=-1)
    assert any("negative" in p for p in validate_record(record))


def test_unknown_kind_and_missing_training_fields_fail():
    assert any("unknown kind" in p for p in validate_record(scoring_record("x", "demo", {})))
    incomplete = scoring_record("c", "control", {"recall": 0.2})
    assert any("tokens_trained" in p for p in validate_record(incomplete))


def test_load_records_raises_on_invalid_file(tmp_path):
    bad = scoring_record("t", "teacher", {"recall": 0.8})
    del bad["instance"]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="instance"):
        load_records(tmp_path)


def test_committed_records_directory_validates():
    load_records("tracker/runs")  # empty today; any future record must validate


def test_summarize_requires_teacher_and_control():
    student = scoring_record("s1", "student", {"recall": 0.5}, **TRAIN_EXTRA)
    text = summarize([student])
    assert "require the teacher denominators" in text
    assert "| Control |" not in text


def test_summarize_renders_retention_with_control_columns():
    teacher = scoring_record("teacher-battery-v1", "teacher", {"recall": 0.8})
    control = scoring_record("control-seed1", "control", {"recall": 0.2}, **TRAIN_EXTRA)
    student = scoring_record("kd-seed1", "student", {"recall": 0.5}, **TRAIN_EXTRA)
    text = summarize([teacher, control, student])
    assert "| Control |" in text
    assert "0.6250" in text  # retention 0.5 / 0.8
    assert "0.5000" in text  # attribution (0.5 - 0.2) / (0.8 - 0.2)
    assert "band pending" in text


def test_summarize_rejects_a_second_teacher_record():
    teacher = scoring_record("teacher-battery-v1", "teacher", {"recall": 0.8})
    dup = scoring_record("teacher-battery-v2", "teacher", {"recall": 0.9})
    control = scoring_record("control-seed1", "control", {"recall": 0.2}, **TRAIN_EXTRA)
    with pytest.raises(ValueError, match="scored once"):
        summarize([teacher, dup, control])
