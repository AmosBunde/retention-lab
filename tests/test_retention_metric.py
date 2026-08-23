import pytest

from retention_lab.battery.scoring import TaskResult
from retention_lab.metrics.retention import (
    CapabilityScore,
    capability_scores,
    normalize_task,
    render_markdown,
    retention_rows,
)


def acc(task, value, chance=0.25, n=8):
    return TaskResult(task, "accuracy", value, chance, n, tuple([value] * n))


def bpb(task, value, chance=6.0, n=4):
    return TaskResult(task, "bpb", value, chance, n, tuple([value] * n))


def cap(name, value):
    return CapabilityScore(name, value, ((name, value),))


def test_accuracy_normalization_anchors():
    assert normalize_task(acc("t", 0.25)) == 0.0
    assert normalize_task(acc("t", 1.0)) == 1.0
    assert normalize_task(acc("t", 0.10)) == pytest.approx(-0.2)


def test_bpb_normalization_anchors():
    assert normalize_task(bpb("t", 6.0)) == 0.0
    assert normalize_task(bpb("t", 0.0)) == 1.0
    assert normalize_task(bpb("t", 3.0)) == pytest.approx(0.5)
    assert normalize_task(bpb("t", 7.5)) == pytest.approx(-0.25)


def test_capability_mean_of_mixed_chance_tasks():
    grouped = {"recall": [acc("a", 0.625, chance=0.25), acc("b", 0.75, chance=0.5)]}
    scores = capability_scores(grouped)
    assert scores["recall"].value == pytest.approx((0.5 + 0.5) / 2)


def test_retention_and_attribution_basic():
    rows = retention_rows(
        teacher={"c": cap("c", 0.8)},
        control={"c": cap("c", 0.2)},
        student={"c": cap("c", 0.5)},
    )
    row = rows[0]
    assert row.retention == pytest.approx(0.625)
    assert row.attribution == pytest.approx(0.5)
    assert row.delta_vs_control == pytest.approx(0.3)


def test_student_above_teacher_and_below_chance_report_as_computed():
    above = retention_rows(
        {"c": cap("c", 0.5)}, {"c": cap("c", 0.1)}, {"c": cap("c", 0.6)}
    )[0]
    assert above.retention == pytest.approx(1.2)
    below = retention_rows(
        {"c": cap("c", 0.5)}, {"c": cap("c", 0.1)}, {"c": cap("c", -0.05)}
    )[0]
    assert below.retention == pytest.approx(-0.1)
    assert below.attribution < 0


def test_control_beating_student_is_negative_attribution():
    row = retention_rows(
        {"c": cap("c", 0.8)}, {"c": cap("c", 0.4)}, {"c": cap("c", 0.3)}
    )[0]
    assert row.delta_vs_control == pytest.approx(-0.1)
    assert row.attribution == pytest.approx(-0.25)


def test_teacher_at_chance_raises():
    with pytest.raises(ValueError, match="at or below chance"):
        retention_rows({"c": cap("c", 0.0)}, {"c": cap("c", 0.1)}, {"c": cap("c", 0.1)})


def test_mismatched_capabilities_raise():
    with pytest.raises(ValueError, match="capability sets differ"):
        retention_rows({"a": cap("a", 0.5)}, {"b": cap("b", 0.1)}, {"a": cap("a", 0.2)})


def test_renderer_always_carries_control_columns_and_pending_bands():
    rows = retention_rows(
        {"c": cap("c", 0.8)}, {"c": cap("c", 0.2)}, {"c": cap("c", 0.5)}
    )
    table = render_markdown(rows)
    assert "| Control |" in table
    assert "band pending" in table
    with_verdict = render_markdown(rows, {"c": "no effect"})
    assert "no effect" in with_verdict
