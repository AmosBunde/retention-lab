"""Results tables generated exclusively from tracker records.

There is no other path into a results table: the generator loads validated
records, reconstructs task results from stored per-item scores, and renders
through the retention module, whose renderer structurally requires the
control columns. Without a teacher record and at least one control record,
no student table exists, which is the control-first rule as code.
"""

from __future__ import annotations

import argparse

from retention_lab.battery.scoring import TaskResult
from retention_lab.metrics.retention import (
    CapabilityScore,
    capability_scores,
    render_markdown,
    retention_rows,
)
from retention_lab.tracker.schema import load_records


def record_capabilities(record: dict) -> dict[str, CapabilityScore]:
    grouped = {
        capability: [
            TaskResult(
                task=entry["task"],
                metric=entry["metric"],
                value=entry["value"],
                chance=entry["chance"],
                n_items=entry["n"],
                per_item=tuple(entry["per_item"]),
            )
            for entry in entries
        ]
        for capability, entries in record["scores"].items()
    }
    return capability_scores(grouped)


def mean_capabilities(
    per_record: list[dict[str, CapabilityScore]],
) -> dict[str, CapabilityScore]:
    """Mean capability scores across seed reruns of one arm."""
    if not per_record:
        raise ValueError("no records to average")
    out = {}
    for capability in per_record[0]:
        values = [r[capability].value for r in per_record]
        out[capability] = CapabilityScore(
            capability, sum(values) / len(values), (("seed-mean", sum(values) / len(values)),)
        )
    return out


def summarize(records: list[dict]) -> str:
    by_kind: dict[str, list[dict]] = {}
    for record in records:
        by_kind.setdefault(record["kind"], []).append(record)
    lines = ["# Results (generated from tracker records only)", ""]
    if not records:
        lines.append("No runs are recorded yet. Tables appear here after real executions.")
        return "\n".join(lines)
    for kind, group in sorted(by_kind.items()):
        for record in group:
            lines.append(
                f"- {record['run_id']} ({kind}): gpu_hours={record['gpu_hours']}, "
                f"cost_usd={record['cost_usd']}, instance={record['instance']}"
            )
    lines.append("")
    teachers = by_kind.get("teacher", [])
    controls = by_kind.get("control", [])
    students = by_kind.get("student", []) + by_kind.get("int8", [])
    if not teachers or not controls:
        lines.append(
            "Retention tables require the teacher denominators and at least "
            "one control record; not all exist yet."
        )
        return "\n".join(lines)
    if len(teachers) > 1:
        raise ValueError("multiple teacher records; the battery is scored once by contract")
    teacher_caps = record_capabilities(teachers[0])
    control_caps = mean_capabilities([record_capabilities(r) for r in controls])
    for student in students:
        rows = retention_rows(teacher_caps, control_caps, record_capabilities(student))
        lines.append(f"## {student['run_id']} (gpu_hours={student['gpu_hours']}, "
                     f"cost_usd={student['cost_usd']})")
        lines.append("")
        lines.append(render_markdown(rows))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render results from tracker records")
    parser.add_argument("--runs", default="tracker/runs")
    args = parser.parse_args()
    print(summarize(load_records(args.runs)))


if __name__ == "__main__":
    main()
