"""Retention with control attribution.

Normalization puts every task on one scale before any comparison: a task
score of 0.0 means chance level and 1.0 means perfect, for accuracy tasks
((raw - chance) / (1 - chance)) and for bits per byte
((chance - raw) / chance, chance being the uniform-model BPB) alike. A
capability score is the unweighted mean of its normalized task scores, so
the capability chance level is 0 by construction and the README formulas
reduce to:

    retention    R_c = S_c / T_c
    attribution  A_c = (S_c - C_c) / (T_c - C_c)

Values outside [0, 1] are reported as computed: a student above its teacher
shows R_c > 1, a student below chance shows S_c < 0, and a control that
beats the student shows a negative delta. The renderer refuses to produce
any table without the control columns; that refusal is the point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from retention_lab.battery.scoring import TaskResult


def normalize_task(result: TaskResult) -> float:
    """Map a raw task score to the chance=0, perfect=1 scale."""
    if result.metric == "accuracy":
        if not result.chance < 1.0:
            raise ValueError(f"{result.task}: chance level {result.chance} leaves no headroom")
        return (result.value - result.chance) / (1.0 - result.chance)
    if result.metric == "bpb":
        if not result.chance > 0.0:
            raise ValueError(f"{result.task}: uniform BPB must be positive")
        return (result.chance - result.value) / result.chance
    raise ValueError(f"{result.task}: unknown metric {result.metric!r}")


@dataclass(frozen=True)
class CapabilityScore:
    capability: str
    value: float
    tasks: tuple[tuple[str, float], ...]  # (task name, normalized score)


def capability_scores(
    grouped: Mapping[str, Sequence[TaskResult]],
) -> dict[str, CapabilityScore]:
    out = {}
    for capability, results in grouped.items():
        if not results:
            raise ValueError(f"capability {capability!r} has no task results")
        normalized = tuple((r.task, normalize_task(r)) for r in results)
        value = sum(v for _, v in normalized) / len(normalized)
        out[capability] = CapabilityScore(capability, value, normalized)
    return out


@dataclass(frozen=True)
class RetentionRow:
    capability: str
    teacher: float
    control: float
    student: float
    retention: float
    attribution: float
    delta_vs_control: float


def retention_rows(
    teacher: Mapping[str, CapabilityScore],
    control: Mapping[str, CapabilityScore],
    student: Mapping[str, CapabilityScore],
) -> list[RetentionRow]:
    """Per-capability retention, attribution, and delta against control.

    All three score sets are mandatory and must cover identical capabilities;
    a retention number without its control context is not a result in this
    study, so there is no code path that produces one.
    """
    if set(teacher) != set(control) or set(teacher) != set(student):
        raise ValueError(
            "capability sets differ: "
            f"teacher={sorted(teacher)} control={sorted(control)} student={sorted(student)}"
        )
    rows = []
    for capability in teacher:
        t, c, s = teacher[capability].value, control[capability].value, student[capability].value
        if t <= 0.0:
            raise ValueError(
                f"{capability}: teacher is at or below chance (T_c={t:.6f}); "
                "retention against this denominator is meaningless and the "
                "battery composition must be revisited through the defect protocol"
            )
        if t == c:
            raise ValueError(
                f"{capability}: teacher and control are identical (={t:.6f}); "
                "attribution is undefined"
            )
        rows.append(
            RetentionRow(
                capability=capability,
                teacher=t,
                control=c,
                student=s,
                retention=s / t,
                attribution=(s - c) / (t - c),
                delta_vs_control=s - c,
            )
        )
    return sorted(rows, key=lambda r: r.capability)


def render_markdown(
    rows: Sequence[RetentionRow], verdicts: Mapping[str, str] | None = None
) -> str:
    """Markdown table with the control columns always present.

    ``verdicts`` maps capability to a band verdict string; before band values
    exist (they require real control reruns), every verdict renders as
    ``band pending`` rather than being silently omitted.
    """
    header = (
        "| Capability | Teacher | Control | Student | Retention R_c | "
        "Attribution A_c | Delta vs control | Band verdict |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in rows:
        verdict = (verdicts or {}).get(row.capability, "band pending")
        lines.append(
            f"| {row.capability} | {row.teacher:.4f} | {row.control:.4f} | "
            f"{row.student:.4f} | {row.retention:.4f} | {row.attribution:.4f} | "
            f"{row.delta_vs_control:+.4f} | {verdict} |"
        )
    return "\n".join(lines)
