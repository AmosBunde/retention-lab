"""Tracker record validation: budget honesty as a schema property.

Every record in tracker/runs/ must name its cost. Validation runs in CI
over all committed records, so a record without GPU hours, cost, instance,
and image tag cannot merge. Only real executions produce records; nothing
in this repository writes a record from a simulation, and the validator
cannot check that by construction, so the review process does: each record
arrives in a pull request quoting the run it came from.
"""

from __future__ import annotations

import json
from pathlib import Path

KINDS = ("teacher", "control", "student", "int8", "generation")

COMMON_REQUIRED = {
    "run_id": str,
    "kind": str,
    "gpu_hours": (int, float),
    "cost_usd": (int, float),
    "instance": str,
    "image_tag": str,
}

SCORING_REQUIRED = {
    "model": str,
    "revision": str,
    "battery_hash": str,
    "slice": str,
    "scores": dict,
    "wall_seconds": (int, float),
}

TRAINING_EXTRA = {
    "config_hash": str,
    "seed": int,
    "tokens_trained": int,
    "reclaims": int,
}


def validate_record(record: dict) -> list[str]:
    problems = []

    def check(fields: dict) -> None:
        for name, types in fields.items():
            if name not in record:
                problems.append(f"missing required field {name!r}")
            elif not isinstance(record[name], types):
                problems.append(f"field {name!r} has type {type(record[name]).__name__}")

    check(COMMON_REQUIRED)
    kind = record.get("kind")
    if kind not in KINDS:
        problems.append(f"unknown kind {kind!r}; known: {KINDS}")
    if kind in ("teacher", "control", "student", "int8"):
        check(SCORING_REQUIRED)
    if kind in ("control", "student"):
        check(TRAINING_EXTRA)
    for money in ("gpu_hours", "cost_usd"):
        value = record.get(money)
        if isinstance(value, int | float) and value < 0:
            problems.append(f"field {money!r} is negative")
    return problems


def load_records(runs_dir: str | Path) -> list[dict]:
    """Load and validate every committed record; any invalid record raises."""
    records = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        with open(path) as fh:
            record = json.load(fh)
        problems = validate_record(record)
        if problems:
            raise ValueError(f"{path}: " + "; ".join(problems))
        records.append(record)
    return records
