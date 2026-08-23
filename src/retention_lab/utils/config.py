"""Config loading with single-parent inheritance.

A config file may name a parent with the top-level key ``inherit`` (a path
relative to the child file). Resolution replaces whole top-level blocks: a
block present in the child fully replaces the parent's block. This coarse
granularity is deliberate; it is what makes the one-variable-per-experiment
rule checkable by counting differing blocks (ADR-0006).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

INHERIT_KEY = "inherit"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving the ``inherit`` chain child-over-parent."""
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping: {path}")
    if INHERIT_KEY not in raw:
        return raw
    parent = load_config(path.parent / raw[INHERIT_KEY])
    child = {k: v for k, v in raw.items() if k != INHERIT_KEY}
    return {**parent, **child}


def overridden_blocks(child_path: str | Path) -> list[str]:
    """Names of top-level blocks the child config changes relative to its parent."""
    child_path = Path(child_path)
    with open(child_path) as fh:
        raw = yaml.safe_load(fh) or {}
    if INHERIT_KEY not in raw:
        raise ValueError(f"config has no parent to diff against: {child_path}")
    parent = load_config(child_path.parent / raw[INHERIT_KEY])
    changed = []
    for key, value in raw.items():
        if key == INHERIT_KEY:
            continue
        if key not in parent or parent[key] != value:
            changed.append(key)
    return changed


def config_hash(config: dict[str, Any]) -> str:
    """sha256 of the canonical JSON serialization of a resolved config."""
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()
