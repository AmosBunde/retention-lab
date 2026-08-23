"""Scoreboard freeze: content hash, check, and the defect protocol.

The frozen scoreboard is not only the battery YAML; the scoring code is the
measuring instrument, so the hash covers both. After the freeze commit
(FREEZE.yaml carries ``frozen: true`` and the hash), any drift in these
inputs fails CI. The remedy for a post-freeze defect is never a silent
patch: raise an issue labeled ``frozen-battery``, land the fix together
with an updated FREEZE.yaml whose ``history`` entry explains what changed
and why, and let the diff show both.

The YAML is hashed as canonical JSON of its parsed content, so formatting
and key order cannot move the hash; source files are hashed as raw bytes,
so every semantic edit does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

SCOREBOARD_INPUTS = (
    "configs/battery/battery.yaml",
    "src/retention_lab/battery/protocol.py",
    "src/retention_lab/battery/scoring.py",
    "src/retention_lab/battery/registry.py",
    "src/retention_lab/metrics/retention.py",
    "src/retention_lab/metrics/bands.py",
)

FREEZE_FILE = "configs/battery/FREEZE.yaml"


def repo_root() -> Path:
    root = Path(__file__).resolve()
    for parent in root.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repository root not found")


def scoreboard_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for rel in SCOREBOARD_INPUTS:
        path = root / rel
        digest.update(rel.encode())
        digest.update(b"\x00")
        if path.suffix in (".yaml", ".yml"):
            with open(path) as fh:
                content = yaml.safe_load(fh)
            digest.update(json.dumps(content, sort_keys=True, separators=(",", ":")).encode())
        else:
            digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def load_freeze(root: Path) -> dict:
    with open(root / FREEZE_FILE) as fh:
        return yaml.safe_load(fh)


def check(root: Path) -> int:
    current = scoreboard_hash(root)
    freeze = load_freeze(root)
    if not freeze.get("frozen"):
        print(f"battery-hash: {current}")
        print("battery-hash: scoreboard is not frozen yet; check passes by definition")
        return 0
    expected = freeze.get("hash")
    if current == expected:
        print(f"battery-hash: OK (frozen at {expected})")
        return 0
    print("battery-hash: FROZEN HASH MISMATCH")
    print(f"  expected: {expected}")
    print(f"  current:  {current}")
    print(
        "  The scoreboard is frozen. If this change is a raised defect fix, "
        "update FREEZE.yaml with the new hash and a history entry in the "
        "same pull request; otherwise revert the scoreboard edit."
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoreboard freeze hash")
    parser.add_argument("--check", action="store_true", help="verify against FREEZE.yaml")
    args = parser.parse_args()
    root = repo_root()
    if args.check:
        raise SystemExit(check(root))
    print(scoreboard_hash(root))


if __name__ == "__main__":
    main()
