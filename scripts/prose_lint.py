"""Prose lint: no contractions, no em dashes, no placeholder markers.

The writing convention is a definition-of-done item for this repository, so
it is enforced by a machine, not by memory. The check scans Markdown prose;
possessives are permitted, contractions are not.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CONTRACTION = re.compile(
    r"\b\w+n't\b|\b\w+'(?:re|ll|ve|m)\b|\b(?:it|that|there|what|let|he|she|who|here)'s\b",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(r"\bTODO\b|\bFIXME\b|\bTBD\b|XXX|lorem ipsum", re.IGNORECASE)
EM_DASH = "—"


def lint_text(text: str, name: str) -> list[str]:
    problems = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if EM_DASH in line:
            problems.append(f"{name}:{lineno}: em dash")
        match = CONTRACTION.search(line)
        if match:
            problems.append(f"{name}:{lineno}: contraction {match.group(0)!r}")
        match = PLACEHOLDER.search(line)
        if match:
            problems.append(f"{name}:{lineno}: placeholder {match.group(0)!r}")
    return problems


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    )
    return [Path(p) for p in out.stdout.split()]


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv] or tracked_markdown()
    problems: list[str] = []
    for path in paths:
        problems.extend(lint_text(path.read_text(), str(path)))
    for problem in problems:
        print(problem)
    if problems:
        print(f"prose-lint: {len(problems)} problem(s)")
        return 1
    print(f"prose-lint: OK ({len(paths)} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
