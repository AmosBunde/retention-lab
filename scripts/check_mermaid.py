"""Render-check every Mermaid block in tracked Markdown files.

GitHub renders Mermaid client-side, so a parse error ships silently unless a
machine renders each block first. The bootstrap pull request caught exactly
one such defect by hand; this check exists so CI catches the next one.

Requires node (npx). Uses the system Chrome named by MERMAID_CHROME when set,
so CI does not download a browser; locally, puppeteer fetches its own.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BLOCK = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    )
    return [Path(p) for p in out.stdout.split()]


def main() -> int:
    blocks: list[tuple[str, str]] = []
    for path in tracked_markdown():
        for i, body in enumerate(BLOCK.findall(path.read_text())):
            blocks.append((f"{path}#{i}", body))
    if not blocks:
        print("mermaid-check: no blocks found")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        pptr = Path(tmp) / "pptr.json"
        pptr_cfg: dict = {"args": ["--no-sandbox"]}
        chrome = os.environ.get("MERMAID_CHROME")
        if chrome:
            pptr_cfg["executablePath"] = chrome
        pptr.write_text(json.dumps(pptr_cfg))

        failures = 0
        for name, body in blocks:
            src = Path(tmp) / (name.replace("/", "_").replace("#", "_") + ".mmd")
            src.write_text(body)
            out = src.with_suffix(".svg")
            proc = subprocess.run(
                ["npx", "--yes", "@mermaid-js/mermaid-cli", "-p", str(pptr),
                 "-i", str(src), "-o", str(out), "--quiet"],
                capture_output=True, text=True,
            )
            if proc.returncode != 0 or not out.exists():
                failures += 1
                print(f"mermaid-check: FAIL {name}")
                sys.stdout.write(proc.stderr[-2000:])
            else:
                print(f"mermaid-check: ok {name}")
        if failures:
            print(f"mermaid-check: {failures} failing block(s)")
            return 1
    print(f"mermaid-check: OK ({len(blocks)} block(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
