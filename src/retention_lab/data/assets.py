"""Download and verify pinned assets against the manifest.

Every file lands under the git-ignored ``assets/`` directory and is verified
against its manifest sha256 before the download counts as complete; a
mismatch removes the offending file and aborts, because a silently wrong
weight is worse than no weight. Re-running is idempotent: files that
already verify are not fetched again.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from retention_lab.utils.config import load_config

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
    tmp.rename(dest)


def verify_or_fetch_model(name: str, spec: dict, assets_root: Path) -> list[str]:
    """Ensure every manifest file of one model verifies; returns actions taken."""
    actions = []
    base = assets_root / "models" / name
    for filename, expected in spec["files"].items():
        dest = base / filename
        if dest.exists() and sha256_file(dest) == expected["sha256"]:
            actions.append(f"ok {name}/{filename}")
            continue
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{filename}"
        fetch(url, dest)
        actual = sha256_file(dest)
        if actual != expected["sha256"]:
            dest.unlink()
            raise RuntimeError(
                f"hash mismatch for {name}/{filename}: expected "
                f"{expected['sha256']}, downloaded {actual}; file removed"
            )
        actions.append(f"fetched {name}/{filename}")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify pinned assets")
    parser.add_argument("--config", default="configs/assets.yaml")
    parser.add_argument("--assets-root", default="assets")
    parser.add_argument(
        "--only", default=None, help="restrict to one model key (teacher, student-pretrained)"
    )
    args = parser.parse_args()

    manifest = load_config(args.config)
    root = Path(args.assets_root)
    models = manifest["models"]
    if args.only:
        models = {args.only: models[args.only]}
    for name, spec in models.items():
        for action in verify_or_fetch_model(name, spec, root):
            print(f"assets: {action}")
    print("assets: OK")


if __name__ == "__main__":
    main()
