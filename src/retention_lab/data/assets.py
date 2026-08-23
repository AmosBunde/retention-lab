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


def _resolve_url(spec: dict, filename: str) -> str:
    prefix = "datasets/" if spec.get("repo_type") == "dataset" else ""
    return (
        f"https://huggingface.co/{prefix}{spec['repo']}/resolve/"
        f"{spec['revision']}/{filename}"
    )


def verify_or_fetch(name: str, spec: dict, base: Path) -> list[str]:
    """Ensure every manifest file of one asset verifies; returns actions taken."""
    actions = []
    for filename, expected in spec["files"].items():
        dest = base / filename
        if dest.exists() and sha256_file(dest) == expected["sha256"]:
            actions.append(f"ok {name}/{filename}")
            continue
        fetch(_resolve_url(spec, filename), dest)
        actual = sha256_file(dest)
        if actual != expected["sha256"]:
            dest.unlink()
            raise RuntimeError(
                f"hash mismatch for {name}/{filename}: expected "
                f"{expected['sha256']}, downloaded {actual}; file removed"
            )
        actions.append(f"fetched {name}/{filename}")
    return actions


def verify_or_fetch_model(name: str, spec: dict, assets_root: Path) -> list[str]:
    return verify_or_fetch(name, spec, assets_root / "models" / name)


def verify_or_fetch_corpus(name: str, spec: dict, assets_root: Path) -> list[str]:
    return verify_or_fetch(name, spec, assets_root / "corpus" / name)


def corpus_shard_paths(spec: dict, assets_root: Path, name: str) -> list[Path]:
    """Local shard paths in manifest order (the order that defines indices)."""
    return [assets_root / "corpus" / name / filename for filename in spec["files"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify pinned assets")
    parser.add_argument("--config", default="configs/assets.yaml")
    parser.add_argument("--assets-root", default="assets")
    parser.add_argument(
        "--only",
        default=None,
        help="restrict to one asset key (teacher, student-pretrained, fineweb-edu-sample)",
    )
    args = parser.parse_args()

    manifest = load_config(args.config)
    root = Path(args.assets_root)
    sections = (
        ("models", verify_or_fetch_model),
        ("corpus", verify_or_fetch_corpus),
    )
    matched = 0
    for section, handler in sections:
        for name, spec in manifest.get(section, {}).items():
            if args.only and name != args.only:
                continue
            matched += 1
            for action in handler(name, spec, root):
                print(f"assets: {action}")
    if args.only and matched == 0:
        raise SystemExit(f"assets: unknown asset key {args.only!r}")
    print("assets: OK")


if __name__ == "__main__":
    main()
