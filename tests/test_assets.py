import subprocess
from pathlib import Path

import pytest

from retention_lab.data.assets import sha256_file, verify_or_fetch_model
from retention_lab.utils.config import load_config

WEIGHT_SUFFIXES = {".bin", ".safetensors", ".pt", ".ckpt", ".parquet", ".arrow", ".gguf"}
MAX_TRACKED_BYTES = 5 * 1024 * 1024


def test_manifest_is_complete_and_pinned():
    manifest = load_config("configs/assets.yaml")
    for name, spec in manifest["models"].items():
        assert len(spec["revision"]) == 40, name
        assert spec["license"], name
        for filename, entry in spec["files"].items():
            assert len(entry["sha256"]) == 64, f"{name}/{filename}"
            assert entry["size"] > 0, f"{name}/{filename}"


def test_verifier_accepts_existing_matching_file(tmp_path):
    content = b"pinned bytes"
    dest = tmp_path / "models" / "m" / "f.json"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(content)
    spec = {
        "repo": "example/none",
        "revision": "0" * 40,
        "files": {"f.json": {"size": len(content), "sha256": sha256_file(dest)}},
    }
    actions = verify_or_fetch_model("m", spec, tmp_path)
    assert actions == ["ok m/f.json"]


def test_verifier_refetches_on_mismatch_and_fails_loud(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "m" / "f.json"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"corrupted")

    def fake_fetch(url, path):
        path.write_bytes(b"still wrong")

    monkeypatch.setattr("retention_lab.data.assets.fetch", fake_fetch)
    spec = {
        "repo": "example/none",
        "revision": "0" * 40,
        "files": {"f.json": {"size": 3, "sha256": "a" * 64}},
    }
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_or_fetch_model("m", spec, tmp_path)
    assert not dest.exists(), "mismatching file must be removed"


def test_no_weight_or_data_file_is_tracked_by_git():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    offenders = []
    for line in out.stdout.split():
        path = Path(line)
        if path.suffix in WEIGHT_SUFFIXES:
            offenders.append(f"{line} (weight/data extension)")
        elif path.exists() and path.stat().st_size > MAX_TRACKED_BYTES:
            offenders.append(f"{line} ({path.stat().st_size} bytes)")
    assert not offenders, f"asset hygiene violation: {offenders}"
