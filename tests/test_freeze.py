import shutil

import yaml

from retention_lab.battery import freeze


def test_hash_is_stable_across_invocations():
    root = freeze.repo_root()
    assert freeze.scoreboard_hash(root) == freeze.scoreboard_hash(root)


def _copy_scoreboard(root, tmp_path):
    for rel in freeze.SCOREBOARD_INPUTS + (freeze.FREEZE_FILE, "pyproject.toml"):
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / rel, dest)


def test_yaml_formatting_does_not_move_the_hash(tmp_path):
    root = freeze.repo_root()
    _copy_scoreboard(root, tmp_path)
    before = freeze.scoreboard_hash(tmp_path)
    battery = tmp_path / "configs/battery/battery.yaml"
    with open(battery) as fh:
        content = yaml.safe_load(fh)
    battery.write_text(yaml.safe_dump(content, sort_keys=True, indent=4))
    assert freeze.scoreboard_hash(tmp_path) == before


def test_semantic_yaml_change_moves_the_hash(tmp_path):
    root = freeze.repo_root()
    _copy_scoreboard(root, tmp_path)
    before = freeze.scoreboard_hash(tmp_path)
    battery = tmp_path / "configs/battery/battery.yaml"
    with open(battery) as fh:
        content = yaml.safe_load(fh)
    content["tasks"]["sciq"]["eval_cap"] = 999
    battery.write_text(yaml.safe_dump(content))
    assert freeze.scoreboard_hash(tmp_path) != before


def test_scoring_code_edit_moves_the_hash(tmp_path):
    root = freeze.repo_root()
    _copy_scoreboard(root, tmp_path)
    before = freeze.scoreboard_hash(tmp_path)
    scoring = tmp_path / "src/retention_lab/battery/scoring.py"
    scoring.write_text(scoring.read_text() + "\n# semantic-looking edit\n")
    assert freeze.scoreboard_hash(tmp_path) != before


def test_check_passes_unfrozen_and_fails_on_frozen_mismatch(tmp_path, capsys):
    root = freeze.repo_root()
    _copy_scoreboard(root, tmp_path)
    assert freeze.check(tmp_path) == 0

    freeze_file = tmp_path / freeze.FREEZE_FILE
    freeze_file.write_text(
        yaml.safe_dump({"frozen": True, "hash": "0" * 64, "frozen_on": "2026-01-01"})
    )
    assert freeze.check(tmp_path) == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out

    freeze_file.write_text(
        yaml.safe_dump(
            {"frozen": True, "hash": freeze.scoreboard_hash(tmp_path), "frozen_on": "x"}
        )
    )
    assert freeze.check(tmp_path) == 0
