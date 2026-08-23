import pytest

from retention_lab.utils.config import config_hash, load_config, overridden_blocks


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


BASE = """
seed: 1
model:
  d_model: 64
train:
  lr: 0.001
"""

CHILD = """
inherit: base.yaml
train:
  lr: 0.01
"""


def test_inherit_replaces_whole_blocks(tmp_path):
    write(tmp_path, "base.yaml", BASE)
    child = write(tmp_path, "child.yaml", CHILD)
    cfg = load_config(child)
    assert cfg["train"] == {"lr": 0.01}
    assert cfg["model"] == {"d_model": 64}
    assert cfg["seed"] == 1
    assert "inherit" not in cfg


def test_overridden_blocks_names_the_single_change(tmp_path):
    write(tmp_path, "base.yaml", BASE)
    child = write(tmp_path, "child.yaml", CHILD)
    assert overridden_blocks(child) == ["train"]


def test_overridden_blocks_requires_a_parent(tmp_path):
    base = write(tmp_path, "base.yaml", BASE)
    with pytest.raises(ValueError):
        overridden_blocks(base)


def test_config_hash_is_order_insensitive():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
