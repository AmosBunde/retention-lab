"""The config-diff test: one variable per experiment, machine-enforced.

Every file in configs/variants/ must inherit the baseline and override
exactly one top-level block. This test IS the enforcement the contract
demands; it runs in CI on every pull request, so a two-block variant can
never merge.
"""

from pathlib import Path

import pytest
import yaml

from retention_lab.utils.config import config_hash, load_config, overridden_blocks

VARIANTS = sorted(Path("configs/variants").glob("*.yaml"))
EXPECTED_BLOCK = {
    "control.yaml": "loss",
    "e1-reverse-kl.yaml": "loss",
    "e2-temperature-4.yaml": "loss",
    "e3-mixture-30.yaml": "mixture",
    "e4-init-pretrained.yaml": "init",
}


def test_every_planned_variant_exists():
    assert [p.name for p in VARIANTS] == sorted(EXPECTED_BLOCK)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda p: p.name)
def test_variant_overrides_exactly_one_block(variant):
    changed = overridden_blocks(variant)
    assert len(changed) == 1, f"{variant.name} overrides {changed}"
    assert changed[0] == EXPECTED_BLOCK[variant.name]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda p: p.name)
def test_variant_resolves_against_baseline_with_distinct_hash(variant):
    baseline = load_config("configs/baseline.yaml")
    resolved = load_config(variant)
    assert set(resolved) == set(baseline), "variants may not add or drop blocks"
    assert config_hash(resolved) != config_hash(baseline)
    assert resolved["seed"] == baseline["seed"], "the shared data seed is not overridable"
    assert resolved["train"] == baseline["train"], "budget and schedule are not overridable"


def test_two_block_variant_is_rejected(tmp_path):
    (tmp_path / "baseline.yaml").write_text(
        Path("configs/baseline.yaml").read_text()
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "inherit": "baseline.yaml",
                "loss": {"name": "reverse_kl"},
                "train": {"steps": 10},
            }
        )
    )
    assert len(overridden_blocks(bad)) == 2  # exactly what the parametrized test forbids


def test_no_op_variant_is_rejected(tmp_path):
    (tmp_path / "baseline.yaml").write_text(
        Path("configs/baseline.yaml").read_text()
    )
    noop = tmp_path / "noop.yaml"
    noop.write_text(yaml.safe_dump({"inherit": "baseline.yaml"}))
    assert len(overridden_blocks(noop)) == 0


def test_control_is_the_lm_only_collapse():
    control = load_config("configs/variants/control.yaml")
    assert control["loss"] == {"name": "lm_only"}
    assert control["teacher"] == load_config("configs/baseline.yaml")["teacher"]
