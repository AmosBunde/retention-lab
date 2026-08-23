import yaml

from retention_lab.smoke import run_smoke

TINY_TEST = {
    "seed": 99,
    "model": {"vocab_size": 32, "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 32},
    "data": {"source": "synthetic", "n_tokens": 20000},
    "train": {"steps": 8, "batch_size": 4, "lr": 0.003},
}


def _config(tmp_path, overrides=None):
    cfg = {**TINY_TEST, **(overrides or {})}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_smoke_is_bit_identical_across_invocations(tmp_path):
    path = _config(tmp_path)
    assert run_smoke(path) == run_smoke(path)


def test_smoke_changes_with_seed(tmp_path):
    a = run_smoke(_config(tmp_path))
    b = run_smoke(_config(tmp_path, {"seed": 100}))
    assert a != b
