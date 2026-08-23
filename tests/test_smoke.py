import yaml

from retention_lab.smoke import run_smoke

TINY_TEST = {
    "seed": 99,
    "model": {"vocab_size": 32, "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 32},
    "teacher_model": {"vocab_size": 32, "d_model": 48, "n_layer": 2, "n_head": 2, "seq_len": 32},
    "data": {"source": "synthetic", "n_tokens": 20000},
    "loss": {"name": "forward_kl", "temperature": 2.0, "alpha": 0.5},
    "teacher_train": {"steps": 6, "batch_size": 4, "lr": 0.01, "warmup_steps": 2},
    "train": {"steps": 8, "batch_size": 4, "lr": 0.01, "warmup_steps": 2},
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


def test_smoke_emits_kd_components(tmp_path):
    traces = run_smoke(_config(tmp_path))
    assert set(traces) == {"teacher_loss", "student_total", "student_ce", "student_kd"}
    assert len(traces["student_kd"]) == 8
