import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from kill_resume_helper import build_trainer
from retention_lab.train.checkpoint import (
    LATEST,
    PREVIOUS,
    resume_if_available,
    save_checkpoint,
)

HELPER = Path(__file__).parent / "kill_resume_helper.py"


def losses(metrics):
    return [m["total"] for m in metrics]


def test_cold_rebuild_resume_is_bit_identical(tmp_path):
    reference = build_trainer()
    ref_losses = losses(reference.train(8))

    first = build_trainer()
    first.train(4)
    ckpt_dir = tmp_path / "roundtrip"
    save_checkpoint(first, ckpt_dir)
    del first

    resumed = build_trainer()  # fresh model, optimizer, RNG
    assert resume_if_available(resumed, ckpt_dir) == 4
    cont_losses = losses(resumed.train(4))
    assert cont_losses == ref_losses[4:]
    for a, b in zip(
        resumed.model.state_dict().values(),
        build_and_train_reference().model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(a, b)


def build_and_train_reference():
    trainer = build_trainer()
    trainer.train(8)
    return trainer


def test_corrupt_latest_falls_back_to_previous(tmp_path, capsys):
    trainer = build_trainer()
    trainer.train(2)
    save_checkpoint(trainer, tmp_path)
    trainer.train(2)
    save_checkpoint(trainer, tmp_path)  # latest at step 4, previous at step 2
    (tmp_path / LATEST).write_bytes(b"reclaimed mid-write")

    fresh = build_trainer()
    assert resume_if_available(fresh, tmp_path) == 2
    out = capsys.readouterr().out
    assert "falling back" in out
    assert PREVIOUS in out


def test_save_is_atomic_and_keeps_previous(tmp_path):
    trainer = build_trainer()
    trainer.train(2)
    save_checkpoint(trainer, tmp_path)
    trainer.train(1)
    save_checkpoint(trainer, tmp_path)
    assert (tmp_path / LATEST).exists()
    assert (tmp_path / PREVIOUS).exists()
    assert not (tmp_path / (LATEST + ".tmp")).exists()


def test_sigkill_mid_training_resumes_bit_identically(tmp_path):
    """The contract's named test: a hard kill, then bit-identical continuation."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    ref_trace = tmp_path / "reference.trace"
    victim_trace = tmp_path / "victim.trace"

    def run(mode, trace, wait=True):
        proc = subprocess.Popen(
            [sys.executable, str(HELPER), mode, str(ckpt), str(trace), "8", "4"],
            cwd=Path(__file__).parent.parent,
        )
        if wait:
            assert proc.wait(timeout=300) == 0
        return proc

    run("reference", ref_trace)

    victim = run("victim", victim_trace, wait=False)
    deadline = time.monotonic() + 300
    while not (ckpt / "READY").exists():
        assert time.monotonic() < deadline, "victim never reached its checkpoint"
        assert victim.poll() is None, "victim exited before it could be killed"
        time.sleep(0.05)
    os.kill(victim.pid, signal.SIGKILL)
    assert victim.wait(timeout=60) == -signal.SIGKILL

    run("resume", victim_trace)

    ref_lines = ref_trace.read_text().splitlines()
    victim_lines = victim_trace.read_text().splitlines()
    assert len(ref_lines) == 8
    # The victim recorded at least steps 1..4; the resumed process appended
    # 5..8. Steps after the checkpoint must match the reference exactly, and
    # every pre-kill step must too.
    resumed_tail = victim_lines[-4:]
    assert resumed_tail == ref_lines[4:]
    assert victim_lines[:4] == ref_lines[:4]
