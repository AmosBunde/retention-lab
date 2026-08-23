"""Subprocess helper for the kill-and-resume test.

Modes: ``reference`` trains uninterrupted; ``victim`` checkpoints on a
cadence, signals readiness after the target checkpoint, then keeps training
slowly until killed; ``resume`` restores from the checkpoint directory and
continues to the end. Every mode appends one formatted loss per line to the
trace file, flushed immediately, so the parent can compare exact strings.
"""

import sys
import time
from pathlib import Path

from retention_lab.data.packing import MixtureStream
from retention_lab.kd.losses import build_objective
from retention_lab.models.toy import build_toy_lm
from retention_lab.train.checkpoint import resume_if_available, save_checkpoint
from retention_lab.train.loop import Trainer
from retention_lab.utils.seeding import numpy_rng, set_global_determinism, torch_generator

BLOCK = {"vocab_size": 16, "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 16}
TRAIN = {"steps": 8, "batch_size": 4, "lr": 0.005, "warmup_steps": 2}


def build_trainer() -> Trainer:
    set_global_determinism(7)
    tokens = numpy_rng(7, "kill-resume").integers(0, 16, size=64 * 17)
    stream = MixtureStream(
        {"corpus": tokens.reshape(64, 17)}, {"corpus": 1.0}, seed=7
    )
    student = build_toy_lm(BLOCK, torch_generator(7, "student"))
    teacher = build_toy_lm({**BLOCK, "d_model": 48}, torch_generator(7, "teacher"))
    objective = build_objective({"name": "forward_kl", "temperature": 2.0, "alpha": 0.5})
    return Trainer(
        model=student, objective=objective, stream=stream, train_cfg=TRAIN, teacher=teacher
    )


def main() -> None:
    mode, ckpt_dir, trace_file = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    total_steps, save_at = int(sys.argv[4]), int(sys.argv[5])
    trainer = build_trainer()
    if mode == "resume" and resume_if_available(trainer, ckpt_dir) is None:
        raise SystemExit("resume mode found no checkpoint")
    with open(trace_file, "a") as trace:
        while trainer.step < total_steps:
            metrics = trainer.train_step()
            trace.write(f"{trainer.step} {metrics['total']:.17g}\n")
            trace.flush()
            if mode == "victim":
                if trainer.step == save_at:
                    save_checkpoint(trainer, ckpt_dir)
                    (ckpt_dir / "READY").touch()
                time.sleep(0.15)  # give the parent time to deliver SIGKILL
    print("done")


if __name__ == "__main__":
    main()
