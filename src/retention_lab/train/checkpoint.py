"""Checkpointing with bit-identical resume and reclaim-safe writes.

A checkpoint carries everything a fresh process needs to continue as if the
kill never happened: model, objective (projection parameters), optimizer,
the step and token counters (the data cursor is the step counter by
construction), and every global RNG stream. Writes are atomic (temp file,
fsync, rename) and the previous checkpoint is kept, so a reclaim during
save can never leave the run without a loadable state: the loader falls
back from a corrupt latest to the previous one loudly.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import numpy.core.multiarray
import torch
from torch.serialization import add_safe_globals

from retention_lab.train.loop import Trainer

LATEST = "ckpt.pt"
PREVIOUS = "ckpt.prev.pt"

# Checkpoints are self-produced, but they still load with weights_only=True
# plus this minimal allowlist (needed for the numpy RNG state), so a swapped
# or tampered checkpoint file cannot execute code at load time.
add_safe_globals(
    [np.dtype, np.ndarray, numpy.core.multiarray._reconstruct, np.dtypes.UInt32DType]
)


def _rng_states() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def trainer_state(trainer: Trainer) -> dict:
    return {
        "step": trainer.step,
        "tokens_seen": trainer.tokens_seen,
        "model": trainer.model.state_dict(),
        "objective": trainer.objective.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "rng": _rng_states(),
    }


def load_trainer_state(trainer: Trainer, state: dict) -> int:
    trainer.model.load_state_dict(state["model"])
    trainer.objective.load_state_dict(state["objective"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.step = int(state["step"])
    trainer.tokens_seen = int(state["tokens_seen"])
    _restore_rng(state["rng"])
    return trainer.step


def save_checkpoint(trainer: Trainer, directory: str | Path) -> Path:
    """Atomically write the latest checkpoint, keeping the previous one."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / LATEST
    tmp = directory / (LATEST + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(trainer_state(trainer), fh)
        fh.flush()
        os.fsync(fh.fileno())
    if latest.exists():
        latest.replace(directory / PREVIOUS)
    tmp.replace(latest)
    return latest


def resume_if_available(trainer: Trainer, directory: str | Path) -> int | None:
    """Restore the newest loadable checkpoint; None means a fresh start.

    A corrupt latest (for example a reclaim mid-rename on a non-atomic
    filesystem) falls back to the previous checkpoint with a loud warning
    instead of failing the run or silently starting over.
    """
    directory = Path(directory)
    for name in (LATEST, PREVIOUS):
        path = directory / name
        if not path.exists():
            continue
        try:
            state = torch.load(path, map_location=trainer.device, weights_only=True)
        except Exception as error:  # noqa: BLE001 - any unreadable file falls through
            print(f"checkpoint: {path} unreadable ({type(error).__name__}); falling back")
            continue
        step = load_trainer_state(trainer, state)
        print(f"checkpoint: resumed from {path} at step {step}")
        return step
    return None
