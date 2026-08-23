"""Learning-rate schedule as a pure function of the step.

Being a pure function is what makes resume trivial: the scheduler has no
state to checkpoint, only the step counter, and any step's rate is exactly
recomputable on any machine.
"""

from __future__ import annotations

import math


def lr_at(
    step: int, base_lr: float, warmup_steps: int, total_steps: int, min_ratio: float
) -> float:
    """Linear warmup to base_lr, then cosine decay to base_lr * min_ratio."""
    if not 0 <= min_ratio <= 1:
        raise ValueError("min_ratio must lie in [0, 1]")
    if warmup_steps >= total_steps:
        raise ValueError("warmup must be shorter than the run")
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)
