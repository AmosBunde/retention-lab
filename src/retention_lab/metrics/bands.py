"""The noise-band policy, frozen at M1 and applied to real runs later.

The band for a capability is the maximum of two components:

- **Seed component.** Each variant trained at two seeds yields one paired
  difference of capability scores. For a difference of two independent
  draws, Var(d) = 2 * sigma^2, so sigma is estimated as
  sqrt(mean(d^2) / 2) pooled across variants, and the component is
  1.96 * sqrt(2) * sigma, the two-sided 95 percent interval for a fresh
  seed-pair difference.
- **Eval-set component.** A bootstrap over evaluation items: tasks are
  resampled item-wise with a recorded seed, the normalized capability
  score is recomputed per resample, and the component is the half-width
  of the central 95 percent interval. For bits per byte the resampled
  statistic is the unweighted mean of per-document BPB; documents are cut
  to equal character size by the registry, so weights are near-equal, and
  this approximation is part of the frozen policy.

A delta with absolute value at or below the band is **no effect**. The
verdict is binding for tables and prose; the boundary case counts as no
effect by definition, so noise can never be promoted to a finding by
rounding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from retention_lab.battery.scoring import TaskResult
from retention_lab.metrics.retention import normalize_task
from retention_lab.utils.seeding import numpy_rng

Z_95 = 1.96


def seed_sigma(paired_diffs: Sequence[float]) -> float:
    """Pooled per-seed standard deviation from paired seed differences."""
    if not paired_diffs:
        raise ValueError("seed sigma needs at least one paired difference")
    return math.sqrt(sum(d * d for d in paired_diffs) / (2 * len(paired_diffs)))


def seed_component(paired_diffs: Sequence[float]) -> float:
    return Z_95 * math.sqrt(2) * seed_sigma(paired_diffs)


def _resampled_task_value(result: TaskResult, rng: np.random.Generator) -> float:
    per_item = np.asarray(result.per_item)
    sample = per_item[rng.integers(0, len(per_item), size=len(per_item))]
    resampled = TaskResult(
        result.task, result.metric, float(sample.mean()), result.chance,
        result.n_items, result.per_item,
    )
    return normalize_task(resampled)


def bootstrap_component(
    results: Sequence[TaskResult], capability: str, seed: int, resamples: int
) -> float:
    """95 percent half-width of the normalized capability score.

    Reproducible bit for bit from the battery bootstrap seed: the RNG stream
    is derived from the capability name, so adding a capability never
    perturbs the draws of existing ones.
    """
    if resamples < 100:
        raise ValueError("bootstrap needs at least 100 resamples")
    rng = numpy_rng(seed, f"bootstrap:{capability}")
    stats = np.empty(resamples)
    for i in range(resamples):
        values = [_resampled_task_value(r, rng) for r in results]
        stats[i] = sum(values) / len(values)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(hi - lo) / 2.0


def band_value(seed_comp: float, eval_comp: float) -> float:
    return max(seed_comp, eval_comp)


def verdict(delta: float, band: float) -> str:
    """Binding classification of a delta against the band."""
    if band < 0:
        raise ValueError("band cannot be negative")
    if abs(delta) <= band:
        return "no effect"
    return "above band" if delta > 0 else "below band"


def verdicts_for_rows(
    deltas: Mapping[str, float], bands: Mapping[str, float]
) -> dict[str, str]:
    missing = set(deltas) - set(bands)
    if missing:
        raise ValueError(f"no band for capabilities: {sorted(missing)}")
    return {cap: verdict(delta, bands[cap]) for cap, delta in deltas.items()}
