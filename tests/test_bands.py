import math

import pytest

from retention_lab.battery.scoring import TaskResult
from retention_lab.metrics.bands import (
    band_value,
    bootstrap_component,
    seed_component,
    seed_sigma,
    verdict,
    verdicts_for_rows,
)


def acc_result(per_item, chance=0.25):
    value = sum(per_item) / len(per_item)
    return TaskResult("t", "accuracy", value, chance, len(per_item), tuple(per_item))


def test_seed_sigma_recovers_known_scale():
    # A single paired difference d gives sigma = |d| / sqrt(2).
    assert seed_sigma([0.2]) == pytest.approx(0.2 / math.sqrt(2))
    assert seed_sigma([0.0, 0.0]) == 0.0


def test_seed_component_shrinks_with_noise():
    loud = seed_component([0.2, 0.3])
    quiet = seed_component([0.02, 0.03])
    assert quiet < loud
    assert seed_component([0.1]) == pytest.approx(1.96 * math.sqrt(2) * 0.1 / math.sqrt(2))


def test_bootstrap_is_reproducible_and_zero_for_constant_scores():
    constant = acc_result([1.0] * 50)
    assert bootstrap_component([constant], "cap", seed=7, resamples=200) == 0.0
    mixed = acc_result([1.0, 0.0] * 25)
    a = bootstrap_component([mixed], "cap", seed=7, resamples=500)
    b = bootstrap_component([mixed], "cap", seed=7, resamples=500)
    assert a == b
    assert a > 0.0


def test_bootstrap_shrinks_with_sample_size():
    small = acc_result([1.0, 0.0] * 10)
    large = acc_result([1.0, 0.0] * 200)
    w_small = bootstrap_component([small], "cap", seed=7, resamples=500)
    w_large = bootstrap_component([large], "cap", seed=7, resamples=500)
    assert w_large < w_small


def test_bootstrap_stream_is_stable_per_capability():
    # Quantile half-widths from different streams can legitimately coincide
    # on the discrete grid of resampled means, so independence of streams is
    # asserted at the RNG level in test_seeding; here only stability matters.
    mixed = acc_result([1.0, 0.0] * 25)
    a = bootstrap_component([mixed], "alpha", seed=7, resamples=300)
    assert bootstrap_component([mixed], "alpha", seed=7, resamples=300) == a


def test_band_is_max_of_components():
    assert band_value(0.02, 0.05) == 0.05
    assert band_value(0.07, 0.05) == 0.07


def test_verdict_boundary_counts_as_no_effect():
    assert verdict(0.05, 0.05) == "no effect"
    assert verdict(-0.05, 0.05) == "no effect"
    assert verdict(0.0500001, 0.05) == "above band"
    assert verdict(-0.0500001, 0.05) == "below band"
    with pytest.raises(ValueError):
        verdict(0.1, -0.01)


def test_verdicts_for_rows_requires_full_coverage():
    out = verdicts_for_rows({"a": 0.1}, {"a": 0.2})
    assert out == {"a": "no effect"}
    with pytest.raises(ValueError, match="no band"):
        verdicts_for_rows({"a": 0.1, "b": 0.3}, {"a": 0.2})
