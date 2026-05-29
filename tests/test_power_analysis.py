"""Unit tests for the CPG power-analysis helpers.

These functions answer "how many episodes before the verdict can fire / before
the interval reaches a target precision?" without running a benchmark. They
reuse the same Agresti-Caffo plus-4 standard error as the CPG CI, so the
strongest check is that they reproduce the half-widths of the published runs.
"""

from __future__ import annotations

import math

import pytest

from wmel.metrics import (
    ac_ci_half_width,
    detectable_gap_at_n,
    required_n_for_half_width,
)


def test_half_width_reproduces_published_acrobot_pooled_150():
    # results/dmc_acrobot/cpg_sweep.json: oracle 40/150 = 0.267, learned 0/150,
    # published AC CI [+0.191, +0.335] -> half-width ~0.072.
    hw = ac_ci_half_width(0.267, 0.0, 150)
    assert hw == pytest.approx(0.072, abs=2e-3)


def test_half_width_reproduces_published_n10():
    # results/dmc_acrobot/cpg.json: oracle 0.30, learned 0.0, n=10,
    # published CI [-0.06, +0.56] -> half-width ~0.31.
    hw = ac_ci_half_width(0.30, 0.0, 10)
    assert hw == pytest.approx(0.31, abs=5e-3)


def test_half_width_reproduces_published_cartpole_size1_inconclusive():
    # results/dmc_cartpole/cem_cpg_pooled.json: oracle 0.5, learned 0.533,
    # n=30, published CI [-0.276, +0.214] -> half-width ~0.245.
    hw = ac_ci_half_width(0.5, 0.533, 30)
    assert hw == pytest.approx(0.245, abs=2e-3)


def test_half_width_is_monotone_decreasing_in_n():
    widths = [ac_ci_half_width(0.5, 0.3, n) for n in (5, 10, 50, 200, 1000)]
    assert all(a > b for a, b in zip(widths, widths[1:]))


def test_half_width_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        ac_ci_half_width(0.5, 0.3, 0)


def test_required_n_meets_target_and_is_minimal():
    target = 0.05
    n = required_n_for_half_width(0.94, 0.92, target)
    assert n is not None
    # The returned n meets the target...
    assert ac_ci_half_width(0.94, 0.92, n) <= target
    # ...and is the smallest such n: one fewer episode misses it.
    assert ac_ci_half_width(0.94, 0.92, n - 1) > target


def test_required_n_rejects_nonpositive_target():
    with pytest.raises(ValueError):
        required_n_for_half_width(0.5, 0.5, 0.0)


def test_swm_top_two_gap_is_not_detectable_at_their_n():
    # swm reports LeWorldModel 94% vs DINO-WM 92% on Push-T at n=100, seed=0,
    # with no CI. At that n the AC interval on the difference straddles zero:
    # the reported ranking gap is within noise.
    assert detectable_gap_at_n(0.94, 0.92, 100) is False
    # A wider, genuine gap (94% vs 78%) does clear zero at the same n.
    assert detectable_gap_at_n(0.94, 0.78, 100) is True


def test_detectable_consistent_with_required_n():
    # If n is large enough to make a gap detectable, the half-width has shrunk
    # below the gap magnitude; sanity-check the two notions agree directionally.
    o, l = 0.9, 0.6
    n = required_n_for_half_width(o, l, 0.10)
    assert detectable_gap_at_n(o, l, n) is True
