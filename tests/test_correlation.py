"""Tests for the rank-correlation primitives (Spearman, Kendall, bootstrap CI).

These back the offline-metric vs downstream-performance study. The strongest
checks are closed-form: perfect monotone data gives +/-1, and small hand-worked
examples pin the tie handling.
"""

from __future__ import annotations

import pytest

from wmel.metrics import (
    CorrelationResult,
    bootstrap_correlation_ci,
    kendall_tau,
    spearman_rho,
)
from wmel.metrics import _rankdata  # noqa: PLC2701 (private, tie-handling check)


# --- rank helper ------------------------------------------------------------

def test_rankdata_average_ranks_with_ties():
    assert _rankdata([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert _rankdata([5, 5, 5]) == [2.0, 2.0, 2.0]


# --- Spearman ---------------------------------------------------------------

def test_spearman_perfect_monotone():
    assert spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman_rho([1, 2, 3, 4, 5], [9, 7, 5, 3, 1]) == pytest.approx(-1.0)


def test_spearman_known_value():
    # No ties -> Spearman = Pearson on the values themselves.
    # [1,2,3,4] vs [1,3,2,4]: cov 4.0, var 5.0 each -> rho 0.8.
    assert spearman_rho([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)


def test_spearman_rejects_degenerate_and_mismatched():
    with pytest.raises(ValueError):
        spearman_rho([1, 2, 3], [5, 5, 5])  # constant -> undefined
    with pytest.raises(ValueError):
        spearman_rho([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        spearman_rho([1.0], [2.0])


# --- Kendall tau-b ----------------------------------------------------------

def test_kendall_perfect_monotone():
    assert kendall_tau([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)


def test_kendall_known_value():
    # [1,2,3,4] vs [1,3,2,4]: 5 concordant, 1 discordant, no ties -> 4/6.
    assert kendall_tau([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)


def test_kendall_tie_corrected():
    # [1,1,2] vs [1,2,2]: one concordant pair, one x-tie, one y-tie ->
    # tau_b = (1-0)/sqrt((3-1)(3-1)) = 0.5.
    assert kendall_tau([1, 1, 2], [1, 2, 2]) == pytest.approx(0.5)


def test_kendall_rejects_degenerate():
    with pytest.raises(ValueError):
        kendall_tau([1, 1, 1], [1, 2, 3])


# --- bootstrap CI -----------------------------------------------------------

def test_bootstrap_perfect_correlation_is_tight():
    r = bootstrap_correlation_ci(list(range(10)), list(range(10)), n_boot=500, seed=0)
    assert isinstance(r, CorrelationResult)
    assert r.rho == pytest.approx(1.0)
    assert r.ci_low == pytest.approx(1.0) and r.ci_high == pytest.approx(1.0)
    assert r.n_pairs == 10
    assert r.method == "spearman"


def test_bootstrap_is_deterministic_given_seed():
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    ys = [2, 1, 4, 3, 6, 5, 8, 7]
    a = bootstrap_correlation_ci(xs, ys, n_boot=400, seed=3)
    b = bootstrap_correlation_ci(xs, ys, n_boot=400, seed=3)
    assert (a.rho, a.ci_low, a.ci_high, a.n_boot) == (b.rho, b.ci_low, b.ci_high, b.n_boot)


def test_bootstrap_ci_brackets_point():
    xs = list(range(12))
    ys = [0, 1, 3, 2, 4, 6, 5, 7, 9, 8, 10, 11]  # strong but imperfect monotone
    r = bootstrap_correlation_ci(xs, ys, n_boot=1000, seed=0)
    assert r.ci_low <= r.rho <= r.ci_high
    assert 0.0 < r.rho < 1.0  # genuinely partial


def test_bootstrap_kendall_method_and_bad_method():
    r = bootstrap_correlation_ci(list(range(8)), list(range(8)), method="kendall", n_boot=200, seed=1)
    assert r.method == "kendall" and r.rho == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bootstrap_correlation_ci([1, 2, 3], [1, 2, 3], method="pearson")


def test_bootstrap_input_validation():
    with pytest.raises(ValueError):
        bootstrap_correlation_ci([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        bootstrap_correlation_ci([1.0], [2.0])
    with pytest.raises(ValueError):
        bootstrap_correlation_ci([1, 2, 3], [1, 2, 3], alpha=0.0)
    with pytest.raises(ValueError):
        bootstrap_correlation_ci([1, 2, 3], [1, 2, 3], n_boot=0)
