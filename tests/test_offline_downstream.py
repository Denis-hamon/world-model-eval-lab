"""Tests for the offline->downstream correlation analysis (stdlib, no GPU).

Exercises correlate_bundle on synthetic bundles: a metric that tracks the
downstream target should correlate and clear zero; a constant metric should be
skipped; non-finite (nan) values should be dropped per metric; cell descriptors
should not be treated as metrics.
"""

from __future__ import annotations

import pytest

from experiments.offline_downstream.correlate import correlate_bundle


def _row(result: dict, metric: str) -> dict:
    for r in result["metrics"]:
        if r["metric"] == metric:
            return r
    raise AssertionError(f"metric {metric!r} not in result")


def test_predictive_metric_clears_zero_and_descriptors_ignored():
    # m_err decreases as success increases -> perfect negative rank correlation;
    # m_const is constant -> degenerate -> skipped; epochs/seed are descriptors.
    cells = [
        {"epochs": e, "seed": 0, "m_err": 1.0 - i / 10.0, "m_const": 0.5, "success_rate": i / 10.0}
        for i, e in enumerate([2, 4, 8, 16, 32, 64, 128, 256])
    ]
    res = correlate_bundle(cells, "success_rate", n_boot=500, seed=0)
    err = _row(res, "m_err")
    assert err["n"] == 8
    assert err["rho"] < -0.99 and err["clears_zero"] is True
    # constant metric -> bootstrap degenerate -> reported skipped, not correlated.
    assert "skipped" in _row(res, "m_const")
    # descriptors are not treated as metrics.
    assert {"epochs", "seed"}.isdisjoint(r["metric"] for r in res["metrics"])


def test_nonfinite_values_are_dropped_per_metric():
    cells = [
        {"m": v, "success_rate": s}
        for v, s in [
            (0.9, 0.0), (0.7, 0.2), (float("nan"), 0.4), (0.4, 0.6),
            (0.3, 0.7), (float("nan"), 0.8), (0.1, 1.0),
        ]
    ]
    res = correlate_bundle(cells, "success_rate", n_boot=500, seed=0)
    m = _row(res, "m")
    assert m["n"] == 5  # the two nan cells dropped
    assert m["rho"] < 0  # still a negative (error-like) relationship


def test_common_subset_is_equal_n_for_fair_comparison():
    # m_full is finite everywhere (n=6); m_partial is nan on 2 cells (n=4). The
    # per-metric rows use different n; the common_subset restricts BOTH to the
    # cells where every metric is finite, so they are compared at equal n.
    cells = [
        {"m_full": 1.0 - i / 10.0, "m_partial": (float("nan") if i in (0, 1) else 0.5 - i / 20.0),
         "success_rate": i / 10.0}
        for i in range(6)
    ]
    res = correlate_bundle(cells, "success_rate", n_boot=300, seed=0)
    assert _row(res, "m_full")["n"] == 6
    assert _row(res, "m_partial")["n"] == 4
    cs = res["common_subset"]
    assert cs["n"] == 4  # cells 2..5 where both metrics are finite
    assert {r["metric"] for r in cs["metrics"]} == {"m_full", "m_partial"}
    assert all(r.get("n") == 4 for r in cs["metrics"] if "n" in r)
    assert "comparability_note" in res


def test_too_few_usable_cells_is_skipped():
    cells = [{"m": 1.0, "success_rate": 0.0}, {"m": 0.5, "success_rate": 1.0}]
    res = correlate_bundle(cells, "success_rate", n_boot=100, seed=0)
    assert "skipped" in _row(res, "m")


def test_kendall_method_supported():
    cells = [{"m": 1.0 - i / 10.0, "success_rate": i / 10.0} for i in range(6)]
    res = correlate_bundle(cells, "success_rate", method="kendall", n_boot=300, seed=0)
    assert res["method"] == "kendall"
    assert _row(res, "m")["rho"] < -0.9


def test_maze_quality_sweep_cell_smoke():
    # Needs torch ([learned]); skipped in the stdlib-only CI job.
    pytest.importorskip("torch")
    from experiments.offline_downstream.maze_quality_sweep import run_cell

    r = run_cell(epochs=2, seed=0, episodes=2, horizon=15)
    assert set(r) >= {
        "epochs", "seed", "m1_mismatch", "m2_kstep_divergence",
        "m3_action_agreement", "success_rate",
    }
    assert 0.0 <= r["success_rate"] <= 1.0
    assert 0.0 <= r["m1_mismatch"] <= 1.0
