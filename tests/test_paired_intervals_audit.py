"""Integration test for the paired-intervals audit over committed result JSONs.

Guards the headline reproduction: the Cartpole size-5 CEM x TD-MPC2 cell (the
LEARNED OUTPERFORMS cell) must reproduce the paper's paired-bootstrap CI and the
agreement of the three interval estimators on clearing zero. Reads committed
per-seed data; no GPU.
"""

from __future__ import annotations

import pytest

from experiments.paired_intervals_audit import CELLS, audit_cell


def _cell(label_contains: str) -> dict:
    for label, env_dir, prefix, size, learned_key in CELLS:
        if label_contains in label:
            return audit_cell(env_dir, prefix, size, learned_key)
    raise AssertionError(f"no cell matching {label_contains!r}")


def test_learned_outperforms_cell_reproduces_paper():
    c = _cell("Cartpole-swingup | CEM | TD-MPC2")
    assert c["n"] == 30
    assert c["oracle_rate"] == pytest.approx(0.467, abs=2e-3)
    assert c["learned_rate"] == pytest.approx(0.733, abs=2e-3)
    # Paired bootstrap reproduces the paper's reported CI for this cell.
    assert c["paired_bootstrap_ci"] == pytest.approx([-0.50, -0.03], abs=0.02)
    # All three interval estimators clear zero (CI_hi < 0).
    assert c["ac_ci"][1] < 0
    assert c["paired_bootstrap_ci"][1] < 0
    assert c["newcombe_ci"][1] < 0
    # McNemar exact (discordant 4 vs 12) is borderline at n=30.
    assert c["mcnemar"]["p_value"] == pytest.approx(0.0768, abs=2e-3)


def test_reacher_cells_clear_zero_under_all_interval_methods():
    for label in (
        "Reacher-easy | random-shooting | MLP-on-TD-MPC2",
        "Reacher-easy | random-shooting | TD-MPC2",
        "Reacher-easy | CEM | MLP-on-TD-MPC2",
        "Reacher-easy | CEM | TD-MPC2",
    ):
        c = _cell(label)
        # Positive gaps (MODEL BOTTLENECK); lower bound clears zero under each.
        assert c["ac_ci"][0] > 0, label
        assert c["paired_bootstrap_ci"][0] > 0, label
        assert c["newcombe_ci"][0] > 0, label
