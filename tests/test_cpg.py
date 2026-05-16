"""Tests for the Counterfactual Planning Gap (CPG) metric and verdict.

The CI uses the Agresti-Caffo "plus-4" adjustment, which keeps the variance
positive at p in {0, 1} and is well-calibrated for small n. These tests
exercise the formula on synthetic EpisodeResult lists - independent of any
env, model, or planner. The Acrobot-specific oracle dynamics is exercised
separately in `tests/test_dmc_acrobot.py`.
"""

from __future__ import annotations

import math

import pytest

from wmel.metrics import (
    CPG_VERDICT_INCONCLUSIVE,
    CPG_VERDICT_LEARNED_OUTPERFORMS,
    CPG_VERDICT_MODEL_AS_GOOD_AS_ORACLE,
    CPG_VERDICT_MODEL_BOTTLENECK,
    CPG_VERDICT_PLANNER_BOTTLENECK,
    CPGResult,
    EpisodeResult,
    counterfactual_planning_gap,
    cpg_verdict,
)


def _episode(success: bool) -> EpisodeResult:
    return EpisodeResult(success=success, steps=100, planning_latencies_ms=(1.0,))


def _ac_ci_at(s_o: int, n_o: int, s_l: int, n_l: int, z: float = 1.96) -> tuple[float, float]:
    """Reference Agresti-Caffo computation for cross-checking."""
    tp_o = (s_o + 1) / (n_o + 2)
    tp_l = (s_l + 1) / (n_l + 2)
    tilde_gap = tp_o - tp_l
    se = math.sqrt(tp_o * (1 - tp_o) / (n_o + 2) + tp_l * (1 - tp_l) / (n_l + 2))
    hw = z * se
    return (tilde_gap - hw, tilde_gap + hw)


def test_cpg_reports_raw_gap_not_ac_adjusted_gap() -> None:
    """The `gap` field is the raw observed difference. The CI uses AC.
    With s_o=3, s_l=0, n=10 each: gap = 0.30 (raw), tilde_gap = 4/12 -
    1/12 = 0.25 - which becomes the centre of the CI, not `gap`.
    """
    oracle = [_episode(True)] * 3 + [_episode(False)] * 7
    learned = [_episode(False)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg.gap == pytest.approx(0.30)
    lo, hi = _ac_ci_at(3, 10, 0, 10)
    assert cpg.gap_ci_low == pytest.approx(lo)
    assert cpg.gap_ci_high == pytest.approx(hi)
    # AC CI on this small-sample data crosses zero - that's the point.
    assert cpg.gap_ci_low < 0 < cpg.gap_ci_high


def test_cpg_ci_is_non_degenerate_at_boundary_proportions() -> None:
    """With AC, both planners at 100% (or both at 0%) still produces a
    non-degenerate CI, unlike the Wald approximation that variance-
    collapses there."""
    # Both at 0% with n=10:
    oracle = [_episode(False)] * 10
    learned = [_episode(False)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg.gap == 0.0
    # AC CI width > 0
    assert (cpg.gap_ci_high - cpg.gap_ci_low) > 0.1

    # Both at 100% with n=10:
    oracle = [_episode(True)] * 10
    learned = [_episode(True)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg.gap == 0.0
    assert (cpg.gap_ci_high - cpg.gap_ci_low) > 0.1


def test_cpg_strong_signal_lower_bound_above_zero() -> None:
    """With s_o=10, s_l=0, n=10 each, the AC CI lower bound IS above zero,
    so this is a legitimate MODEL BOTTLENECK signal."""
    oracle = [_episode(True)] * 10
    learned = [_episode(False)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg.gap == 1.0
    assert cpg.gap_ci_low > 0
    # AC ratio for max-extreme: tp_o = 11/12, tp_l = 1/12, diff = 10/12 = 0.833,
    # half-width = 1.96 * sqrt(11*1/12^3 + 1*11/12^3) = 1.96 * sqrt(22/1728) ~ 0.221.
    expected_lo, expected_hi = _ac_ci_at(10, 10, 0, 10)
    assert cpg.gap_ci_low == pytest.approx(expected_lo)
    assert cpg.gap_ci_high == pytest.approx(expected_hi)


def test_cpg_can_be_negative_when_learned_outperforms() -> None:
    oracle = [_episode(False)] * 10
    learned = [_episode(True)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg.gap == -1.0
    assert cpg.gap_ci_high < 0


def test_cpg_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        counterfactual_planning_gap([], [_episode(True)])
    with pytest.raises(ValueError):
        counterfactual_planning_gap([_episode(True)], [])


def test_cpg_uneven_n_handled() -> None:
    oracle = [_episode(True)] * 50 + [_episode(False)] * 50
    learned = [_episode(True)] * 15 + [_episode(False)] * 5
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg.n_episodes_oracle == 100
    assert cpg.n_episodes_learned == 20
    assert cpg.gap == pytest.approx(-0.25)
    # CI strictly negative under AC because the small sample (n=20) is
    # heavily weighted toward learned successes (15/20 = 0.75).
    assert cpg.gap_ci_high < 0


def test_cpg_result_is_frozen_dataclass() -> None:
    cpg = counterfactual_planning_gap([_episode(True)] * 5, [_episode(False)] * 5)
    with pytest.raises((AttributeError, TypeError)):
        cpg.gap = 0.99  # type: ignore[misc]


# ---------- Verdict branches ----------


def test_verdict_model_bottleneck_when_ci_strictly_positive() -> None:
    """Oracle 100%, learned 0%, n=10 each. AC CI lower bound > 0 -> MODEL BOTTLENECK."""
    oracle = [_episode(True)] * 10
    learned = [_episode(False)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg_verdict(cpg) == CPG_VERDICT_MODEL_BOTTLENECK


def test_verdict_learned_outperforms_when_ci_strictly_negative() -> None:
    oracle = [_episode(False)] * 10
    learned = [_episode(True)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg_verdict(cpg) == CPG_VERDICT_LEARNED_OUTPERFORMS


def test_verdict_planner_bottleneck_when_both_zero_and_ci_crosses() -> None:
    """Both planners 0/10 -> CI crosses 0, both rates near 0 -> PLANNER BOTTLENECK."""
    oracle = [_episode(False)] * 10
    learned = [_episode(False)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg_verdict(cpg) == CPG_VERDICT_PLANNER_BOTTLENECK


def test_verdict_model_as_good_as_oracle_when_both_perfect() -> None:
    """Both planners 10/10 -> CI crosses 0, both rates near 1 -> MODEL AS GOOD AS ORACLE."""
    oracle = [_episode(True)] * 10
    learned = [_episode(True)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg_verdict(cpg) == CPG_VERDICT_MODEL_AS_GOOD_AS_ORACLE


def test_verdict_inconclusive_when_ci_crosses_zero_in_middle_regime() -> None:
    """The headline-Acrobot case: oracle 3/10, learned 0/10. Raw gap is
    +0.30 but the AC CI crosses zero. Verdict should be INCONCLUSIVE -
    the data is suggestive of a model bottleneck but not statistically
    distinguishable from zero at n=10."""
    oracle = [_episode(True)] * 3 + [_episode(False)] * 7
    learned = [_episode(False)] * 10
    cpg = counterfactual_planning_gap(oracle, learned)
    assert cpg_verdict(cpg) == CPG_VERDICT_INCONCLUSIVE


def test_verdict_strings_are_stable_constants() -> None:
    """The verdict text is exported as named constants so downstream
    consumers (and the committed cpg.json) cannot drift silently."""
    assert CPG_VERDICT_MODEL_BOTTLENECK == "MODEL BOTTLENECK"
    assert CPG_VERDICT_LEARNED_OUTPERFORMS == "LEARNED OUTPERFORMS ORACLE"
    assert CPG_VERDICT_PLANNER_BOTTLENECK == "PLANNER BOTTLENECK"
    assert CPG_VERDICT_MODEL_AS_GOOD_AS_ORACLE == "MODEL AS GOOD AS ORACLE"
    assert CPG_VERDICT_INCONCLUSIVE == "INCONCLUSIVE"
