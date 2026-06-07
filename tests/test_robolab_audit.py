"""Tests for the RoboLab-120 calibration audit (Stage 0 of the policy case study).

The audit is a no-GPU reading of published leaderboard numbers, so the tests
pin (a) that the transcribed overall rates and protocol match the cited source,
and (b) the qualitative calibration finding: the top of the leaderboard is
statistically resolved while the closest tail pair is within noise at the
reported sample size.
"""

from __future__ import annotations

from experiments.robolab_audit.audit import (
    N_OVERALL,
    OVERALL,
    audit_scope,
    n_to_separate,
)


def _pair(scope: dict, better: str, worse: str) -> dict:
    for row in scope["pairs"]:
        if row["better"] == better and row["worse"] == worse:
            return row
    raise AssertionError(f"pair {better} vs {worse} not found")


def test_published_numbers_and_protocol_match_source():
    # arXiv:2604.09860: N=10 episodes/task, 120 tasks -> n=1200 overall.
    assert N_OVERALL == 1200
    assert OVERALL == {
        "pi0.5": 0.280,
        "pi0-FAST": 0.155,
        "GR00T N1.6": 0.072,
        "pi0": 0.050,
        "PaliGemma": 0.034,
    }


def test_overall_ranking_order():
    scope = audit_scope(OVERALL, N_OVERALL)
    assert scope["ranking"] == ["pi0.5", "pi0-FAST", "GR00T N1.6", "pi0", "PaliGemma"]


def test_leaderboard_top_is_resolved():
    scope = audit_scope(OVERALL, N_OVERALL)
    top = _pair(scope, "pi0.5", "pi0-FAST")
    assert top["separable"] is True
    assert top["ac_ci"][0] > 0  # interval clears zero
    assert top["n_to_separate"] is None


def test_leaderboard_tail_is_within_noise():
    # pi0 vs PaliGemma: a +1.6pp gap at n=1200 does not clear zero.
    scope = audit_scope(OVERALL, N_OVERALL)
    tail = _pair(scope, "pi0", "PaliGemma")
    assert tail["separable"] is False
    assert tail["ac_ci"][0] <= 0 <= tail["ac_ci"][1]
    # The audit reports the per-arm n that would actually resolve it...
    assert tail["n_to_separate"] is not None and tail["n_to_separate"] > N_OVERALL


def test_n_to_separate_actually_separates():
    # The returned n must make the gate fire (not a half-width proxy that
    # undershoots), and one fewer episode must not.
    from wmel.metrics import detectable_gap_at_n

    n = n_to_separate(0.050, 0.034)
    assert n is not None
    assert detectable_gap_at_n(0.050, 0.034, n) is True
    assert detectable_gap_at_n(0.050, 0.034, n - 1) is False


def test_identical_rates_never_separate():
    assert n_to_separate(0.05, 0.05) is None


def test_audit_scope_structure_and_pair_count():
    scope = audit_scope(OVERALL, N_OVERALL)
    assert len(scope["pairs"]) == 10  # C(5, 2)
    for row in scope["pairs"]:
        assert row["rate_better"] >= row["rate_worse"]
        assert set(row) >= {"better", "worse", "raw_gap", "ac_ci", "separable"}


def test_not_every_pair_is_resolved():
    # The headline finding: the leaderboard is not fully resolved at n=1200.
    scope = audit_scope(OVERALL, N_OVERALL)
    separable = sum(1 for r in scope["pairs"] if r["separable"])
    assert 0 < separable < len(scope["pairs"])
    # Specifically, the one unresolved pair is the closest tail pair.
    unresolved = [(r["better"], r["worse"]) for r in scope["pairs"] if not r["separable"]]
    assert unresolved == [("pi0", "PaliGemma")]
