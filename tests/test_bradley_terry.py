"""Unit tests for the paired Bradley-Terry ranking utility.

The ranking generalises the two-arm paired comparison (``paired_bootstrap_gap_ci``)
to an N-model leaderboard with uncertainty on the ordering itself. The strongest
checks are the closed-form ones: for two players the Bradley-Terry strength is
exactly the (prior-smoothed) empirical win fraction, and a clean transitive set
must recover its obvious ordering.
"""

from __future__ import annotations

import pytest

from wmel.metrics import EpisodeResult, paired_bradley_terry_ranking


def _wins(success_pattern: list[bool]) -> list[EpisodeResult]:
    return [EpisodeResult(success=s, steps=10 if s else 50) for s in success_pattern]


def test_two_player_strength_is_empirical_win_fraction():
    # A beats B on 30 episodes, B beats A on 10, no ties. For two players the
    # unregularised (prior=0) Bradley-Terry MLE is exactly the win fraction.
    results = {
        "A": _wins([True] * 30 + [False] * 10),
        "B": _wins([False] * 30 + [True] * 10),
    }
    ranking = paired_bradley_terry_ranking(results, prior=0.0, n_boot=200)
    assert ranking.strengths["A"] == pytest.approx(0.75, abs=1e-6)
    assert ranking.strengths["B"] == pytest.approx(0.25, abs=1e-6)
    assert ranking.ranks == {"A": 1, "B": 2}
    # Raw battle counts are auditable on the result.
    assert ranking.win_matrix["A"]["B"] == 30
    assert ranking.win_matrix["B"]["A"] == 10
    assert ranking.n_decisive["A"]["B"] == 40


def test_strengths_sum_to_one():
    results = {
        "A": _wins([True, True, False, True]),
        "B": _wins([False, True, True, False]),
        "C": _wins([False, False, False, True]),
    }
    ranking = paired_bradley_terry_ranking(results, n_boot=200)
    assert sum(ranking.strengths.values()) == pytest.approx(1.0, abs=1e-9)


def test_transitive_ordering_recovered():
    # 5 episodes (A,B,C) = (1,0,0) and 5 = (1,1,0):
    #   A beats B 5-0, A beats C 10-0, B beats C 5-0.  A > B > C.
    results = {
        "A": _wins([True] * 10),
        "B": _wins([False] * 5 + [True] * 5),
        "C": _wins([False] * 10),
    }
    ranking = paired_bradley_terry_ranking(results, n_boot=200)
    assert ranking.ranks == {"A": 1, "B": 2, "C": 3}
    assert ranking.strengths["A"] > ranking.strengths["B"] > ranking.strengths["C"]


def test_identical_models_tie():
    pattern = [True, False, True, True, False]
    results = {"A": _wins(pattern), "B": _wins(pattern)}
    ranking = paired_bradley_terry_ranking(results, n_boot=200)
    # Every episode is a tie -> no decisive comparisons -> equal strength.
    assert ranking.strengths["A"] == pytest.approx(ranking.strengths["B"])
    assert ranking.ranks["A"] == ranking.ranks["B"] == 1
    assert ranking.n_decisive["A"]["B"] == 0


def test_default_prior_handles_separation():
    # A wins every decisive comparison (never loses); the unregularised MLE
    # diverges, but the default prior keeps it finite and ranks A first.
    results = {
        "A": _wins([True] * 40),
        "B": _wins([False] * 40),
    }
    ranking = paired_bradley_terry_ranking(results, n_boot=200)  # prior=0.5 default
    assert ranking.strengths["A"] > 0.95
    assert all(0.0 <= s <= 1.0 for s in ranking.strengths.values())
    assert ranking.ranks["A"] == 1
    # A wins in every bootstrap resample, so its rank interval is a point.
    assert ranking.rank_ci["A"] == (1, 1)


def test_prior_zero_rejects_separation():
    # A beats B and C decisively; B and C never win. The unregularised MLE is
    # not identifiable, so prior=0 must refuse rather than return an arbitrary
    # ranking.
    results = {
        "A": _wins([True] * 5),
        "B": _wins([False] * 5),
        "C": _wins([False] * 5),
    }
    with pytest.raises(ValueError):
        paired_bradley_terry_ranking(results, prior=0.0)
    # The default prior makes the same data well-defined and ranks A first.
    ranking = paired_bradley_terry_ranking(results, n_boot=100)
    assert ranking.ranks["A"] == 1
    assert all(0.0 <= s <= 1.0 for s in ranking.strengths.values())


def test_prior_zero_works_on_connected_graph():
    # A 3-cycle (A>B, B>C, C>A), every model with one win and one loss: the
    # comparison graph is connected, so prior=0 is well-defined and symmetric.
    results = {
        "A": _wins([True, False, False]),
        "B": _wins([False, True, False]),
        "C": _wins([False, False, True]),
    }
    ranking = paired_bradley_terry_ranking(results, prior=0.0, n_boot=100)
    for name in ("A", "B", "C"):
        assert ranking.strengths[name] == pytest.approx(1 / 3, abs=1e-6)
        assert ranking.ranks[name] == 1


def test_point_estimate_lies_within_bootstrap_intervals():
    results = {
        "A": _wins([True] * 18 + [False] * 2),
        "B": _wins([True] * 12 + [False] * 8),
        "C": _wins([True] * 6 + [False] * 14),
    }
    ranking = paired_bradley_terry_ranking(results, n_boot=1000, seed=0)
    for name in ("A", "B", "C"):
        lo, hi = ranking.strength_ci[name]
        assert lo <= ranking.strengths[name] <= hi
        r_lo, r_hi = ranking.rank_ci[name]
        assert r_lo <= ranking.ranks[name] <= r_hi


def test_near_tie_has_overlapping_rank_intervals():
    # A barely edges B (11-9 decisive): the ranking cannot separate them, so
    # the rank intervals overlap -- the leaderboard analogue of INCONCLUSIVE.
    results = {
        "A": _wins([True] * 11 + [False] * 9),
        "B": _wins([False] * 11 + [True] * 9),
    }
    ranking = paired_bradley_terry_ranking(results, n_boot=1000, seed=0)
    a_lo, a_hi = ranking.rank_ci["A"]
    b_lo, b_hi = ranking.rank_ci["B"]
    assert a_lo <= b_hi and b_lo <= a_hi  # intervals overlap


def test_cost_outcome_with_higher_is_better_false():
    # Outcome is steps-to-go (lower is better); A is always cheaper than B.
    results = {
        "A": [EpisodeResult(success=True, steps=10) for _ in range(8)],
        "B": [EpisodeResult(success=True, steps=50) for _ in range(8)],
    }
    ranking = paired_bradley_terry_ranking(
        results, outcome=lambda r: float(r.steps), higher_is_better=False, n_boot=100
    )
    assert ranking.ranks["A"] == 1
    assert ranking.ranks["B"] == 2


def test_deterministic_given_seed():
    results = {
        "A": _wins([True] * 14 + [False] * 6),
        "B": _wins([True] * 10 + [False] * 10),
        "C": _wins([True] * 7 + [False] * 13),
    }
    one = paired_bradley_terry_ranking(results, n_boot=500, seed=7)
    two = paired_bradley_terry_ranking(results, n_boot=500, seed=7)
    assert one.strength_ci == two.strength_ci
    assert one.rank_ci == two.rank_ci
    assert one.strengths == two.strengths


def test_groups_stratified_bootstrap_runs():
    results = {
        "A": _wins([True] * 6 + [False] * 6),
        "B": _wins([False] * 6 + [True] * 6),
    }
    groups = ["easy"] * 6 + ["hard"] * 6
    ranking = paired_bradley_terry_ranking(results, groups=groups, n_boot=300)
    assert ranking.n_episodes == 12
    assert set(ranking.ranks) == {"A", "B"}
    assert all(r >= 1 for r in ranking.ranks.values())


def test_rejects_fewer_than_two_models():
    with pytest.raises(ValueError):
        paired_bradley_terry_ranking({"A": _wins([True, False])})


def test_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        paired_bradley_terry_ranking({"A": _wins([True, False]), "B": _wins([True])})


def test_rejects_empty():
    with pytest.raises(ValueError):
        paired_bradley_terry_ranking({"A": [], "B": []})


def test_rejects_mismatched_groups_length():
    results = {"A": _wins([True, False, True]), "B": _wins([False, True, False])}
    with pytest.raises(ValueError):
        paired_bradley_terry_ranking(results, groups=["x", "y"])


def test_rejects_bad_alpha_and_nboot_and_prior():
    results = {"A": _wins([True, False]), "B": _wins([False, True])}
    with pytest.raises(ValueError):
        paired_bradley_terry_ranking(results, alpha=0.0)
    with pytest.raises(ValueError):
        paired_bradley_terry_ranking(results, n_boot=0)
    with pytest.raises(ValueError):
        paired_bradley_terry_ranking(results, prior=-1.0)
