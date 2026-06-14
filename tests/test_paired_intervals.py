"""Tests for the paired-binary interval methods (McNemar, Newcombe, Holm).

These complement the independent-proportions Agresti-Caffo interval with the
paired-design-correct estimators. The strongest checks are closed-form: McNemar
exact p-values reduce to a binomial on the discordant pairs, and Holm is a
deterministic step-down on a known family.
"""

from __future__ import annotations

import pytest

from wmel.metrics import (
    EpisodeResult,
    holm_correction,
    mcnemar_exact,
    newcombe_paired_diff_ci,
)


def _paired(both: int, oracle_only: int, learned_only: int, neither: int):
    """Build index-aligned (oracle, learned) arms from a 2x2 paired table."""
    o = [True] * both + [True] * oracle_only + [False] * learned_only + [False] * neither
    l = [True] * both + [False] * oracle_only + [True] * learned_only + [False] * neither
    mk = lambda s: EpisodeResult(success=s, steps=0)
    return [mk(s) for s in o], [mk(s) for s in l]


# --- McNemar exact ----------------------------------------------------------

def test_mcnemar_counts_and_learned_outperforms_cell():
    # The Cartpole size-5 CEM x TD-MPC2 cell: 2x2 table both=10, oracle-only=4,
    # learned-only=12, neither=4 (n=30). Discordant split 4/12.
    oracle, learned = _paired(both=10, oracle_only=4, learned_only=12, neither=4)
    r = mcnemar_exact(oracle, learned)
    assert (r.both, r.oracle_only, r.learned_only, r.neither) == (10, 4, 12, 4)
    assert r.n_discordant == 16
    # Two-sided exact: 2 * P(Binomial(16, 1/2) <= 4) = 2 * 2517/65536.
    assert r.p_value == pytest.approx(2 * 2517 / 65536, abs=1e-9)
    assert r.p_value == pytest.approx(0.0768, abs=1e-3)
    # Borderline at n=30: does NOT reject at alpha=0.05 even though AC clears 0.
    assert r.p_value > 0.05


def test_mcnemar_boundary_oracle_always_succeeds():
    # Reacher-like: oracle 30/30, learned 24/30, all discordant are oracle-only
    # (learned never wins where oracle loses). Reduces to a one-sided binomial.
    oracle, learned = _paired(both=24, oracle_only=6, learned_only=0, neither=0)
    r = mcnemar_exact(oracle, learned)
    assert r.n_discordant == 6
    # 2 * P(Binomial(6, 1/2) = 0) = 2 * 1/64.
    assert r.p_value == pytest.approx(2 / 64, abs=1e-9)
    assert r.p_value < 0.05  # significant -> the gap is real


def test_mcnemar_symmetric_is_not_significant():
    oracle, learned = _paired(both=0, oracle_only=5, learned_only=5, neither=20)
    r = mcnemar_exact(oracle, learned)
    assert r.p_value == 1.0


def test_mcnemar_no_discordant_pairs_is_one():
    oracle, learned = _paired(both=10, oracle_only=0, learned_only=0, neither=10)
    r = mcnemar_exact(oracle, learned)
    assert r.n_discordant == 0
    assert r.p_value == 1.0


def test_mcnemar_rejects_unequal_and_empty():
    o, l = _paired(1, 1, 1, 1)
    with pytest.raises(ValueError):
        mcnemar_exact(o, l[:-1])
    with pytest.raises(ValueError):
        mcnemar_exact([], [])


# --- Newcombe paired difference CI ------------------------------------------

def test_newcombe_point_estimate_and_bracketing():
    oracle, learned = _paired(both=10, oracle_only=4, learned_only=12, neither=4)
    diff, lo, hi = newcombe_paired_diff_ci(oracle, learned)
    # diff = p_oracle - p_learned = 14/30 - 22/30.
    assert diff == pytest.approx(14 / 30 - 22 / 30, abs=1e-9)
    assert -1.0 <= lo < diff < hi <= 1.0
    # Negative gap (learned ahead); the interval sits in the negative region.
    assert lo < 0 and hi < 0.1


def test_newcombe_symmetric_interval_around_zero():
    # Equal marginals -> diff 0, phi 0, interval symmetric about 0.
    oracle, learned = _paired(both=5, oracle_only=5, learned_only=5, neither=5)
    diff, lo, hi = newcombe_paired_diff_ci(oracle, learned)
    assert diff == pytest.approx(0.0, abs=1e-12)
    assert lo == pytest.approx(-hi, abs=1e-9)
    assert lo < 0 < hi


def test_newcombe_boundary_stays_bounded():
    # Oracle 30/30 (boundary): Wilson components do not collapse, CI stays finite.
    oracle, learned = _paired(both=24, oracle_only=6, learned_only=0, neither=0)
    diff, lo, hi = newcombe_paired_diff_ci(oracle, learned)
    assert diff == pytest.approx(0.2, abs=1e-9)
    assert -1.0 <= lo < diff < hi <= 1.0


def test_newcombe_rejects_unequal_and_empty():
    o, l = _paired(1, 1, 1, 1)
    with pytest.raises(ValueError):
        newcombe_paired_diff_ci(o[:-1], l)
    with pytest.raises(ValueError):
        newcombe_paired_diff_ci([], [])


# --- Holm-Bonferroni --------------------------------------------------------

def test_holm_known_family():
    assert holm_correction([0.01, 0.02, 0.04]) == pytest.approx([0.03, 0.04, 0.04])


def test_holm_preserves_input_order():
    # Same family, permuted: adjusted values returned in the input order.
    assert holm_correction([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_holm_monotone_and_capped_at_one():
    assert holm_correction([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_holm_empty_and_invalid():
    assert holm_correction([]) == []
    with pytest.raises(ValueError):
        holm_correction([0.5, 1.2])


def test_holm_is_less_conservative_than_bonferroni():
    # The largest adjusted p under Holm never exceeds plain Bonferroni (m*p).
    raw = [0.01, 0.02, 0.03, 0.5]
    adj = holm_correction(raw)
    assert all(a <= len(raw) * p + 1e-12 for a, p in zip(adj, raw))
