"""Closed-form tests for the pure offline-metric helpers used by the Stage-2
keystone (``experiments/offline_downstream/tdmpc2_offline_metrics.py``).

The helpers take arbitrary dynamics/score callables, so they are exercised here
with synthetic callables and hand-computed expectations -- no torch, no
dm_control, no checkpoints. The full cell sweep is checkpoint-gated and run on
the box; this guards the arithmetic the sweep depends on.
"""

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from experiments.offline_downstream.tdmpc2_offline_metrics import (  # noqa: E402
    _l2,
    mean_action_ranking_agreement,
    mean_kstep_divergence,
    mean_one_step_l2,
)


def test_l2_basic_and_length_check():
    assert _l2((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert _l2((1.0,), (1.0,)) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        _l2((0.0, 0.0), (1.0,))


def test_one_step_l2_constant_offset():
    # learned = oracle shifted by (3, 4) everywhere -> L2 is 5 for every pair,
    # so the mean over any state x action grid is exactly 5.
    oracle = lambda s, a: s
    learned = lambda s, a: (s[0] + 3.0, s[1] + 4.0)
    states = [(0.0, 0.0), (1.0, 1.0), (-2.0, 5.0)]
    actions = [(0.0,), (1.0,)]
    assert mean_one_step_l2(oracle, learned, states, actions) == pytest.approx(5.0)


def test_one_step_l2_zero_when_identical():
    oracle = lambda s, a: (s[0] + a[0], s[1])
    assert mean_one_step_l2(oracle, oracle, [(0.0, 0.0)], [(1.0,), (2.0,)]) == pytest.approx(0.0)


def test_one_step_l2_requires_inputs():
    with pytest.raises(ValueError):
        mean_one_step_l2(lambda s, a: s, lambda s, a: s, [], [(0.0,)])
    with pytest.raises(ValueError):
        mean_one_step_l2(lambda s, a: s, lambda s, a: s, [(0.0,)], [])


def test_kstep_divergence_compounds_linearly():
    # oracle holds the state; learned drifts +1 in dim 0 each step, fed its own
    # output -> after K steps the gap is exactly K (in dim 0 only).
    oracle = lambda s, a: s
    learned = lambda s, a: (s[0] + 1.0, s[1])
    states = [(0.0, 0.0), (10.0, -3.0)]
    seq = [(0.0,), (0.0,), (0.0,)]  # K = 3
    assert mean_kstep_divergence(oracle, learned, states, seq) == pytest.approx(3.0)


def test_kstep_divergence_zero_when_identical():
    dyn = lambda s, a: (s[0] + a[0], s[1] + 1.0)
    assert mean_kstep_divergence(dyn, dyn, [(0.0, 0.0)], [(1.0,), (2.0,)]) == pytest.approx(0.0)


def test_action_ranking_perfect_agreement():
    # learned == oracle and the score orders the three actions distinctly
    # -> Kendall tau = +1 at the one state -> mean +1.
    oracle = lambda s, a: (a[0], 0.0)
    score = lambda st: st[0]
    actions = [(0.0,), (1.0,), (2.0,)]
    assert mean_action_ranking_agreement(oracle, oracle, [(0.0, 0.0)], actions, score) == pytest.approx(1.0)


def test_action_ranking_reversed():
    oracle = lambda s, a: (a[0], 0.0)
    learned = lambda s, a: (-a[0], 0.0)  # exact rank reversal of the score
    score = lambda st: st[0]
    actions = [(0.0,), (1.0,), (2.0,)]
    assert mean_action_ranking_agreement(oracle, learned, [(0.0, 0.0)], actions, score) == pytest.approx(-1.0)


def test_action_ranking_none_when_all_degenerate():
    # Oracle maps every action to the same successor -> constant score vector
    # -> tau undefined at every state -> None (JSON null), never NaN.
    oracle = lambda s, a: (0.0, 0.0)
    learned = lambda s, a: (a[0], 0.0)
    score = lambda st: st[0]
    actions = [(0.0,), (1.0,), (2.0,)]
    assert mean_action_ranking_agreement(oracle, learned, [(0.0, 0.0)], actions, score) is None


def test_action_ranking_requires_two_actions():
    with pytest.raises(ValueError):
        mean_action_ranking_agreement(lambda s, a: s, lambda s, a: s, [(0.0,)], [(0.0,)], lambda st: st[0])
