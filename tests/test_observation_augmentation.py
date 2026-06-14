"""Closed-form tests for the observation-augmentation algebra (reframed T2.3).

All pure and stdlib-only: a synthetic 2-D env with a known dynamics doubles as
its own oracle, so the augmented oracle must reproduce the augmented env step
EXACTLY (the invariant the whole stress test rests on). No torch, no
dm_control, no checkpoints.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.adapters.base import BenchmarkEnvironment  # noqa: E402
from experiments.obs_robustness.observation_augmentation import (  # noqa: E402
    HighFrequencyFeatures,
    ObsAugmentedEnv,
    RedundantFeatures,
    join,
    make_augmented_oracle,
    make_augmented_score,
    one_step_mse_split,
    split,
)


# A deterministic toy env whose dynamics IS its oracle, so exact reproduction
# can be asserted. State is 2-D; the action is added componentwise.
def _toy_dynamics(state, action):
    return (state[0] + action[0], state[1] + action[1])


class ToyEnv(BenchmarkEnvironment):
    def __init__(self):
        self._s = (0.0, 0.0)

    def reset(self):
        self._s = (1.0, -1.0)
        return self._s

    def step(self, action):
        self._s = _toy_dynamics(self._s, action)
        return self._s

    def is_success(self):
        return self._s[0] >= 5.0

    def perturb(self):
        return None

    @property
    def observation(self):
        return self._s

    @property
    def goal(self):
        return ()

    @property
    def action_space(self):
        return ((1.0, 0.0), (0.0, 1.0))


BASE_DIM = 2


def test_split_join_roundtrip():
    aug = join((1.0, 2.0), (3.0, 4.0, 5.0))
    state, nuis = split(aug, BASE_DIM)
    assert state == (1.0, 2.0)
    assert nuis == (3.0, 4.0, 5.0)
    assert join(state, nuis) == aug


def test_split_rejects_short_obs():
    with pytest.raises(ValueError):
        split((1.0,), BASE_DIM)


@pytest.mark.parametrize("spec_factory", [
    lambda: RedundantFeatures(width=3, base_dim=BASE_DIM),
    lambda: HighFrequencyFeatures(width=3, base_dim=BASE_DIM),
])
def test_augmented_oracle_reproduces_augmented_env_exactly(spec_factory):
    # The toy env's dynamics equals its oracle, so the augmented oracle must
    # match the augmented env step bit-for-bit over a multi-step rollout.
    spec = spec_factory()
    env = ObsAugmentedEnv(ToyEnv(), spec)
    oracle = make_augmented_oracle(_toy_dynamics, BASE_DIM, spec)

    obs = env.reset()
    assert len(obs) == BASE_DIM + spec.width
    for a in [(1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]:
        predicted = oracle(obs, a)
        actual = env.step(a)
        assert predicted == pytest.approx(actual, abs=0.0)
        obs = actual


def test_augmented_score_ignores_nuisance():
    # Score reads only the state slice: changing the nuisance must not move it.
    base_score = lambda s: s[0] * 10.0
    score = make_augmented_score(base_score, BASE_DIM)
    a = join((2.0, 9.0), (0.1, 0.2, 0.3))
    b = join((2.0, 9.0), (-7.0, 100.0, 0.0))
    assert score(a) == pytest.approx(20.0)
    assert score(a) == score(b)


def test_redundant_initial_is_state_features_and_step_uses_next_state():
    spec = RedundantFeatures(width=2, base_dim=BASE_DIM)
    import math
    s0 = (0.5, -0.5)
    assert spec.initial(s0) == pytest.approx((math.tanh(0.5), math.tanh(-0.5)))
    # step depends only on next_state, not on current state/nuisance
    ns = (1.0, 2.0)
    assert spec.step(s0, (9.0, 9.0), ns) == pytest.approx((math.tanh(1.0), math.tanh(2.0)))


def test_high_freq_is_state_feature_and_step_uses_next_state():
    import math
    spec = HighFrequencyFeatures(width=2, base_dim=BASE_DIM, base_freq=12.0)
    s0 = (0.5, -0.5)
    # feature j = sin(base_freq * (1 + j) * state[j % base_dim])
    assert spec.initial(s0) == pytest.approx((math.sin(12.0 * 0.5), math.sin(12.0 * 2 * -0.5)))
    ns = (0.1, 0.2)
    assert spec.step(s0, (9.0, 9.0), ns) == pytest.approx(
        (math.sin(12.0 * 0.1), math.sin(12.0 * 2 * 0.2)))
    # bounded in [-1, 1]
    assert all(-1.0 <= v <= 1.0 for v in spec.initial((100.0, -100.0)))


def test_high_freq_rejects_bad_args():
    with pytest.raises(ValueError):
        HighFrequencyFeatures(width=1, base_dim=BASE_DIM, base_freq=0.0)
    with pytest.raises(ValueError):
        HighFrequencyFeatures(width=1, base_dim=0)


def test_one_step_mse_split_zero_when_identical():
    spec = HighFrequencyFeatures(width=2, base_dim=BASE_DIM)
    oracle = make_augmented_oracle(_toy_dynamics, BASE_DIM, spec)
    states = [join((0.0, 0.0), (0.3, 0.6)), join((1.0, 1.0), (0.2, 0.8))]
    actions = [(1.0, 0.0), (0.0, 1.0)]
    out = one_step_mse_split(oracle, oracle, states, actions, BASE_DIM)
    assert out["mse_total"] == pytest.approx(0.0)
    assert out["mse_state"] == pytest.approx(0.0)
    assert out["mse_nuisance"] == pytest.approx(0.0)


def test_one_step_mse_split_isolates_state_vs_nuisance():
    # learned = oracle on the state slice but adds a constant error of 2.0 to
    # every nuisance dim -> mse_state == 0, mse_nuisance == 4.0, and mse_total
    # is the dimension-weighted average of the two.
    spec = HighFrequencyFeatures(width=3, base_dim=BASE_DIM)
    oracle = make_augmented_oracle(_toy_dynamics, BASE_DIM, spec)

    def learned(aug_obs, action):
        s, n = split(oracle(aug_obs, action), BASE_DIM)
        return join(s, tuple(x + 2.0 for x in n))

    states = [join((0.0, 0.0), (0.3, 0.6, 0.5)), join((2.0, -1.0), (0.2, 0.8, 0.4))]
    actions = [(1.0, 0.0), (0.0, 1.0)]
    out = one_step_mse_split(oracle, learned, states, actions, BASE_DIM)
    assert out["mse_state"] == pytest.approx(0.0)
    assert out["mse_nuisance"] == pytest.approx(4.0)
    # 2 state dims contribute 0, 3 nuisance dims contribute 4 each, per (s,a).
    n_state = out["n_state_dims_scored"]
    n_nuis = out["n_nuisance_dims_scored"]
    assert out["mse_total"] == pytest.approx((0.0 * n_state + 4.0 * n_nuis) / (n_state + n_nuis))
