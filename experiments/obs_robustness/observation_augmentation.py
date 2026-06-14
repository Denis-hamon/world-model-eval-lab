"""Observation-richness stress test for the CPG verdict (pure, stdlib-only).

Does the prediction != decision dissociation survive when the world model no
longer sees the clean, privileged low-dimensional state? This module is the
non-pixel way to ask that question. (A from-pixels task would introduce a
pixel/field-of-view evaluation axis, which is the signature of the
DINO-WM / stable-worldmodel agenda; the lab keeps a deliberate non-affiliation
guardrail against that axis -- see experiments/GPU_ROADMAP.md and this
directory's README. Nuisance-augmented state delivers the same scientific
payload without the axis.)

The construction wraps any ``BenchmarkEnvironment`` so the observation becomes
``true_state ++ nuisance``, where the nuisance dimensions are task-irrelevant
but deterministic, hence EXACTLY reproducible by the oracle. Crucially:

  - the oracle dynamics still steps the true physics (the simulator remains the
    oracle), so an augmented-oracle arm reproduces the augmented env step to the
    same precision the base oracle reproduces the base env;
  - the planner's score reads only the true-state slice, so DECISIONS depend on
    real physics alone, not on the nuisance;
  - the learned world model is trained on the augmented observation, so its
    one-step prediction error (the naive M1 foil) is moved up or down by the
    nuisance design -- while the closed-loop gap should not move.

Two nuisance kinds, both deterministic functions of the state (so they are
reconstructable from the augmented observation alone -- a hard requirement,
since the oracle is handed only ``aug_obs`` -- and contribute nothing to
decisions). They differ only in *learnability*, which is the whole point:

  redundant : smooth low-frequency features of the state. A model that predicts
              the state well predicts these well too, so they pad the one-step
              error with easy targets -- expected to DEFLATE the mean one-step
              error (measured per cell; see the README's honesty note).
  high_freq : high-frequency features of the state, ``sin(K * state)`` with K
              large. Deterministic and bounded, but a finite smooth MLP cannot
              resolve the high frequency, so the one-step prediction error stays
              large -- expected to INFLATE the mean one-step error.

Note this is genuinely a *one-step* difficulty: unlike a chaotic temporal map
(whose one-step update is a trivially fittable parabola, and whose divergence
only shows up over multi-step rollouts), ``sin(K * state)`` is hard to predict
from the state in a single step, which is exactly the quantity the keystone's
M1 foil measures.

Everything here is stdlib-only and pure, so the algebra is unit-tested with a
synthetic env and synthetic dynamics; the torch/dm_control experiment that uses
it (``cartpole_obs_robustness.py``) lives beside it.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from wmel.adapters.base import Action, BenchmarkEnvironment, Observation

# A dynamics callable maps (state, action) -> next_state; a score maps
# state -> float (lower is better, the wmel convention).
Dynamics = Callable[[Observation, Action], Observation]
Score = Callable[..., float]


# --------------------------------------------------------------------------- #
# Split / join algebra.                                                        #
# --------------------------------------------------------------------------- #

def split(aug_obs: Observation, base_dim: int) -> tuple[tuple, tuple]:
    """Return ``(state, nuisance)`` from an augmented observation."""
    if len(aug_obs) < base_dim:
        raise ValueError(f"augmented obs of length {len(aug_obs)} shorter than base_dim {base_dim}")
    return tuple(aug_obs[:base_dim]), tuple(aug_obs[base_dim:])


def join(state: Sequence[float], nuisance: Sequence[float]) -> tuple:
    """Concatenate a true-state slice and a nuisance slice into one obs tuple."""
    return tuple(float(x) for x in state) + tuple(float(x) for x in nuisance)


# --------------------------------------------------------------------------- #
# Nuisance specifications. Each is reconstructable from the augmented obs:      #
# `initial(state)` seeds the nuisance at reset; `step(state, nuisance,          #
# next_state)` advances it, depending only on quantities the oracle has.        #
# --------------------------------------------------------------------------- #

class RedundantFeatures:
    """Nuisance = smooth, bounded, low-frequency features of the CURRENT state,
    so the next nuisance is a pure function of the next state. Easy to predict
    -- a model good at the state is good at these -- so they are expected to
    deflate the mean one-step error.

    Feature j is ``tanh(state[j mod base_dim] * (1 + (j // base_dim)))``: a
    smooth, bounded, state-determined value that repeats the state slice with
    growing gain so wider nuisance is not trivially identical columns.
    """

    kind = "redundant"

    def __init__(self, width: int, base_dim: int):
        if width < 0 or base_dim <= 0:
            raise ValueError("width must be >= 0 and base_dim > 0")
        self.width = width
        self.base_dim = base_dim

    def _features(self, state: Sequence[float]) -> tuple:
        return tuple(
            math.tanh(state[j % self.base_dim] * (1.0 + (j // self.base_dim)))
            for j in range(self.width)
        )

    def initial(self, state: Sequence[float]) -> tuple:
        return self._features(state)

    def step(self, state: Sequence[float], nuisance: Sequence[float],
             next_state: Sequence[float]) -> tuple:
        return self._features(next_state)


class HighFrequencyFeatures:
    """Nuisance = high-frequency features of the CURRENT state, so the next
    nuisance is a pure function of the next state (reconstructable by the
    oracle). Feature j is ``sin(base_freq * (1 + j) * state[j mod base_dim])``:
    bounded in [-1, 1] and deterministic, but the frequency grows with j, and a
    finite smooth MLP cannot resolve a high-frequency sinusoid of its input, so
    the one-step prediction error stays large -- expected to inflate the mean
    one-step error while contributing nothing to decisions.

    This is a genuine ONE-STEP difficulty (the quantity the keystone's M1 foil
    measures), not a multi-step divergence: there is no temporal recurrence to
    diverge, only a hard-to-fit instantaneous map from state to nuisance.
    """

    kind = "high_freq"

    def __init__(self, width: int, base_dim: int, base_freq: float = 12.0):
        if width < 0 or base_dim <= 0:
            raise ValueError("width must be >= 0 and base_dim > 0")
        if base_freq <= 0:
            raise ValueError("base_freq must be positive")
        self.width = width
        self.base_dim = base_dim
        self.base_freq = float(base_freq)

    def _features(self, state: Sequence[float]) -> tuple:
        return tuple(
            math.sin(self.base_freq * (1 + j) * state[j % self.base_dim])
            for j in range(self.width)
        )

    def initial(self, state: Sequence[float]) -> tuple:
        return self._features(state)

    def step(self, state: Sequence[float], nuisance: Sequence[float],
             next_state: Sequence[float]) -> tuple:
        return self._features(next_state)


# --------------------------------------------------------------------------- #
# Augmented env wrapper + matched oracle / score.                              #
# --------------------------------------------------------------------------- #

class ObsAugmentedEnv(BenchmarkEnvironment):
    """Wrap a base env so observations are ``true_state ++ nuisance``. Success
    and goal delegate to the base env (they read the true physics); the action
    space is unchanged."""

    def __init__(self, base_env: BenchmarkEnvironment, spec):
        self._base = base_env
        self._spec = spec
        self._state: tuple = tuple()
        self._nuisance: tuple = tuple()

    def reset(self) -> Observation:
        self._state = tuple(self._base.reset())
        self._nuisance = tuple(self._spec.initial(self._state))
        return join(self._state, self._nuisance)

    def step(self, action: Action) -> Observation:
        next_state = tuple(self._base.step(action))
        next_nuisance = tuple(self._spec.step(self._state, self._nuisance, next_state))
        self._state, self._nuisance = next_state, next_nuisance
        return join(self._state, self._nuisance)

    def is_success(self) -> bool:
        return self._base.is_success()

    def perturb(self) -> None:
        self._base.perturb()

    @property
    def observation(self) -> Observation:
        return join(self._state, self._nuisance)

    @property
    def goal(self) -> Observation:
        return self._base.goal

    @property
    def action_space(self) -> Sequence[Action]:
        return self._base.action_space


def make_augmented_oracle(base_oracle: Dynamics, base_dim: int, spec) -> Dynamics:
    """Lift a base oracle into augmented-obs space. Given ``(aug_obs, action)``
    it steps the true state with ``base_oracle`` and advances the nuisance with
    ``spec.step`` -- exactly the recurrence ``ObsAugmentedEnv.step`` uses, so the
    augmented oracle reproduces the augmented env wherever the base oracle
    reproduces the base env."""

    def _dynamics(aug_obs: Observation, action: Action) -> Observation:
        state, nuisance = split(aug_obs, base_dim)
        next_state = tuple(base_oracle(state, action))
        next_nuisance = tuple(spec.step(state, nuisance, next_state))
        return join(next_state, next_nuisance)

    return _dynamics


def make_augmented_score(base_score: Score, base_dim: int) -> Score:
    """Lift a base score into augmented-obs space: read only the true-state
    slice, so the planner's objective ignores the nuisance entirely."""

    def _score(aug_obs: Observation, goal: Observation = ()) -> float:
        state, _ = split(aug_obs, base_dim)
        return base_score(state)

    return _score


# --------------------------------------------------------------------------- #
# One-step error, split into decision-relevant (state) and nuisance parts.     #
# --------------------------------------------------------------------------- #

def one_step_mse_split(oracle_dyn: Dynamics, learned_dyn: Dynamics,
                       aug_states: Sequence[Observation], actions: Sequence[Action],
                       base_dim: int) -> dict:
    """Mean squared one-step error of ``learned_dyn`` vs ``oracle_dyn`` over the
    ``aug_states`` x ``actions`` grid, reported as total (all dims), state
    (decision-relevant slice) and nuisance (the rest). MSE is per-dimension
    (summed squared error divided by the dimension count) so the three figures
    are on the same scale regardless of how many nuisance dims there are."""
    if not aug_states or not actions:
        raise ValueError("need at least one state and one action")
    sse_state = sse_nuis = 0.0
    n_state = n_nuis = 0
    for s in aug_states:
        for a in actions:
            o, l = oracle_dyn(s, a), learned_dyn(s, a)
            os, on = split(o, base_dim)
            ls, ln = split(l, base_dim)
            for x, y in zip(ls, os):
                sse_state += (float(x) - float(y)) ** 2
                n_state += 1
            for x, y in zip(ln, on):
                sse_nuis += (float(x) - float(y)) ** 2
                n_nuis += 1
    mse_state = sse_state / n_state if n_state else 0.0
    mse_nuis = sse_nuis / n_nuis if n_nuis else 0.0
    total = (sse_state + sse_nuis) / (n_state + n_nuis) if (n_state + n_nuis) else 0.0
    return {"mse_total": total, "mse_state": mse_state, "mse_nuisance": mse_nuis,
            "n_state_dims_scored": n_state, "n_nuisance_dims_scored": n_nuis}
