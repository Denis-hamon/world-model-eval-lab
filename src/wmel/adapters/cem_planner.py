"""Cross-Entropy Method (CEM) planner over a user-supplied dynamics callable.

Companion to `TabularWorldModelPlanner` (random-shooting MPC). Same contract,
same `dynamics: (state, action) -> next_state` signature, same scoring
function, same `LeWMAdapterStub` plumbing - only the search strategy
differs. Use both planners on the same env / dynamics pair to separate
planner-capacity contributions from learned-dynamics-quality contributions:
if CEM lifts the oracle arm's success rate, the random-shooting MPC was
not exploiting the perfect-information setting fully; if CEM lifts the
learned arm's success rate but not the oracle's, the learned dynamics was
fine and the random-shoot was leaving it on the table.

Algorithm
---------
Maintain a per-timestep categorical distribution `probs[h, a]` over the
discrete `action_space`. Initial distribution: uniform. Each iteration:

  1. Sample `num_samples` candidate sequences from the current distribution.
  2. Score each: take the earliest-best step along the rollout (same
     semantics as `TabularWorldModelPlanner`) and use that score.
  3. Pick the `num_elites` lowest-scoring (best) candidates.
  4. Refit the categorical distribution from the elites' per-timestep
     action histograms; mix with the uniform prior at rate `smoothing`
     so the distribution does not collapse to a delta before convergence.

The returned action sequence is the best-scoring candidate across all
iterations, truncated at its earliest-best step (matching the random-shoot
planner's "plan toward arrival rather than overshoot" behaviour).

Compute accounting
------------------
`compute_per_plan_call = num_iterations * num_samples * plan_horizon`,
exposed via the `LeWMAdapterStub` attribute so scorecards report compute
on a comparable basis to random-shoot.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

from wmel.adapters.base import Action, Observation
from wmel.adapters.lewm_adapter_stub import LeWMAdapterStub, Latent

Dynamics = Callable[[Observation, Action], Observation]
Score = Callable[[Observation, Observation], float]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


class CEMPlanner(LeWMAdapterStub):
    """Cross-Entropy Method MPC over a discrete `action_space`.

    Parameters mirror `TabularWorldModelPlanner` where possible; CEM-specific
    knobs are `num_iterations`, `num_samples`, `num_elites`, and `smoothing`.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        action_space: Sequence[Action],
        num_iterations: int = 3,
        num_samples: int = 24,
        num_elites: int = 6,
        plan_horizon: int = 15,
        smoothing: float = 0.1,
        score: Score | None = None,
        seed: int | None = None,
    ) -> None:
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not (0 < num_elites <= num_samples):
            raise ValueError("num_elites must be in (0, num_samples]")
        if plan_horizon <= 0:
            raise ValueError("plan_horizon must be positive")
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError("smoothing must be in [0, 1]")
        if not action_space:
            raise ValueError("action_space must not be empty")
        self._dynamics = dynamics
        self._actions = list(action_space)
        self._action_to_idx = {a: i for i, a in enumerate(self._actions)}
        self._iterations = num_iterations
        self._num_samples = num_samples
        self._num_elites = num_elites
        self._plan_horizon = plan_horizon
        self._smoothing = smoothing
        self._score: Score = score if score is not None else _manhattan
        self._rng = random.Random(seed)
        self.compute_per_plan_call = float(num_iterations * num_samples * plan_horizon)

    @property
    def name(self) -> str:
        return "cem"

    def encode(self, observation: Observation) -> Latent:
        return observation

    def rollout(self, latent: Latent, actions: Sequence[Action]) -> list[Latent]:
        trajectory: list[Latent] = [latent]
        state = latent
        for action in actions:
            state = self._dynamics(state, action)
            trajectory.append(state)
        return trajectory

    def score(self, latent: Latent, goal_latent: Latent) -> float:
        return self._score(latent, goal_latent)

    def plan(
        self,
        observation: Observation,
        goal: Observation,
        horizon: int,
    ) -> list[Action]:
        if horizon <= 0:
            return []

        z0 = self.encode(observation)
        gz = self.encode(goal)
        H = min(horizon, self._plan_horizon)
        n_act = len(self._actions)

        # Uniform initial distribution over actions, per timestep.
        uniform_row = [1.0 / n_act] * n_act
        probs: list[list[float]] = [list(uniform_row) for _ in range(H)]

        best_sequence: list[Action] = []
        best_score = float("inf")

        for _ in range(self._iterations):
            scored: list[tuple[float, int, list[Action]]] = []
            for _ in range(self._num_samples):
                candidate = [
                    self._actions[self._rng.choices(range(n_act), weights=probs[h], k=1)[0]]
                    for h in range(H)
                ]
                trajectory = self.rollout(z0, candidate)
                cand_score = self.score(trajectory[-1], gz)
                cand_idx = len(candidate)
                for idx, state in enumerate(trajectory[1:], start=1):
                    s = self.score(state, gz)
                    if s < cand_score:
                        cand_score = s
                        cand_idx = idx
                scored.append((cand_score, cand_idx, candidate))

            scored.sort(key=lambda item: item[0])
            if scored[0][0] < best_score:
                best_score = scored[0][0]
                best_sequence = scored[0][2][: scored[0][1]]

            elites = scored[: self._num_elites]
            # Refit categorical distribution per timestep from elite histograms,
            # mixed with the uniform prior to avoid premature collapse.
            n_elite = max(1, len(elites))
            new_probs: list[list[float]] = []
            for h in range(H):
                counts = [0.0] * n_act
                for _s, _i, cand in elites:
                    counts[self._action_to_idx[cand[h]]] += 1.0
                empirical = [c / n_elite for c in counts]
                mixed = [
                    (1.0 - self._smoothing) * empirical[a] + self._smoothing * uniform_row[a]
                    for a in range(n_act)
                ]
                new_probs.append(mixed)
            probs = new_probs

        return best_sequence
