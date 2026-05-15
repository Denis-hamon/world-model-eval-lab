"""A toy concrete implementation of the `LeWMAdapterStub` contract.

This is **not** a learned model. It is a deliberately simple subclass that
fills in every method of the adapter stub with a clearly-marked toy: identity
encoding, user-provided dynamics, Manhattan distance scoring, and random-shoot
MPC for `plan`.

The point is to prove that the evaluation contract is implementable end-to-end
without depending on torch, jax, gym, or any learned model. A real adapter
would replace `encode`, `rollout`, and `score` with calls into an actual
world model; the surrounding plumbing would remain the same.
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


class TabularWorldModelPlanner(LeWMAdapterStub):
    """Random-shooting MPC over a user-supplied tabular dynamics function.

    Parameters
    ----------
    dynamics
        Callable `(state, action) -> next_state`. Represents what a learned
        world model would output as a one-step prediction. For the toy
        environments shipped here, this is just the env's transition logic
        wrapped as a function. In a real adapter, it would be a learned model.
    action_space
        The set of actions to sample from.
    num_candidates
        Number of candidate action sequences to sample per `plan` call.
    plan_horizon
        Maximum length of a sampled candidate sequence. Capped by the runtime
        `horizon` argument passed to `plan`.
    score
        Optional override for the latent-space distance to goal. Defaults to
        Manhattan distance, which assumes 2D integer-tuple observations.
    seed
        Seed for the candidate sampler.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        action_space: Sequence[Action],
        num_candidates: int = 200,
        plan_horizon: int = 20,
        score: Score | None = None,
        seed: int | None = None,
    ) -> None:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if plan_horizon <= 0:
            raise ValueError("plan_horizon must be positive")
        if not action_space:
            raise ValueError("action_space must not be empty")
        self._dynamics = dynamics
        self._actions = list(action_space)
        self._num_candidates = num_candidates
        self._plan_horizon = plan_horizon
        self._score: Score = score if score is not None else _manhattan
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "tabular-world-model"

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
        candidate_length = min(horizon, self._plan_horizon)

        best_sequence: list[Action] = []
        best_score = float("inf")

        for _ in range(self._num_candidates):
            candidate = [self._rng.choice(self._actions) for _ in range(candidate_length)]
            trajectory = self.rollout(z0, candidate)

            # Prefer the earliest moment the trajectory gets closest to the goal,
            # so we plan toward arrival rather than overshoot.
            best_step_score = self.score(trajectory[-1], gz)
            best_step_idx = len(candidate)
            for idx, state in enumerate(trajectory[1:], start=1):
                s = self.score(state, gz)
                if s < best_step_score:
                    best_step_score = s
                    best_step_idx = idx

            if best_step_score < best_score:
                best_score = best_step_score
                best_sequence = candidate[:best_step_idx]

        return best_sequence
