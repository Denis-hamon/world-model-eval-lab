"""Documented stub describing the contract a world-model-based planner would satisfy.

This module does **not** reimplement LeWorldModel, JEPA, or any specific
research artifact. It exists solely to make the evaluation contract explicit
so that a real world model - research, in-house, or otherwise - can be plugged
into the benchmark runner without changing the rest of the codebase.

The expected contract is:

    1. `encode(observation)`      -> latent state
    2. `rollout(latent, actions)` -> trajectory of future latent states
    3. `score(latent, goal)`      -> scalar distance to goal in latent space
    4. `plan(observation, goal, horizon)` -> selected action sequence

A real adapter would typically:

    - sample or enumerate candidate action sequences,
    - imagine each candidate via `rollout`,
    - score each terminal latent against the encoded goal,
    - return the best-scoring sequence.

Nothing here imports a third-party model. Subclass this stub to wire in one.
"""

from __future__ import annotations

from typing import Sequence

from wmel.adapters.base import Action, Observation, PlannerPolicy

Latent = object


class LeWMAdapterStub(PlannerPolicy):
    """Abstract adapter outlining the world-model-based planner contract.

    This class intentionally raises `NotImplementedError` for every method that
    would touch a real model. Subclasses are expected to provide:

    - `encode`        - observation to latent state.
    - `rollout`       - action-conditioned forward simulation in latent space.
    - `score`         - latent-space distance from a state to the goal latent.
    - `plan`          - the full search procedure that uses the above.

    Subclasses should NOT bundle third-party weights or training code. They
    should depend on an external library or service if a real model is needed.
    """

    @property
    def name(self) -> str:
        return "lewm-adapter-stub"

    def encode(self, observation: Observation) -> Latent:
        """Map an observation to a latent state. To be implemented by subclasses."""
        raise NotImplementedError(
            "LeWMAdapterStub.encode is a contract stub. Subclass and wire in a model."
        )

    def rollout(
        self,
        latent: Latent,
        actions: Sequence[Action],
    ) -> list[Latent]:
        """Action-conditioned forward simulation in latent space."""
        raise NotImplementedError(
            "LeWMAdapterStub.rollout is a contract stub. Subclass and wire in a model."
        )

    def score(self, latent: Latent, goal_latent: Latent) -> float:
        """Scalar distance (lower is better) from `latent` to `goal_latent`."""
        raise NotImplementedError(
            "LeWMAdapterStub.score is a contract stub. Subclass and wire in a model."
        )

    def plan(
        self,
        observation: Observation,
        goal: Observation,
        horizon: int,
    ) -> list[Action]:
        """Return the best-scoring action sequence of length up to `horizon`.

        A typical implementation:

            z0 = self.encode(observation)
            gz = self.encode(goal)
            best = None
            for candidate in self._candidate_action_sequences(horizon):
                trajectory = self.rollout(z0, candidate)
                final_score = self.score(trajectory[-1], gz)
                if best is None or final_score < best[0]:
                    best = (final_score, candidate)
            return list(best[1]) if best else []
        """
        raise NotImplementedError(
            "LeWMAdapterStub.plan is a contract stub. Subclass and implement search."
        )
