"""A trivial random policy used as a sanity-check baseline."""

from __future__ import annotations

import random
from typing import Sequence

from wmel.adapters.base import Action, Observation, PlannerPolicy


class RandomPolicy(PlannerPolicy):
    """Returns a sequence of `horizon` uniformly random actions.

    Useful as a floor: any non-trivial planner should beat it on success rate
    in any reasonable benchmark.
    """

    def __init__(self, action_space: Sequence[Action], seed: int | None = None) -> None:
        if not action_space:
            raise ValueError("action_space must not be empty")
        self._actions = list(action_space)
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def plan(
        self,
        observation: Observation,
        goal: Observation,
        horizon: int,
    ) -> list[Action]:
        if horizon <= 0:
            return []
        return [self._rng.choice(self._actions) for _ in range(horizon)]
