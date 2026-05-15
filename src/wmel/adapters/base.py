"""Abstract interfaces every policy and environment must implement.

The point of this module is to define a minimal contract so that any world
model - research, proprietary, or stub - can be plugged into the benchmark
runner without changes to the rest of the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

Observation = Any
Action = Any


class PlannerPolicy(ABC):
    """A policy that produces an action sequence given an observation and a goal.

    Implementations are free to do anything internally (search, learned model
    rollouts, scripted heuristics). The runner only observes the planned
    sequence and the time it took to produce it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for reporting."""

    @abstractmethod
    def plan(
        self,
        observation: Observation,
        goal: Observation,
        horizon: int,
    ) -> list[Action]:
        """Return a list of at most `horizon` actions to attempt from `observation`."""


class BenchmarkEnvironment(ABC):
    """A small, deterministic environment that exposes the benchmark contract.

    The interface is intentionally narrow: reset, step, success check, optional
    perturbation, and read-only access to the current observation and goal.
    """

    @abstractmethod
    def reset(self) -> Observation:
        """Reset to the start state and return the initial observation."""

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Apply `action` and return the resulting observation."""

    @abstractmethod
    def is_success(self) -> bool:
        """Whether the goal has been reached in the current state."""

    @abstractmethod
    def perturb(self) -> None:
        """Apply a perturbation to the environment. May be a no-op."""

    @property
    @abstractmethod
    def observation(self) -> Observation:
        """The current observation."""

    @property
    @abstractmethod
    def goal(self) -> Observation:
        """The current goal observation."""

    @property
    @abstractmethod
    def action_space(self) -> Sequence[Action]:
        """All actions accepted by `step`. Used by adapters that need to enumerate or sample."""
