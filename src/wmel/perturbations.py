"""Pluggable perturbation library.

A `Perturbation` is a thing the benchmark runner triggers at most once per
episode. Each perturbation has two hooks, either or both of which it may
override:

- `apply_to_env(env)`: mutate the environment state (call `env.perturb()`,
  set sensor noise, block a cell, etc.). Default: no-op.
- `transform_actions(remaining)`: return the queue of actions still to
  execute, possibly transformed. Default: pass through unchanged.

This split keeps state-level perturbations (env-side) and action-level
perturbations (runner-side, e.g., actuator drops) cleanly separated while
fitting both into a single contract.

For honest accounting, the runner marks an episode `perturbed=True` only
when the perturbation hooks were actually invoked at the scheduled step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from wmel.adapters.base import Action, BenchmarkEnvironment


class Perturbation(ABC):
    """Abstract perturbation strategy.

    Concrete subclasses override `apply_to_env`, `transform_actions`, or
    both. The default implementations are no-ops so each subclass only
    deals with the hook it cares about. A subclass that overrides neither
    is technically permitted but will leave the environment and action
    queue untouched - useful only as a control or test fixture.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier.

        Recorded on the `Scorecard` as `perturbation_name` so two runs of
        the same policy under different perturbations stay distinguishable
        in JSON, Markdown, and printed output.
        """

    def apply_to_env(self, env: BenchmarkEnvironment) -> None:
        """Mutate environment state at the perturbation moment. Default: no-op."""
        return None

    def transform_actions(self, remaining_actions: Sequence[Action]) -> list[Action]:
        """Return the queue of pending actions, possibly transformed.

        Implementations should treat the input as immutable and return a
        fresh list.
        """
        return list(remaining_actions)


class EnvPerturbation(Perturbation):
    """Delegate to `env.perturb()`. The runner's default.

    Useful when the environment owns the definition of "what a perturbation
    is" (the two-room and maze toys do this: a one-cell rewind opposite to
    the last movement).
    """

    @property
    def name(self) -> str:
        return "env-default"

    def apply_to_env(self, env: BenchmarkEnvironment) -> None:
        env.perturb()


class DropNextActions(Perturbation):
    """Drop the next `k` queued actions, forcing the policy to replan.

    Models actuator drops, network gaps, or a debouncing layer that swallows
    a burst of commands. The next `plan()` call sees the post-perturbation
    state and must recover.

    If `k` exceeds the number of remaining actions, the queue is fully
    drained and the policy is forced to replan immediately from the current
    state.
    """

    def __init__(self, k: int = 1) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self._k = k

    @property
    def name(self) -> str:
        return f"drop-next-{self._k}"

    def transform_actions(self, remaining_actions: Sequence[Action]) -> list[Action]:
        return list(remaining_actions)[self._k:]


class CompositePerturbation(Perturbation):
    """Apply several perturbations in order at the same trigger moment.

    Example:

        CompositePerturbation(EnvPerturbation(), DropNextActions(k=3))

    Fires `env.perturb()` first, then drops the next three queued actions.
    Useful for testing combined failure modes.

    Hook ordering: all `apply_to_env` hooks fire first, in the order parts
    were passed; then all `transform_actions` hooks fire, in the same order.
    Because `transform_actions` takes only the action queue (not the env),
    this batching has no observable effect for the current API surface, but
    the spec is pinned in case the hook signatures grow later.
    """

    def __init__(self, *parts: Perturbation) -> None:
        if not parts:
            raise ValueError("CompositePerturbation needs at least one part")
        self._parts = parts

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self._parts)

    def apply_to_env(self, env: BenchmarkEnvironment) -> None:
        for part in self._parts:
            part.apply_to_env(env)

    def transform_actions(self, remaining_actions: Sequence[Action]) -> list[Action]:
        actions = list(remaining_actions)
        for part in self._parts:
            actions = part.transform_actions(actions)
        return actions
