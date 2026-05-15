"""Tests for the perturbation library."""

from __future__ import annotations

import pytest

from wmel.adapters.base import BenchmarkEnvironment
from wmel.perturbations import (
    CompositePerturbation,
    DropNextActions,
    EnvPerturbation,
    Perturbation,
)


class _CountingEnv(BenchmarkEnvironment):
    """Minimal env that counts how many times `perturb()` was called."""

    def __init__(self) -> None:
        self.perturb_calls = 0

    def reset(self): return 0
    def step(self, action): return 0
    def is_success(self): return False
    def perturb(self): self.perturb_calls += 1

    @property
    def observation(self): return 0
    @property
    def goal(self): return 1
    @property
    def action_space(self): return ("up",)


def test_env_perturbation_delegates_to_env_perturb() -> None:
    env = _CountingEnv()
    perturbation = EnvPerturbation()
    perturbation.apply_to_env(env)
    assert env.perturb_calls == 1


def test_env_perturbation_does_not_transform_actions() -> None:
    perturbation = EnvPerturbation()
    actions = ["up", "down", "left"]
    assert perturbation.transform_actions(actions) == actions
    # Defensive copy: returned list is not the same object.
    assert perturbation.transform_actions(actions) is not actions


def test_drop_next_actions_drops_correct_prefix() -> None:
    perturbation = DropNextActions(k=2)
    assert perturbation.transform_actions(["a", "b", "c", "d"]) == ["c", "d"]
    assert perturbation.name == "drop-next-2"


def test_drop_next_actions_empty_when_k_exceeds_queue() -> None:
    perturbation = DropNextActions(k=5)
    assert perturbation.transform_actions(["a", "b"]) == []


def test_drop_next_actions_does_not_touch_env() -> None:
    env = _CountingEnv()
    DropNextActions(k=3).apply_to_env(env)
    assert env.perturb_calls == 0


def test_drop_next_actions_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        DropNextActions(k=0)
    with pytest.raises(ValueError):
        DropNextActions(k=-1)


def test_composite_perturbation_applies_in_order() -> None:
    env = _CountingEnv()
    calls: list[str] = []

    class _Recorder(Perturbation):
        def __init__(self, tag: str) -> None:
            self._tag = tag

        @property
        def name(self): return self._tag

        def apply_to_env(self, env): calls.append(self._tag)

    composite = CompositePerturbation(_Recorder("a"), _Recorder("b"), _Recorder("c"))
    composite.apply_to_env(env)
    assert calls == ["a", "b", "c"]
    assert composite.name == "a+b+c"


def test_composite_perturbation_chains_action_transforms() -> None:
    composite = CompositePerturbation(DropNextActions(k=1), DropNextActions(k=2))
    # 1 + 2 = 3 actions dropped from the front.
    assert composite.transform_actions(["a", "b", "c", "d", "e"]) == ["d", "e"]


def test_composite_perturbation_chains_env_and_action() -> None:
    env = _CountingEnv()
    composite = CompositePerturbation(EnvPerturbation(), DropNextActions(k=2))
    composite.apply_to_env(env)
    assert env.perturb_calls == 1
    assert composite.transform_actions(["a", "b", "c"]) == ["c"]


def test_composite_perturbation_rejects_empty_parts() -> None:
    with pytest.raises(ValueError):
        CompositePerturbation()


def test_composite_perturbation_hook_ordering_is_env_first_then_actions() -> None:
    """Spec: all `apply_to_env` hooks fire in order, then all `transform_actions`
    hooks fire in order. Locks in the docstring's batched-by-type ordering."""
    events: list[str] = []
    env = _CountingEnv()

    class _Recorder(Perturbation):
        def __init__(self, tag: str) -> None:
            self._tag = tag

        @property
        def name(self): return self._tag

        def apply_to_env(self, env): events.append(f"env-{self._tag}")

        def transform_actions(self, remaining):
            events.append(f"act-{self._tag}")
            return list(remaining)

    composite = CompositePerturbation(_Recorder("a"), _Recorder("b"))
    composite.apply_to_env(env)
    composite.transform_actions(["x", "y"])

    assert events == ["env-a", "env-b", "act-a", "act-b"]


def test_drop_next_actions_returns_fresh_list() -> None:
    """Defensive copy: callers must not be able to mutate the perturbation's
    internal state via the returned list."""
    perturbation = DropNextActions(k=1)
    actions = ["a", "b", "c"]
    out = perturbation.transform_actions(actions)
    assert out is not actions
    out.append("z")
    assert "z" not in actions


def test_composite_perturbation_returns_fresh_list() -> None:
    composite = CompositePerturbation(DropNextActions(k=1))
    actions = ["a", "b", "c"]
    out = composite.transform_actions(actions)
    assert out is not actions
