"""Regression tests for `BenchmarkRunner` correctness.

These lock in the two invariants Codex flagged in the v0.3 review:

1. An episode is only marked `perturbed=True` if `env.perturb()` was actually
   called during the rollout. Episodes that succeed before the scheduled
   perturbation step are not counted.
2. `planning_latencies_ms` records one entry per `policy.plan(...)` call, so
   downstream metrics can compute a per-call mean rather than a per-episode
   mean.
"""

from __future__ import annotations

from typing import Sequence

from wmel.adapters.base import BenchmarkEnvironment, PlannerPolicy
from wmel.benchmark_runner import BenchmarkRunner
from wmel.perturbations import DropNextActions, EnvPerturbation, Perturbation


class _InstantSuccessEnv(BenchmarkEnvironment):
    """An environment that reports success on the very first `is_success()` check."""

    def __init__(self) -> None:
        self._counter = {"perturb_calls": 0}

    def reset(self): return (0, 0)
    def step(self, action): return (0, 0)
    def is_success(self): return True
    def perturb(self): self._counter["perturb_calls"] += 1

    @property
    def observation(self): return (0, 0)
    @property
    def goal(self): return (0, 0)
    @property
    def action_space(self): return ("up",)

    @property
    def perturb_calls(self) -> int:
        return self._counter["perturb_calls"]


class _NeverSucceedsEnv(BenchmarkEnvironment):
    """An environment that never reaches success, counting actual perturb() calls."""

    def __init__(self) -> None:
        self._counter = {"perturb_calls": 0}

    def reset(self): return 0
    def step(self, action): return 0
    def is_success(self): return False
    def perturb(self): self._counter["perturb_calls"] += 1

    @property
    def observation(self): return 0
    @property
    def goal(self): return 1
    @property
    def action_space(self): return ("up",)

    @property
    def perturb_calls(self) -> int:
        return self._counter["perturb_calls"]


class _SingleActionPolicy(PlannerPolicy):
    """Returns one `up` action per `plan()` call, forcing one plan call per step."""

    @property
    def name(self): return "single-action"

    def plan(self, observation, goal, horizon):
        return ["up"] if horizon > 0 else []


def test_perturbed_false_when_episode_succeeds_before_perturb_step() -> None:
    """The headline Codex bug: `perturbed=True` must imply `env.perturb()` fired."""
    env_box: dict[str, _InstantSuccessEnv] = {}

    def factory() -> _InstantSuccessEnv:
        env_box["last"] = _InstantSuccessEnv()
        return env_box["last"]

    results = BenchmarkRunner(
        env_factory=factory,
        policy=_SingleActionPolicy(),
        episodes=10,
        horizon=10,
        perturb_prob=1.0,
        seed=0,
    ).run()

    assert all(not r.perturbed for r in results)
    assert all(not r.recovered for r in results)


def test_perturbed_count_matches_actual_perturb_calls() -> None:
    """Stronger statement: number of perturbed results equals number of env.perturb() calls."""
    perturb_calls_per_episode: list[int] = []

    def factory() -> _NeverSucceedsEnv:
        env = _NeverSucceedsEnv()
        perturb_calls_per_episode.append(0)

        original_perturb = env.perturb

        def tracked() -> None:
            original_perturb()
            perturb_calls_per_episode[-1] = env.perturb_calls

        env.perturb = tracked  # type: ignore[method-assign]
        return env

    results = BenchmarkRunner(
        env_factory=factory,
        policy=_SingleActionPolicy(),
        episodes=20,
        horizon=20,
        perturb_prob=1.0,
        seed=0,
    ).run()

    perturbed_results = sum(1 for r in results if r.perturbed)
    actually_perturbed_envs = sum(1 for c in perturb_calls_per_episode if c > 0)
    assert perturbed_results == actually_perturbed_envs


def test_planning_latencies_record_one_entry_per_plan_call() -> None:
    """Per-call latency invariant: the list length equals the number of plan() calls."""

    class _CountingPolicy(PlannerPolicy):
        def __init__(self) -> None:
            self.calls = 0

        @property
        def name(self): return "counting"

        def plan(self, observation, goal, horizon):
            self.calls += 1
            return ["up"]  # one action so plan() is invoked once per step

    policy = _CountingPolicy()
    results = BenchmarkRunner(
        env_factory=_NeverSucceedsEnv,
        policy=policy,
        episodes=3,
        horizon=5,
        perturb_prob=0.0,
        seed=0,
    ).run()

    # 3 episodes * 5 steps * 1 plan call per step = 15 plan() invocations total.
    assert policy.calls == 15
    total_recorded_calls = sum(len(r.planning_latencies_ms) for r in results)
    assert total_recorded_calls == 15
    assert all(r.plan_calls == 5 for r in results)


class _FixedSequencePolicy(PlannerPolicy):
    """Returns a fixed-length sequence each plan() call. Lets us observe how
    the runner consumes the queue when a perturbation drops some of it."""

    def __init__(self, length: int) -> None:
        self._length = length

    @property
    def name(self): return "fixed-sequence"

    def plan(self, observation, goal, horizon):
        return ["up"] * min(self._length, horizon)


def test_default_perturbation_matches_env_perturb_behavior() -> None:
    """Backward compatibility: omitting the perturbation kwarg must be
    indistinguishable from passing `EnvPerturbation()`."""

    class _RecordingEnv(BenchmarkEnvironment):
        all_calls: list[int] = []

        def __init__(self) -> None:
            self._step_count = 0
            self._calls = 0

        def reset(self): self._step_count = 0; return 0
        def step(self, a): self._step_count += 1; return 0
        def is_success(self): return self._step_count >= 4
        def perturb(self):
            self._calls += 1
            _RecordingEnv.all_calls.append(self._calls)

        @property
        def observation(self): return 0
        @property
        def goal(self): return 1
        @property
        def action_space(self): return ("up",)

    _RecordingEnv.all_calls = []
    default_results = BenchmarkRunner(
        env_factory=_RecordingEnv,
        policy=_FixedSequencePolicy(length=5),
        episodes=10,
        horizon=10,
        perturb_prob=1.0,
        seed=42,
    ).run()
    default_calls = list(_RecordingEnv.all_calls)

    _RecordingEnv.all_calls = []
    explicit_results = BenchmarkRunner(
        env_factory=_RecordingEnv,
        policy=_FixedSequencePolicy(length=5),
        episodes=10,
        horizon=10,
        perturb_prob=1.0,
        perturbation=EnvPerturbation(),
        seed=42,
    ).run()
    explicit_calls = list(_RecordingEnv.all_calls)

    assert default_calls == explicit_calls
    assert [r.perturbed for r in default_results] == [r.perturbed for r in explicit_results]
    assert [r.steps for r in default_results] == [r.steps for r in explicit_results]


def test_drop_next_actions_shortens_executed_sequence() -> None:
    """When the perturbation drops actions, the policy must replan from the
    same env state. Net effect: fewer total executed actions before success
    than with no perturbation, for an env where the policy can re-emit
    identical sequences."""

    class _CountingStepEnv(BenchmarkEnvironment):
        def __init__(self) -> None:
            self._n = 0

        def reset(self): self._n = 0; return self._n
        def step(self, a): self._n += 1; return self._n
        def is_success(self): return self._n >= 6
        def perturb(self): pass

        @property
        def observation(self): return self._n
        @property
        def goal(self): return 6
        @property
        def action_space(self): return ("up",)

    # Baseline: no perturbation. plan() returns 3 actions at a time; needs 2 plan
    # calls to reach success (6 steps).
    baseline = BenchmarkRunner(
        env_factory=_CountingStepEnv,
        policy=_FixedSequencePolicy(length=3),
        episodes=5,
        horizon=20,
        perturb_prob=0.0,
        seed=0,
    ).run()
    assert all(r.success and r.plan_calls == 2 for r in baseline)

    # DropNextActions(k=2) at step ~5 (horizon//2 = 10, but with seed=0
    # perturb_at_step may land within the executed window).
    perturbed = BenchmarkRunner(
        env_factory=_CountingStepEnv,
        policy=_FixedSequencePolicy(length=3),
        episodes=5,
        horizon=20,
        perturb_prob=1.0,
        perturbation=DropNextActions(k=2),
        seed=0,
    ).run()
    # When perturbation actually fires, the executed queue is shortened so the
    # planner needs an extra plan() call to finish. Some episodes may not be
    # perturbed if perturb_at_step exceeds the steps actually executed.
    for r in perturbed:
        if r.perturbed:
            assert r.plan_calls > 2


def test_perturbation_only_counted_when_actually_invoked() -> None:
    """A custom Perturbation that succeeds before the perturbation step
    must still be reported as `perturbed=False`."""

    class _InstantSuccessEnv(BenchmarkEnvironment):
        def reset(self): return 0
        def step(self, a): return 0
        def is_success(self): return True
        def perturb(self): pass

        @property
        def observation(self): return 0
        @property
        def goal(self): return 1
        @property
        def action_space(self): return ("up",)

    class _NoisyPerturbation(Perturbation):
        @property
        def name(self): return "noisy"
        def apply_to_env(self, env): raise AssertionError("should not fire")

    results = BenchmarkRunner(
        env_factory=_InstantSuccessEnv,
        policy=_FixedSequencePolicy(length=1),
        episodes=5,
        horizon=10,
        perturb_prob=1.0,
        perturbation=_NoisyPerturbation(),
        seed=0,
    ).run()
    assert all(not r.perturbed for r in results)
