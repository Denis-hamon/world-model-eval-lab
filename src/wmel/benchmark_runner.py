"""A small, deterministic benchmark runner.

Given an environment factory and a planner policy, runs N episodes and returns
structured results that can be fed into `wmel.metrics.compute_scorecard`.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from wmel.adapters.base import BenchmarkEnvironment, PlannerPolicy
from wmel.metrics import EpisodeResult
from wmel.perturbations import EnvPerturbation, Perturbation


EnvFactory = Callable[[], BenchmarkEnvironment]


@dataclass
class BenchmarkRunner:
    """Run a policy against an environment for a fixed number of episodes.

    Parameters
    ----------
    env_factory
        Callable returning a fresh environment instance. The runner calls it
        once per episode to keep episodes independent.
    policy
        The `PlannerPolicy` under test.
    episodes
        Number of episodes to run.
    horizon
        Maximum number of actions per episode.
    perturb_prob
        Probability that the runner will *schedule* a perturbation for an
        episode. The perturbation actually fires at most once, at a uniformly
        chosen step in `[1, horizon // 2]`. If the episode ends earlier (e.g.,
        the policy succeeds before that step), the perturbation does not fire
        and the episode is reported as `perturbed=False`. This keeps the
        `perturbation_recovery_rate` denominator honest at the cost of a
        smaller effective sample when policies are very fast.
    perturbation
        Optional `Perturbation` strategy. When `None`, defaults to
        `EnvPerturbation()` (delegates to `env.perturb()` for backward
        compatibility). Pass a custom subclass to inject action-level or
        composite perturbations from `wmel.perturbations`.
    seed
        Seed for the runner's RNG (controls perturbation triggering, not the
        environment or the policy - those manage their own determinism).
    """

    env_factory: EnvFactory
    policy: PlannerPolicy
    episodes: int = 50
    horizon: int = 50
    perturb_prob: float = 0.0
    perturbation: Perturbation | None = None
    seed: int | None = None

    def run(self) -> list[EpisodeResult]:
        rng = random.Random(self.seed)
        perturbation = self.perturbation if self.perturbation is not None else EnvPerturbation()
        results: list[EpisodeResult] = []

        for _ in range(self.episodes):
            env = self.env_factory()
            env.reset()

            perturb_intended = rng.random() < self.perturb_prob
            perturb_at_step = (
                rng.randint(1, max(1, self.horizon // 2))
                if perturb_intended
                else -1
            )

            latencies_ms: list[float] = []
            steps_taken = 0
            success = False
            perturb_applied = False

            while steps_taken < self.horizon:
                if env.is_success():
                    success = True
                    break

                t0 = time.perf_counter_ns()
                planned = self.policy.plan(env.observation, env.goal, self.horizon - steps_taken)
                latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)

                if not planned:
                    break

                remaining: deque = deque(planned)
                while remaining:
                    if steps_taken == perturb_at_step and not perturb_applied:
                        perturbation.apply_to_env(env)
                        remaining = deque(perturbation.transform_actions(list(remaining)))
                        perturb_applied = True
                        if not remaining:
                            break

                    action = remaining.popleft()
                    env.step(action)
                    steps_taken += 1
                    if env.is_success():
                        success = True
                        break
                    if steps_taken >= self.horizon:
                        break

                if success:
                    break

            results.append(
                EpisodeResult(
                    success=success,
                    steps=steps_taken,
                    planning_latencies_ms=tuple(latencies_ms),
                    perturbed=perturb_applied,
                    recovered=perturb_applied and success,
                )
            )

        return results
