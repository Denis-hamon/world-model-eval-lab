"""DeepMind Control Suite Reacher-easy, wrapped to fit the framework's
`BenchmarkEnvironment` contract.

The third non-toy environment in the repository, and the first with a
**two-dimensional action** (two joint torques rather than one). The point
is to test whether the CPG verdict pattern established on Acrobot-swingup
and Cartpole-swingup -- and in particular the INCONCLUSIVE-to-MODEL-
BOTTLENECK transition that the power analysis characterises -- reproduces
on a structurally different control task (a 2-DOF planar arm reaching a
randomized target) rather than an underactuated swing-up.

Design choices mirror `wmel.envs.dmc_cartpole`, with two differences forced
by the task:

1. **Two-dimensional action**. Reacher actuates both joints, so the action
   spec is `(2,)`. The discrete action set is the Cartesian product of a
   per-dimension level set; with 3 levels per joint that is 9 actions
   (kept coarse so random-shooting / CEM search stays tractable).
2. **Oracle reconstruction is exact, not lossy.** Reacher's observation
   exposes `position = physics.position()` (= qpos) and
   `velocity = physics.velocity()` (= qvel) directly, so the oracle
   reconstructs the joint state without an atan2 round-trip. The one
   subtlety is the target: it is a per-episode randomized `geom_pos`, not
   part of qpos, so the oracle recovers it from `to_target` (which is
   `target_xy - finger_xy`) once the finger position is known from forward
   kinematics. See `make_reacher_oracle_dynamics`.

Observation flattening uses sorted keys, so the deterministic layout is
`(position[0], position[1], to_target[0], to_target[1], velocity[0],
velocity[1])`, 6 floats. The reward is the DMC `easy` sparse-ish tolerance
(1.0 inside the target radius), so `is_success` keys on a 0.5 threshold.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Tuple

try:
    import numpy as np
    from dm_control import suite
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wmel.envs.dmc_reacher requires the `control` extras. "
        "Install with `pip install -e \".[control]\"`."
    ) from exc

from wmel.adapters.base import BenchmarkEnvironment


Action = Tuple[float, ...]
Observation = Tuple[float, ...]


# 3 levels per joint -> 9 discrete actions. Coarser than the 5-level 1-D
# action sets of Acrobot/Cartpole because the action is 2-D and the planner
# searches over A^horizon.
DEFAULT_PER_DIM_LEVELS: tuple[float, ...] = (-1.0, 0.0, 1.0)
DEFAULT_SUCCESS_THRESHOLD = 0.5


def _flatten_observation(obs_dict) -> Observation:
    """DMC observation `dict` -> flat `tuple[float, ...]`, sorted-key order.

    Keys sort as `position` < `to_target` < `velocity`, giving the layout
    `(pos0, pos1, to_target0, to_target1, vel0, vel1)` for reacher.
    """
    parts: list[float] = []
    for key in sorted(obs_dict.keys()):
        arr = np.asarray(obs_dict[key]).flatten()
        parts.extend(float(x) for x in arr)
    return tuple(parts)


def _discrete_action_space(per_dim_levels: tuple[float, ...]) -> tuple[Action, ...]:
    """Cartesian product of per-dimension torque levels -> tuple of 2-tuples."""
    return tuple(product(per_dim_levels, repeat=2))


class DMCReacherEnv(BenchmarkEnvironment):
    """DMC Reacher-easy behind the `BenchmarkEnvironment` interface."""

    def __init__(
        self,
        per_dim_levels: tuple[float, ...] = DEFAULT_PER_DIM_LEVELS,
        success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
        task_kwargs: dict | None = None,
    ) -> None:
        if not per_dim_levels:
            raise ValueError("per_dim_levels must contain at least one torque")
        self._levels = per_dim_levels
        self._actions: tuple[Action, ...] = _discrete_action_space(per_dim_levels)
        self._success_threshold = float(success_threshold)
        self._task_kwargs = task_kwargs if task_kwargs is not None else {"random": 0}
        self._env = suite.load(
            domain_name="reacher",
            task_name="easy",
            task_kwargs=self._task_kwargs,
        )
        self._goal: Observation = tuple()
        self._last_obs: Observation = tuple()
        self._last_reward: float = 0.0

    def reset(self) -> Observation:
        ts = self._env.reset()
        self._last_obs = _flatten_observation(ts.observation)
        self._last_reward = 0.0
        return self._last_obs

    def step(self, action: Action) -> Observation:
        if not (isinstance(action, tuple) and len(action) == 2):
            raise ValueError(
                f"DMCReacherEnv expects a 2-tuple action like (0.5, -1.0); got {action!r}"
            )
        torque = np.asarray(action, dtype=np.float32)
        ts = self._env.step(torque)
        self._last_obs = _flatten_observation(ts.observation)
        self._last_reward = float(ts.reward) if ts.reward is not None else 0.0
        return self._last_obs

    def is_success(self) -> bool:
        return self._last_reward >= self._success_threshold

    def perturb(self) -> None:
        return None

    @property
    def observation(self) -> Observation:
        return self._last_obs

    @property
    def goal(self) -> Observation:
        return self._goal

    @property
    def action_space(self) -> tuple[Action, ...]:
        return self._actions

    @property
    def last_reward(self) -> float:
        return self._last_reward


def make_reacher_oracle_dynamics(
    per_dim_levels: tuple[float, ...] = DEFAULT_PER_DIM_LEVELS,
    task_kwargs: dict | None = None,
    reset_every: int = 800,
) -> Callable[[Observation, Action], Observation]:
    """Build a `(state, action) -> next_state` oracle dynamics callable for
    Reacher-easy, side-effect-free from the caller's perspective.

    State layout from `_flatten_observation` is
    `(pos0, pos1, to_target0, to_target1, vel0, vel1)`. Reconstruction:

        qpos = [pos0, pos1]       (exact: position() is qpos)
        qvel = [vel0, vel1]       (exact: velocity() is qvel)

    The target is a per-episode randomized `geom_pos` not present in qpos.
    `to_target = target_xy - finger_xy`, so after setting qpos and running
    forward kinematics we read the finger position and recover
    `target_xy = finger_xy + to_target`, write it into the target geom, and
    forward again. This makes the oracle reproduce `env.step` to numerical
    precision regardless of which episode's target the caller's state came
    from.
    """
    sim_env = suite.load(
        domain_name="reacher",
        task_name="easy",
        task_kwargs=task_kwargs if task_kwargs is not None else {"random": 0},
    )
    sim_env.reset()

    counter = {"calls": 0}

    def _dynamics(state: Observation, action: Action) -> Observation:
        if counter["calls"] >= reset_every:
            sim_env.reset()
            counter["calls"] = 0

        pos0, pos1, tt0, tt1, vel0, vel1 = state
        physics = sim_env.physics
        # position() is qpos, velocity() is qvel -- exact, no atan2 round-trip.
        physics.data.qpos[:] = np.array([pos0, pos1], dtype=physics.data.qpos.dtype)
        physics.data.qvel[:] = np.array([vel0, vel1], dtype=physics.data.qvel.dtype)
        physics.forward()
        # Recover the per-episode target from to_target = target_xy - finger_xy.
        finger_xy = physics.named.data.geom_xpos["finger", :2]
        physics.named.model.geom_pos["target", "x"] = float(finger_xy[0]) + tt0
        physics.named.model.geom_pos["target", "y"] = float(finger_xy[1]) + tt1
        physics.forward()

        torque = np.asarray(action, dtype=np.float32)
        ts = sim_env.step(torque)
        counter["calls"] += 1
        return _flatten_observation(ts.observation)

    return _dynamics


def reacher_reach_score(state: Observation, goal: Observation = ()) -> float:
    """Lower-is-better score for Reacher: distance from finger to target.

    The flat observation layout is `(pos0, pos1, to_target0, to_target1,
    vel0, vel1)`, and `to_target` is the finger-to-target vector, so the
    score is its Euclidean norm. The planner minimises score, so it drives
    the finger onto the target. Unlike the swing-up scores, this is the
    exact quantity the DMC reward thresholds (no approximation): the reward
    is a tolerance on this same distance.

    The `goal` argument is ignored to keep the
    `Score = Callable[[Observation, Observation], float]` contract.
    """
    if len(state) < 4:
        return 0.0
    tt0, tt1 = state[2], state[3]
    return float((tt0 * tt0 + tt1 * tt1) ** 0.5)
