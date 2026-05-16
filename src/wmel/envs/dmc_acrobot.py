"""DeepMind Control Suite Acrobot-swingup, wrapped to fit the framework's
`BenchmarkEnvironment` contract.

This is the first non-toy environment shipped with the package. The point is
to move evaluation onto something with real continuous-control dynamics,
where prediction-vs-planning trade-offs are measurable and where results are
comparable to the published RL literature (Hafner et al. 2023 / Dreamer-V3,
Hansen et al. 2024 / TD-MPC2, etc.).

Design choices (deliberate, documented limitations):

1. **Action discretisation**. The DMC contract takes a continuous 1-D torque
   in [-1, 1]; the wmel planner contract takes a finite, hashable
   `action_space`. We discretise to a fixed 5-level set
   (-1, -0.5, 0, 0.5, 1) by default and pass it through as `tuple[float]`
   actions. This is the same kind of compromise TD-MPC and Dreamer make at
   evaluation time when they bin a continuous head into a categorical
   policy.

2. **Success criterion**. Acrobot-swingup returns a scalar reward in [0, 1]
   where ~1 corresponds to the tip held upright (see dm_control/suite/
   acrobot.py). We define `is_success() = (last_reward >= upright_threshold)`
   with `upright_threshold = 0.6` by default. This is **strict**: the agent
   needs to actively hold the swing-up, not just touch the top.

3. **Observation flattening**. DMC returns a `dict` of arrays (orientations,
   velocity). We flatten to a `tuple[float, ...]` so it is hashable and can
   round-trip through JSON.

4. **No rendering**. `dm-control` will warn about `DISPLAY` being missing on
   headless boxes; physics still runs. We do not call any rendering API.

This module imports `dm_control` lazily; importing `wmel.envs.dmc_acrobot`
without the `control` extras installed raises a clear ImportError.
"""

from __future__ import annotations

from typing import Tuple

try:
    import numpy as np
    from dm_control import suite
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wmel.envs.dmc_acrobot requires the `control` extras. "
        "Install with `pip install -e \".[control]\"`."
    ) from exc

from wmel.adapters.base import BenchmarkEnvironment


Action = Tuple[float, ...]
Observation = Tuple[float, ...]


DEFAULT_DISCRETE_LEVELS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
DEFAULT_UPRIGHT_THRESHOLD = 0.6
DEFAULT_EPISODE_BUDGET = 1000


def _flatten_observation(obs_dict) -> Observation:
    """DMC observation `dict` -> flat `tuple[float, ...]`, deterministic order.

    Keys are sorted so the layout of the flattened vector is stable across
    episodes and across Python versions.
    """
    parts: list[float] = []
    for key in sorted(obs_dict.keys()):
        arr = np.asarray(obs_dict[key]).flatten()
        parts.extend(float(x) for x in arr)
    return tuple(parts)


class DMCAcrobotEnv(BenchmarkEnvironment):
    """DMC Acrobot-swingup behind the `BenchmarkEnvironment` interface.

    Parameters
    ----------
    discrete_levels
        Tuple of torques to expose as the discrete `action_space`. Each
        action in the contract is `(level,)` (a 1-tuple of a float).
        Defaults to `(-1.0, -0.5, 0.0, 0.5, 1.0)`.
    upright_threshold
        Reward threshold above which `is_success()` returns True. Defaults
        to 0.6.
    task_kwargs
        Forwarded to `dm_control.suite.load` (e.g., `{"random": 0}` for a
        deterministic episode init). Defaults to `{"random": 0}`.
    """

    def __init__(
        self,
        discrete_levels: tuple[float, ...] = DEFAULT_DISCRETE_LEVELS,
        upright_threshold: float = DEFAULT_UPRIGHT_THRESHOLD,
        task_kwargs: dict | None = None,
    ) -> None:
        if not discrete_levels:
            raise ValueError("discrete_levels must contain at least one torque")
        self._levels = discrete_levels
        self._actions: tuple[Action, ...] = tuple((float(t),) for t in discrete_levels)
        self._upright_threshold = float(upright_threshold)
        self._task_kwargs = task_kwargs if task_kwargs is not None else {"random": 0}
        self._env = suite.load(
            domain_name="acrobot",
            task_name="swingup",
            task_kwargs=self._task_kwargs,
        )
        # Always-fixed "goal" representation for the contract. The contract
        # callers do not use the value (Acrobot has no spatial goal); we
        # provide a stable placeholder.
        self._goal: Observation = tuple()
        self._last_obs: Observation = tuple()
        self._last_reward: float = 0.0

    def reset(self) -> Observation:
        ts = self._env.reset()
        self._last_obs = _flatten_observation(ts.observation)
        self._last_reward = 0.0
        return self._last_obs

    def step(self, action: Action) -> Observation:
        if not (isinstance(action, tuple) and len(action) == 1):
            raise ValueError(
                f"DMCAcrobotEnv expects a 1-tuple action like (0.5,); got {action!r}"
            )
        torque = np.asarray(action, dtype=np.float32)
        ts = self._env.step(torque)
        self._last_obs = _flatten_observation(ts.observation)
        self._last_reward = float(ts.reward) if ts.reward is not None else 0.0
        return self._last_obs

    def is_success(self) -> bool:
        return self._last_reward >= self._upright_threshold

    def perturb(self) -> None:
        """No-op by default. Real perturbations on a continuous-control env
        belong in the perturbation library and are out of scope for v0.8.
        """
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
        """The dense DMC reward of the most recent step. Useful for richer
        scorecard fields once perturbation-aware metrics arrive in v0.9."""
        return self._last_reward
