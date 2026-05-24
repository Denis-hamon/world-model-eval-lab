"""DeepMind Control Suite Cartpole-swingup, wrapped to fit the framework's
`BenchmarkEnvironment` contract.

The second non-toy environment in the repository. The point is to test
whether the v0.12 / v0.13 verdict pattern (\\textsc{model bottleneck}
across random / TD-MPC2 / MLP-on-TD-MPC2-data arms, random-shoot
bracketing the oracle low, CEM closing that gap on the oracle but not
on either learned arm) generalises from Acrobot-swingup to a different
underactuated control problem.

Cartpole-swingup is *easier* than Acrobot: the cart can move along a rail
(actuated DoF), the pole hangs from a single revolute joint, and the
unactuated DoF is one rather than two. We pick this gap on purpose: if
the same verdict pattern reproduces here, the framework is generalising
across the easy/hard axis at fixed family; if it diverges (e.g. \\textsc{model
as good as oracle} or \\textsc{learned outperforms oracle}), we have
demonstrated a second branch of the verdict tree in vivo.

Design choices mirror `wmel.envs.dmc_acrobot`:

1. **Action discretisation**. 5-level torque set $\\{-1, -0.5, 0, 0.5, 1\\}$,
   same compromise.
2. **Success criterion**. `is_success() = (last_reward >= upright_threshold)`
   with `upright_threshold = 0.6` by default. Cartpole-swingup's dense
   reward is a tolerance around the upright + centered configuration; a
   threshold of $0.6$ is strict enough to require active balancing.
3. **Observation flattening**. Sorted keys, so the deterministic layout is
   $\\mathrm{position\\_flat}, \\mathrm{velocity\\_flat}$, i.e.
   $(\\mathrm{cart\\_x}, \\cos\\theta, \\sin\\theta, \\mathrm{cart\\_v},
   \\dot\\theta)$, $5$ floats.
4. **No rendering**.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

try:
    import numpy as np
    from dm_control import suite
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wmel.envs.dmc_cartpole requires the `control` extras. "
        "Install with `pip install -e \".[control]\"`."
    ) from exc

from wmel.adapters.base import BenchmarkEnvironment


Action = Tuple[float, ...]
Observation = Tuple[float, ...]


DEFAULT_DISCRETE_LEVELS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
DEFAULT_UPRIGHT_THRESHOLD = 0.6


def _flatten_observation(obs_dict) -> Observation:
    """DMC observation `dict` -> flat `tuple[float, ...]`, deterministic order.

    Keys are sorted (`position` < `velocity`), giving the layout
    `(cart_x, cos_theta, sin_theta, cart_v, theta_dot)` for cartpole-swingup.
    """
    parts: list[float] = []
    for key in sorted(obs_dict.keys()):
        arr = np.asarray(obs_dict[key]).flatten()
        parts.extend(float(x) for x in arr)
    return tuple(parts)


class DMCCartpoleEnv(BenchmarkEnvironment):
    """DMC Cartpole-swingup behind the `BenchmarkEnvironment` interface.

    Parameters mirror `DMCAcrobotEnv`. The only structural difference is
    the observation dimensionality ($5$ instead of $6$) and the layout of
    the underlying physics state.
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
            domain_name="cartpole",
            task_name="swingup",
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
        if not (isinstance(action, tuple) and len(action) == 1):
            raise ValueError(
                f"DMCCartpoleEnv expects a 1-tuple action like (0.5,); got {action!r}"
            )
        torque = np.asarray(action, dtype=np.float32)
        ts = self._env.step(torque)
        self._last_obs = _flatten_observation(ts.observation)
        self._last_reward = float(ts.reward) if ts.reward is not None else 0.0
        return self._last_obs

    def is_success(self) -> bool:
        return self._last_reward >= self._upright_threshold

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


def make_cartpole_oracle_dynamics(
    task_kwargs: dict | None = None,
    reset_every: int = 800,
) -> Callable[[Observation, Action], Observation]:
    """Build a `(state, action) -> next_state` oracle dynamics callable for
    Cartpole-swingup, side-effect-free from the caller's perspective.

    State layout from `_flatten_observation` is
    `(cart_x, cos_theta, sin_theta, cart_v, theta_dot)`. Reconstruction:

        qpos = [cart_x, atan2(sin_theta, cos_theta)]
        qvel = [cart_v, theta_dot]

    The atan2 reconstruction is lossy modulo $2\\pi$ revolutions of the
    pole, but Cartpole dynamics depend only on $\\sin / \\cos$ of the
    pole angle, so the loss does not affect physics.
    """
    sim_env = suite.load(
        domain_name="cartpole",
        task_name="swingup",
        task_kwargs=task_kwargs if task_kwargs is not None else {"random": 0},
    )
    sim_env.reset()

    counter = {"calls": 0}

    def _dynamics(state: Observation, action: Action) -> Observation:
        if counter["calls"] >= reset_every:
            sim_env.reset()
            counter["calls"] = 0

        cart_x, cos_t, sin_t, cart_v, theta_dot = state
        theta = math.atan2(sin_t, cos_t)
        physics = sim_env.physics
        physics.named.data.qpos["slider"] = cart_x
        physics.named.data.qpos["hinge_1"] = theta
        physics.named.data.qvel["slider"] = cart_v
        physics.named.data.qvel["hinge_1"] = theta_dot
        physics.forward()

        torque = np.asarray(action, dtype=np.float32)
        ts = sim_env.step(torque)
        counter["calls"] += 1
        return _flatten_observation(ts.observation)

    return _dynamics


def cartpole_upright_score(state: Observation, goal: Observation = ()) -> float:
    """Lower-is-better score for Cartpole-swingup: negative cosine of the pole angle.

    Cartpole-swingup is solved when the pole is upright (i.e.\\
    $\\cos\\theta = +1$). The flat observation layout from
    `_flatten_observation` is `(cart_x, cos_theta, sin_theta, cart_v,
    theta_dot)`, so the score is `-state[1]`.

    The `goal` argument is ignored to keep the
    `Score = Callable[[Observation, Observation], float]` contract.

    Caveat: this score is a simplification of the DMC reward, which also
    penalises the cart drifting away from the centre of the rail. The
    rail-centering term is not in the score; the planner can therefore
    win the score by pushing the pole upright in a way that drifts the
    cart toward the rail boundary. This is the same kind of approximation
    `acrobot_upright_score` makes (it ignores the smooth tolerance around
    the tip-to-target distance), kept here for symmetry between the two
    environments.
    """
    if len(state) < 2:
        return 0.0
    return -float(state[1])
