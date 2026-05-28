"""Batched CEM planner + batched dynamics adapters for the horizon sweep.

Why this exists
---------------
`wmel.adapters.cem_planner.CEMPlanner` calls `dynamics(state, action)`
once per (sample, horizon-step), so a horizon sweep at H=30 with
num_samples=24 issues 720 sequential dynamics calls per CEM iteration.
For the TD-MPC2 latent dynamics that is the wall-clock bottleneck of the
phase-5o horizon ablation: each call pays a Python+tensor round-trip and
a fresh CUDA kernel launch.

`BatchedCEMPlanner` keeps the exact CEM math (deterministic rollouts,
identical sample order from the same `seed`, identical scoring rule) but
steps all `num_samples` rollouts in lockstep through one batched dynamics
call per horizon step. With num_samples=24 this is a 24x reduction in
dynamics-call overhead per plan.

The batched dynamics adapters take a per-call list of (state, action)
tuples and return a list of next-state tuples, mirroring the wmel
`Dynamics` contract but in vectorised form.

This module is experiment-local (used by `cem_cpg_horizon_sweep.py`).
It does not touch the wmel `CEMPlanner` runtime; the existing per-call
planner remains the published API and remains used by every other
experiment.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence, Tuple

import torch

from wmel.adapters.base import Action, Observation
from wmel.adapters.lewm_adapter_stub import LeWMAdapterStub, Latent
from wmel.adapters.mlp_world_model import MLPWorldModel
from wmel.adapters.tdmpc2_adapter import TDMPC2Dynamics


# Batched dynamics signature: take a list of states and a list of actions
# of equal length B, return a list of next-states of length B.
BatchedDynamics = Callable[[Sequence[Observation], Sequence[Action]], list[Observation]]
Score = Callable[[Observation, Observation], float]


class BatchedCEMPlanner(LeWMAdapterStub):
    """Cross-Entropy Method MPC with batched dynamics steps.

    `plan()`-compatible analogue of `wmel.adapters.cem_planner.CEMPlanner`
    whose `dynamics` callable has the batched signature
    `(states, actions) -> next_states`. Given a deterministic dynamics,
    the planned action sequence is identical to the unbatched CEMPlanner
    at the same `seed`, num_iterations, num_samples, plan_horizon and
    smoothing (sample-major, time-major RNG order; same scoring and
    elite-refit rules).

    `rollout()` is intentionally not implemented — this class is only used
    by `cem_cpg_horizon_sweep.py` via `plan()`, and the BenchmarkRunner
    never calls `rollout()` on the planner.
    """

    def __init__(
        self,
        dynamics: BatchedDynamics,
        action_space: Sequence[Action],
        num_iterations: int = 3,
        num_samples: int = 24,
        num_elites: int = 6,
        plan_horizon: int = 15,
        smoothing: float = 0.1,
        score: Score | None = None,
        seed: int | None = None,
    ) -> None:
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not (0 < num_elites <= num_samples):
            raise ValueError("num_elites must be in (0, num_samples]")
        if plan_horizon <= 0:
            raise ValueError("plan_horizon must be positive")
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError("smoothing must be in [0, 1]")
        if not action_space:
            raise ValueError("action_space must not be empty")
        self._dynamics = dynamics
        self._actions = list(action_space)
        self._action_to_idx = {a: i for i, a in enumerate(self._actions)}
        self._iterations = num_iterations
        self._num_samples = num_samples
        self._num_elites = num_elites
        self._plan_horizon = plan_horizon
        self._smoothing = smoothing
        self._score: Score = score if score is not None else _manhattan
        self._rng = random.Random(seed)
        self.compute_per_plan_call = float(num_iterations * num_samples * plan_horizon)

    @property
    def name(self) -> str:
        return "cem-batched"

    def encode(self, observation: Observation) -> Latent:
        return observation

    def score(self, latent: Latent, goal_latent: Latent) -> float:
        return self._score(latent, goal_latent)

    def plan(self, observation: Observation, goal: Observation, horizon: int) -> list[Action]:
        if horizon <= 0:
            return []

        z0 = self.encode(observation)
        gz = self.encode(goal)
        H = min(horizon, self._plan_horizon)
        n_act = len(self._actions)
        S = self._num_samples

        uniform_row = [1.0 / n_act] * n_act
        probs: list[list[float]] = [list(uniform_row) for _ in range(H)]

        best_sequence: list[Action] = []
        best_score = float("inf")

        for _ in range(self._iterations):
            # 1. Sample all S candidates upfront (same RNG order as unbatched CEM:
            #    we iterate sample-major, time-major to match the original).
            candidates: list[list[Action]] = []
            for _ in range(S):
                candidates.append([
                    self._actions[self._rng.choices(range(n_act), weights=probs[h], k=1)[0]]
                    for h in range(H)
                ])

            # 2. Batched rollout: at each h, dynamics(states[B], actions[B]).
            #    trajectories[s] is the per-sample list of states from z0 to z_H.
            states: list[Observation] = [z0] * S
            trajectories: list[list[Observation]] = [[z0] for _ in range(S)]
            for h in range(H):
                actions_h = [candidates[s][h] for s in range(S)]
                next_states = self._dynamics(states, actions_h)
                states = next_states
                for s in range(S):
                    trajectories[s].append(next_states[s])

            # 3. Score each candidate using the SAME earliest-best rule as
            #    the unbatched planner (truncate at the min-scoring intermediate
            #    state). Score loop is per-sample but very cheap.
            scored: list[tuple[float, int, list[Action]]] = []
            for s in range(S):
                traj = trajectories[s]
                cand_score = self._score(traj[-1], gz)
                cand_idx = len(candidates[s])
                for idx, state in enumerate(traj[1:], start=1):
                    sc = self._score(state, gz)
                    if sc < cand_score:
                        cand_score = sc
                        cand_idx = idx
                scored.append((cand_score, cand_idx, candidates[s]))

            scored.sort(key=lambda item: item[0])
            if scored[0][0] < best_score:
                best_score = scored[0][0]
                best_sequence = scored[0][2][: scored[0][1]]

            elites = scored[: self._num_elites]
            n_elite = max(1, len(elites))
            new_probs: list[list[float]] = []
            for h in range(H):
                counts = [0.0] * n_act
                for _s, _i, cand in elites:
                    counts[self._action_to_idx[cand[h]]] += 1.0
                empirical = [c / n_elite for c in counts]
                mixed = [
                    (1.0 - self._smoothing) * empirical[a] + self._smoothing * uniform_row[a]
                    for a in range(n_act)
                ]
                new_probs.append(mixed)
            probs = new_probs

        return best_sequence


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def make_tdmpc2_batched_dynamics(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> BatchedDynamics:
    """Batched-signature wrapper around the TDMPC2Dynamics checkpoint loader.

    The underlying `TDMPC2Dynamics.forward(obs, a)` is already vectorised;
    we just stage the (state, action) tuples into a (B, obs_dim) tensor
    pair per call. Math is identical to `make_tdmpc2_dynamics(...)`
    called B times: same model state, same forward.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    levels: Tuple[float, ...] = tuple(float(t) for t in ckpt["action_levels"])
    action_set = {(t,) for t in levels}

    model = TDMPC2Dynamics(**arch).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    @torch.no_grad()
    def _batched(states: Sequence[Observation], actions: Sequence[Action]) -> list[Observation]:
        if len(states) != len(actions):
            raise ValueError(f"batched dynamics needs equal-length states and actions; got {len(states)} vs {len(actions)}")
        for a in actions:
            if a not in action_set:
                raise KeyError(f"action {a!r} is not in the checkpoint's discrete action set {sorted(action_set)}")
        obs_t = torch.tensor([list(s) for s in states], dtype=torch.float32, device=device)
        a_t = torch.tensor([list(a) for a in actions], dtype=torch.float32, device=device)
        next_t = model(obs_t, a_t)
        return [tuple(float(x) for x in row) for row in next_t]

    return _batched


def make_mlp_batched_dynamics(model: MLPWorldModel, action_space: Sequence[Action]) -> BatchedDynamics:
    """Batched wrapper around the v0.11 MLPWorldModel forward.

    The MLP itself is vectorised; we batch the obs and action-index tensors.
    """
    actions = list(action_space)
    action_to_idx = {a: i for i, a in enumerate(actions)}
    device = next(model.parameters()).device

    @torch.no_grad()
    def _batched(states: Sequence[Observation], acts: Sequence[Action]) -> list[Observation]:
        if len(states) != len(acts):
            raise ValueError(f"batched dynamics needs equal-length states and actions")
        obs_t = torch.tensor([list(s) for s in states], dtype=torch.float32, device=device)
        a_idx = torch.tensor([action_to_idx[a] for a in acts], dtype=torch.long, device=device)
        next_t = model(obs_t, a_idx)
        return [tuple(float(x) for x in row) for row in next_t]

    return _batched


def make_oracle_batched_dynamics(unbatched: Callable[[Observation, Action], Observation]) -> BatchedDynamics:
    """Trivial batched wrapper around the per-call oracle dynamics.

    The oracle is dm_control-backed and not vectorised internally; we just
    iterate. Provided so the planner can use a single batched signature
    across all three arms.
    """
    def _batched(states: Sequence[Observation], actions: Sequence[Action]) -> list[Observation]:
        if len(states) != len(actions):
            raise ValueError(f"batched dynamics needs equal-length states and actions")
        return [unbatched(s, a) for s, a in zip(states, actions)]
    return _batched
