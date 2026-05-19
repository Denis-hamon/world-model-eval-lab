"""TD-MPC2 dynamics adapter for the wmel planner contract.

This is the first published-world-model adapter shipped by the framework
(the v0.11 worked example uses a small bespoke MLP, not a published model).
The point is to plug TD-MPC2's encoder + latent dynamics into
`TabularWorldModelPlanner` without dragging TD-MPC2's training-time deps
(hydra, gymnasium, tensordict, ...) into `src/wmel/`.

Approach: re-implement the minimal TD-MPC2 building blocks needed to load
weights (SimNorm, NormedLinear, the encoder/dynamics MLP layout), and pair
them with a small post-hoc decoder z -> obs that the training script trains
on the agent's replay buffer. The adapter knows how to *load and run* this
trio; the training script (`experiments/dmc_acrobot/tdmpc2_cpg.py`) knows
how to *produce* the checkpoint.

The contract matches `wmel.adapters.mlp_world_model.learned_dynamics`:
the factory returns a `(state, action) -> next_state` callable, where
`state` is the flat DMC Acrobot observation tuple
`(sin_upper, sin_lower, cos_upper, cos_lower, v_shoulder, v_elbow)` and
`action` is a 1-tuple `(torque,)` drawn from the discrete torque set
the checkpoint was trained against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wmel.adapters.tdmpc2_adapter requires PyTorch. "
        "Install with `pip install -e \".[learned]\"`."
    ) from exc


Observation = Tuple[float, ...]
Action = Tuple[float, ...]


class _SimNorm(nn.Module):
    """Simplicial normalisation (Lavoie et al., 2022; used by TD-MPC2).

    Reshapes the last dim into groups of `simnorm_dim`, applies softmax per
    group, then folds back. Must match TD-MPC2's `common.layers.SimNorm`
    layout exactly so that pretrained weights load cleanly.
    """

    def __init__(self, simnorm_dim: int = 8) -> None:
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class _NormedLinear(nn.Linear):
    """Linear + LayerNorm + activation. Same param order as TD-MPC2's
    `common.layers.NormedLinear`: `(weight, bias, ln.weight, ln.bias)`.
    """

    def __init__(self, in_features: int, out_features: int, act: nn.Module | None = None) -> None:
        super().__init__(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ln(super().forward(x)))


def _build_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int, final_act: nn.Module | None) -> nn.Sequential:
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(_NormedLinear(dims[i], dims[i + 1]))
    if final_act is not None:
        layers.append(_NormedLinear(dims[-2], dims[-1], act=final_act))
    else:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class TDMPC2Dynamics(nn.Module):
    """TD-MPC2 latent encoder + dynamics, plus a small obs decoder.

    The encoder and dynamics layouts mirror TD-MPC2's `common.layers.enc`
    and `common.world_model.WorldModel._dynamics`. The decoder is local to
    this adapter; TD-MPC2 itself does not reconstruct observations.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        enc_dim: int,
        mlp_dim: int,
        num_enc_layers: int,
        simnorm_dim: int,
        decoder_hidden: int,
    ) -> None:
        super().__init__()
        enc_hidden = [enc_dim] * max(num_enc_layers - 1, 1)
        self.encoder = _build_mlp(
            in_dim=obs_dim,
            hidden_dims=enc_hidden,
            out_dim=latent_dim,
            final_act=_SimNorm(simnorm_dim),
        )
        self.dynamics = _build_mlp(
            in_dim=latent_dim + action_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=latent_dim,
            final_act=_SimNorm(simnorm_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, obs_dim),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def step_latent(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, a], dim=-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, obs: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.decode(self.step_latent(self.encode(obs), a))


CheckpointDict = dict  # keys: "model_state", "arch", "action_levels", optionally "meta"


def make_tdmpc2_dynamics(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> Callable[[Observation, Action], Observation]:
    """Load a TD-MPC2 checkpoint and return a `(state, action) -> next_state`
    callable compatible with `wmel.adapters.tabular_world_model.TabularWorldModelPlanner`.

    The checkpoint must have been produced by
    `experiments.dmc_acrobot.tdmpc2_cpg` (or any pipeline that follows the
    same dict schema):

      - `model_state`: state_dict of `TDMPC2Dynamics`
      - `arch`: dict of `TDMPC2Dynamics.__init__` kwargs
      - `action_levels`: ordered tuple of float torques used as the
        discrete action set during training. Each wmel action is the
        1-tuple `(level,)`.

    The returned callable is pure: it allocates fresh tensors on every
    call and never mutates the model state. Throughput is dominated by
    Python-tensor round-trips, not by GPU compute, so we default to CPU
    inference (the planner does thousands of single-state evaluations and
    GPU launch overhead would dominate). Pass `device="cuda"` if you have
    a batched planner that can amortise the launch cost.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    levels: Tuple[float, ...] = tuple(float(t) for t in ckpt["action_levels"])
    action_set = {(t,) for t in levels}

    model = TDMPC2Dynamics(**arch).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    @torch.no_grad()
    def _dynamics(state: Observation, action: Action) -> Observation:
        if action not in action_set:
            raise KeyError(
                f"action {action!r} is not in the checkpoint's discrete action set {sorted(action_set)}"
            )
        obs_t = torch.tensor([state], dtype=torch.float32, device=device)
        a_t = torch.tensor([list(action)], dtype=torch.float32, device=device)
        next_t = model(obs_t, a_t).squeeze(0)
        return tuple(float(x) for x in next_t)

    return _dynamics
