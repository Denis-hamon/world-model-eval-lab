"""A Markovian world model in PyTorch for fully observed continuous-control envs.

This is the second learned-dynamics adapter (after the maze MLP). The intent
is to demonstrate the contract on a *non-toy* environment - DMC Acrobot - by
training a small neural net on rollout data and plugging the trained model
into `TabularWorldModelPlanner` via the existing `dynamics=` argument.

Architecture choice

We use a Markovian MLP rather than a GRU/Transformer. Acrobot is **fully
observed** and **Markov**: the next state depends only on the current state
and the action, not on history. A recurrent model adds parameters and
slows training without adding capacity that matches the dynamics. A GRU
adapter belongs to the roadmap when this framework moves to partially
observed envs (e.g., Acrobot from pixels) where temporal context carries
information the current observation does not.

The class name (`MLPWorldModel`) and the file name reflect this honesty.
The doc page on Pages explains the choice for readers who expect "world
model = recurrent" to find one here.

What this module ships

- `MLPWorldModel`: `(obs, action_idx) -> next_obs` predictor.
- `collect_random_rollouts(env_factory, n_episodes, max_steps_per_episode)`:
  generates `(obs, action, next_obs)` transitions for training.
- `train_world_model(transitions, ...)`: standard supervised training.
- `learned_dynamics(model, action_space)`: wraps the trained model as a
  `(state, action) -> next_state` callable compatible with
  `TabularWorldModelPlanner`.
- `acrobot_upright_score(state, goal)`: task-specific score function
  estimating the tip's vertical height from the flattened observation.
  Lower score = closer to upright. The planner minimises this.

Importing this module without torch raises a clear ImportError. Importing
the acrobot helpers without dm-control raises a clear ImportError too.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence, Tuple

try:
    import torch
    from torch import nn
    from torch.optim import Adam
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wmel.adapters.mlp_world_model requires PyTorch. "
        "Install with `pip install -e \".[learned]\"`."
    ) from exc

from wmel.adapters.base import BenchmarkEnvironment


Observation = Tuple[float, ...]
Action = Tuple[float, ...]


class MLPWorldModel(nn.Module):
    """Predict the next observation given the current observation and an action.

    Input: concatenation of (obs, one_hot(action_idx)). Output: next obs.

    Honest naming: this is an MLP, not a GRU. Acrobot is fully observed
    and Markov, so a recurrent model adds parameters without adding
    capacity. See the module docstring.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        if obs_dim <= 0 or n_actions <= 0 or hidden <= 0:
            raise ValueError("obs_dim, n_actions and hidden must all be positive")
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """obs: (batch, obs_dim) float; action_idx: (batch,) long. Returns (batch, obs_dim)."""
        a_oh = nn.functional.one_hot(action_idx, num_classes=self.n_actions).float()
        return self.net(torch.cat([obs, a_oh], dim=-1))


def collect_random_rollouts(
    env_factory: Callable[[], BenchmarkEnvironment],
    n_episodes: int,
    max_steps_per_episode: int,
    seed: int = 0,
) -> list[tuple[Observation, int, Observation]]:
    """Run `n_episodes` random rollouts. Return flat list of (obs, action_idx, next_obs).

    The action index is the position of the chosen action in the env's
    `action_space`, so it can be one-hot-encoded by the model.
    """
    rng = random.Random(seed)
    template = env_factory()
    actions = template.action_space
    transitions: list[tuple[Observation, int, Observation]] = []

    for _ in range(n_episodes):
        env = env_factory()
        obs = env.reset()
        for _ in range(max_steps_per_episode):
            a_idx = rng.randrange(len(actions))
            action = actions[a_idx]
            next_obs = env.step(action)
            transitions.append((tuple(obs), a_idx, tuple(next_obs)))
            obs = next_obs
            if env.is_success():
                break
    return transitions


def train_world_model(
    transitions: Sequence[tuple[Observation, int, Observation]],
    obs_dim: int,
    n_actions: int,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
    val_fraction: float = 0.1,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[MLPWorldModel, dict]:
    """Train an `MLPWorldModel` on `transitions` and return (model, train_log).

    `train_log` is a small dict with the final training and validation MSE
    on held-out transitions, useful for honest reporting independent of
    planning.

    Caveat: the train/val split is at the **transition** level, not the
    **trajectory** level - adjacent transitions from the same rollout can
    appear on both sides of the split. The val MSE therefore measures
    one-step prediction accuracy on transitions the model has not been
    trained on, but it is **not** a trajectory-level generalisation
    estimate. That stronger guarantee would need either episode-level
    splitting or rollout-error compounded over multiple steps.
    """
    if not transitions:
        raise ValueError("no transitions to train on")

    torch.manual_seed(seed)
    rng = random.Random(seed)

    shuffled = list(transitions)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val = shuffled[:n_val]
    train = shuffled[n_val:]

    def to_tensors(batch):
        obs = torch.tensor([t[0] for t in batch], dtype=torch.float32)
        act = torch.tensor([t[1] for t in batch], dtype=torch.long)
        nxt = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        return obs, act, nxt

    train_obs, train_act, train_next = to_tensors(train)
    val_obs, val_act, val_next = to_tensors(val)

    model = MLPWorldModel(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden)
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    final_train_loss = math.inf
    final_val_loss = math.inf
    for epoch in range(epochs):
        perm = torch.randperm(len(train_obs))
        epoch_loss = 0.0
        seen = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            obs_b, act_b, nxt_b = train_obs[idx], train_act[idx], train_next[idx]
            pred = model(obs_b, act_b)
            loss = loss_fn(pred, nxt_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
            seen += len(idx)
        final_train_loss = epoch_loss / max(1, seen)
        with torch.no_grad():
            val_pred = model(val_obs, val_act)
            final_val_loss = float(loss_fn(val_pred, val_next).item())
        if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
            print(
                f"epoch {epoch:>4}: train_mse = {final_train_loss:.6f}, "
                f"val_mse = {final_val_loss:.6f}"
            )

    model.eval()
    return model, {
        "epochs": epochs,
        "train_transitions": len(train),
        "val_transitions": len(val),
        "final_train_mse": float(final_train_loss),
        "final_val_mse": final_val_loss,
        "hidden": hidden,
    }


def learned_dynamics(
    model: MLPWorldModel,
    action_space: Sequence[Action],
) -> Callable[[Observation, Action], Observation]:
    """Wrap a trained `MLPWorldModel` as a `(state, action) -> next_state` callable.

    The callable is what `TabularWorldModelPlanner(dynamics=...)` expects.
    Conversion: state is a tuple of floats; action is one of the hashable
    tuples in `action_space`. The model receives one-hot-encoded action
    indices and predicts the next state.
    """
    model.eval()
    action_to_idx = {a: i for i, a in enumerate(action_space)}

    def _dynamics(state: Observation, action: Action) -> Observation:
        a_idx = action_to_idx[action]
        obs_t = torch.tensor([state], dtype=torch.float32)
        act_t = torch.tensor([a_idx], dtype=torch.long)
        with torch.no_grad():
            next_t = model(obs_t, act_t).squeeze(0)
        return tuple(float(x) for x in next_t)

    return _dynamics


def acrobot_upright_score(state: Observation, goal: Observation = ()) -> float:
    """Lower-is-better score for Acrobot: negative estimated tip height.

    DMC's Acrobot exposes orientations as
    `np.concatenate((horizontal, vertical))` where `horizontal` is the
    body-frame z-axis projected onto the world x-axis (i.e. `sin(angle)`)
    and `vertical` is the projection onto the world z-axis
    (i.e. `cos(angle)`); see `dm_control/suite/acrobot.py:Physics.
    orientations`. Both angles are **global (world-frame inclinations)**,
    not relative joint angles.

    After our sorted-keys flatten in
    `wmel.envs.dmc_acrobot._flatten_observation`, the observation is laid
    out as:

        state = (sin_upper, sin_lower, cos_upper, cos_lower, v0, v1)

    With unit-length arms, the tip's vertical position is

        tip_y = cos(upper) + cos(lower) = state[2] + state[3]

    Maximising `tip_y` is the swing-up objective. The score returned here
    is `-tip_y`, so that `TabularWorldModelPlanner` (which minimises
    score) drives the tip upward.

    The `goal` argument is ignored: the goal is the constant "upright" and
    is baked into the formula. We keep the signature to match the
    `Score = Callable[[Observation, Observation], float]` contract.

    Caveat: this formula approximates the DMC reward (which is a smooth
    tolerance around the actual `tip -> target` distance). It is exact
    enough to give a non-degenerate cost gradient over the relevant
    region; for a precise reward proxy the model would need to predict
    `physics.to_target()` directly.
    """
    if len(state) < 4:
        return 0.0
    cos_upper = state[2]
    cos_lower = state[3]
    return -float(cos_upper + cos_lower)
