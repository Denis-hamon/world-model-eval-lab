"""A PyTorch-learned dynamics function, drop-in compatible with the
`TabularWorldModelPlanner` contract.

This module is the proof that the framework's evaluation contract is not
stdlib-specific: any callable that maps `(state, action) -> next_state` is
fair game, including a callable backed by a small neural network trained
from data. The intent is to refute the "okay but this only works because
the dynamics is hand-written" objection by demonstrating an end-to-end
plug-in of a learned model on the maze toy.

Scope (deliberately small):

- A tiny MLP (2 hidden layers, default 32 units) takes (state, action) as
  input and predicts (dx, dy) deltas.
- Training data is the full set of (state, action, next_state) transitions
  enumerated from the env's `dynamics` method. The grid is small enough
  (16 free cells x 4 actions = 64 transitions for the default maze) that
  the MLP memorises it deterministically in seconds on CPU.
- The trained MLP is wrapped as a stateless callable that the existing
  `TabularWorldModelPlanner` accepts as its `dynamics` argument. The
  planner does not know or care that the dynamics is learned.

What this is NOT:

- Not a generic env-learning framework. The training procedure assumes
  the env is small enough to enumerate.
- Not a benchmark of learned vs hand-coded dynamics. The point is the
  contract, not a horse race.

Importing this module without torch installed raises a clear ImportError.
The rest of the package keeps stdlib-only at runtime.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wmel.adapters.learned_dynamics_torch requires PyTorch. "
        "Install with `pip install -e \".[learned]\"`."
    ) from exc

from wmel.adapters.base import BenchmarkEnvironment

Position = Tuple[int, int]
Action = str

ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")
_ACTION_ID: dict[Action, int] = {a: i for i, a in enumerate(ACTIONS)}


class MazeDynamicsMLP(nn.Module):
    """A tiny MLP predicting (dx, dy) given (state, action) for a grid env.

    Input  : 6 dims  [x_norm, y_norm, action_one_hot(4)]
    Output : 2 dims  [dx, dy] (rounded at inference time)
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _encode(state: Position, action: Action, width: int, height: int) -> torch.Tensor:
    """One sample as a (6,) float tensor."""
    x, y = state
    x_norm = (x / max(width - 1, 1)) * 2.0 - 1.0
    y_norm = (y / max(height - 1, 1)) * 2.0 - 1.0
    a_idx = _ACTION_ID[action]
    onehot = [0.0, 0.0, 0.0, 0.0]
    onehot[a_idx] = 1.0
    return torch.tensor([x_norm, y_norm, *onehot], dtype=torch.float32)


def collect_transitions(
    env: BenchmarkEnvironment,
    actions: Sequence[Action] = ACTIONS,
) -> list[tuple[Position, Action, Position]]:
    """Enumerate every `(state, action, next_state)` transition from `env`.

    Requires the env to expose `width`, `height`, `walls` (set of blocked
    positions), and a pure `dynamics(state, action) -> state` method. The
    maze toy in `examples/maze_toy/` qualifies; other toy envs may not yet
    (the two-room env, for instance, does not expose `walls` as a set or a
    pure `dynamics` method - see the contributing notes).
    """
    if not all(hasattr(env, attr) for attr in ("width", "height", "walls", "dynamics")):
        raise TypeError(
            "collect_transitions requires width, height, walls, and dynamics on env"
        )

    width = env.width  # type: ignore[attr-defined]
    height = env.height  # type: ignore[attr-defined]
    walls = env.walls  # type: ignore[attr-defined]

    out: list[tuple[Position, Action, Position]] = []
    for x in range(width):
        for y in range(height):
            if (x, y) in walls:
                continue
            for action in actions:
                ns = env.dynamics((x, y), action)  # type: ignore[attr-defined]
                out.append(((x, y), action, ns))
    return out


def train_maze_dynamics(
    env: BenchmarkEnvironment,
    *,
    epochs: int = 800,
    lr: float = 1e-2,
    hidden: int = 32,
    seed: int = 0,
    verbose: bool = False,
) -> MazeDynamicsMLP:
    """Train an MLP to fit every `(state, action, next_state)` transition.

    Returns the trained model. Designed to memorise the transition table
    of a small grid env on CPU in well under a second.
    """
    torch.manual_seed(seed)

    transitions = collect_transitions(env)
    if not transitions:
        raise ValueError("env yielded zero transitions to train on")

    width = env.width  # type: ignore[attr-defined]
    height = env.height  # type: ignore[attr-defined]

    inputs = torch.stack([_encode(s, a, width, height) for s, a, _ in transitions])
    targets = torch.tensor(
        [[ns[0] - s[0], ns[1] - s[1]] for s, _, ns in transitions],
        dtype=torch.float32,
    )

    model = MazeDynamicsMLP(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(inputs)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:>4}: loss = {loss.item():.6f}")

    model.eval()
    return model


def torch_dynamics(
    model: MazeDynamicsMLP,
    width: int,
    height: int,
) -> Callable[[Position, Action], Position]:
    """Wrap a trained MLP as a stateless `(state, action) -> state` callable.

    Rounds the predicted (dx, dy) to the nearest integer; the resulting
    callable can be passed directly to `TabularWorldModelPlanner` as
    `dynamics=`.

    The wrapper does **not** validate that the input state is in-grid or
    that the predicted next state is. If the planner ever queries the
    learned model on an out-of-distribution input (off-grid coordinates,
    or a state not seen during training), the MLP extrapolates and the
    output may be nonsense. For the maze toy this never happens in
    practice because the MLP recovers the oracle bit-exact on every free
    cell, so simulated rollouts stay in-grid; on a more interesting env
    the wrapper would need clamping or rejection of OOB queries.
    """
    model.eval()

    def _dynamics(state: Position, action: Action) -> Position:
        x_in = _encode(state, action, width, height).unsqueeze(0)
        with torch.no_grad():
            dxdy = model(x_in).squeeze(0)
        dx = int(round(float(dxdy[0].item())))
        dy = int(round(float(dxdy[1].item())))
        return (state[0] + dx, state[1] + dy)

    return _dynamics
