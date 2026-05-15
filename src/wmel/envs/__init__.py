"""Bundled toy environments.

These live under `wmel.envs` (as opposed to `examples/`) so that they are
part of the installable package and reachable from the `wmel` console
script after `pip install`. The `examples/<env>/environment.py` modules
re-export the same classes for backward compatibility with existing
imports like `from examples.maze_toy.environment import MazeEnv`.
"""

from wmel.envs.maze_toy import (
    DEFAULT_LAYOUT as MAZE_DEFAULT_LAYOUT,
    MazeEnv,
    VALID_ACTIONS as MAZE_ACTIONS,
)
from wmel.envs.two_room_toy import (
    TwoRoomEnv,
    VALID_ACTIONS as TWO_ROOM_ACTIONS,
    two_room_waypoint_for,
)

__all__ = [
    "MazeEnv",
    "MAZE_ACTIONS",
    "MAZE_DEFAULT_LAYOUT",
    "TwoRoomEnv",
    "TWO_ROOM_ACTIONS",
    "two_room_waypoint_for",
]
