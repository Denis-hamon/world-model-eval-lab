"""Re-export of `wmel.envs.maze_toy` for backward compatibility.

The maze env now lives at `wmel.envs.maze_toy` so it ships with the
installed package and the `wmel` console script can use it. Existing
imports like `from examples.maze_toy.environment import MazeEnv` keep
working via this re-export.
"""

from wmel.envs.maze_toy import (
    DEFAULT_LAYOUT,
    MazeEnv,
    VALID_ACTIONS,
)

__all__ = ["DEFAULT_LAYOUT", "MazeEnv", "VALID_ACTIONS"]
