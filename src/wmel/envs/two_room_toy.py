"""A tiny deterministic two-room grid environment.

Layout (width x height, default 9 x 7):

    . . . . W . . . .
    . . . . W . . . G
    . . . . W . . . .
    . . . . . . . . .   <- doorway row
    . . . . W . . . .
    . S . . W . . . .
    . . . . W . . . .

`S` is the start, `G` is the goal, `W` cells are wall cells, `.` are free
cells. The doorway is a single open row at `door_y` along the wall column
`wall_x`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wmel.adapters.base import BenchmarkEnvironment

Action = str
Position = tuple[int, int]

VALID_ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")


@dataclass
class TwoRoomEnv(BenchmarkEnvironment):
    """Two rooms separated by a vertical wall with one doorway.

    The environment is deterministic by default. `perturb()` snaps the agent
    one step in the direction it came from (a small displacement that the
    policy must recover from).
    """

    width: int = 9
    height: int = 7
    wall_x: int = 4
    door_y: int = 3
    start: Position = (1, 1)
    goal_pos: Position = (8, 5)
    _agent: Position = field(init=False)
    _last_delta: tuple[int, int] = field(init=False, default=(0, 0))

    def __post_init__(self) -> None:
        if not (0 <= self.wall_x < self.width):
            raise ValueError("wall_x out of bounds")
        if not (0 <= self.door_y < self.height):
            raise ValueError("door_y out of bounds")
        if self._is_wall(self.start) or self._out_of_bounds(self.start):
            raise ValueError("start is invalid")
        if self._is_wall(self.goal_pos) or self._out_of_bounds(self.goal_pos):
            raise ValueError("goal is invalid")
        self._agent = self.start

    def reset(self) -> Position:
        self._agent = self.start
        self._last_delta = (0, 0)
        return self._agent

    def step(self, action: Action) -> Position:
        if action not in VALID_ACTIONS:
            raise ValueError(f"unknown action: {action!r}")
        dx, dy = _delta(action)
        nx, ny = self._agent[0] + dx, self._agent[1] + dy
        candidate: Position = (nx, ny)
        if self._out_of_bounds(candidate) or self._is_wall(candidate):
            return self._agent
        self._agent = candidate
        self._last_delta = (dx, dy)
        return self._agent

    def is_success(self) -> bool:
        return self._agent == self.goal_pos

    def perturb(self) -> None:
        """Push the agent one step opposite to its last movement, if possible."""
        dx, dy = self._last_delta
        if dx == 0 and dy == 0:
            return
        candidate: Position = (self._agent[0] - dx, self._agent[1] - dy)
        if self._out_of_bounds(candidate) or self._is_wall(candidate):
            return
        self._agent = candidate

    @property
    def observation(self) -> Position:
        return self._agent

    @property
    def goal(self) -> Position:
        return self.goal_pos

    @property
    def action_space(self) -> tuple[Action, ...]:
        return VALID_ACTIONS

    @property
    def doorway(self) -> Position:
        return (self.wall_x, self.door_y)

    def _out_of_bounds(self, pos: Position) -> bool:
        x, y = pos
        return not (0 <= x < self.width and 0 <= y < self.height)

    def _is_wall(self, pos: Position) -> bool:
        x, y = pos
        return x == self.wall_x and y != self.door_y


def _delta(action: Action) -> tuple[int, int]:
    if action == "up":
        return (0, 1)
    if action == "down":
        return (0, -1)
    if action == "right":
        return (1, 0)
    if action == "left":
        return (-1, 0)
    raise ValueError(f"unknown action: {action!r}")


def two_room_waypoint_for(env: TwoRoomEnv):
    """Build a waypoint function for `env` that points at the doorway when needed."""
    wall_x = env.wall_x
    doorway = env.doorway

    def _waypoint(observation: Position, goal: Position) -> Position | None:
        if (observation[0] < wall_x) == (goal[0] < wall_x):
            return None
        return doorway

    return _waypoint
