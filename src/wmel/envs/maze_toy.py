"""A small maze environment that exercises planning horizon.

Unlike the two-room env, naive Manhattan-greedy without a waypoint hint
cannot solve this one - the direct path is blocked and the planner has
to commit to a detour. This is the environment used to demonstrate the
`TabularWorldModelPlanner` contract end-to-end.

The default layout (read top-down; rows are stored highest-y first):

    # # # # # # #
    # S # . . . #
    # . # . # . #
    # . # . # . #
    # . . . # . #
    # . # # # G #
    # # # # # # #

`S` is the start (1, 5), `G` is the goal (5, 1). Width 7, height 7.
The optimal path is 12 actions long.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wmel.adapters.base import BenchmarkEnvironment

Action = str
Position = tuple[int, int]

VALID_ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")

DEFAULT_LAYOUT: tuple[str, ...] = (
    "#######",
    "#S#...#",
    "#.#.#.#",
    "#.#.#.#",
    "#...#.#",
    "#.###G#",
    "#######",
)


def _parse_layout(rows: tuple[str, ...]) -> tuple[int, int, Position, Position, frozenset[Position]]:
    height = len(rows)
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("layout rows must have equal width")

    walls: set[Position] = set()
    start: Position | None = None
    goal: Position | None = None

    for row_index, row in enumerate(rows):
        y = height - 1 - row_index
        for x, ch in enumerate(row):
            pos = (x, y)
            if ch == "#":
                walls.add(pos)
            elif ch == "S":
                start = pos
            elif ch == "G":
                goal = pos
            elif ch != ".":
                raise ValueError(f"unknown layout character {ch!r} at {pos}")

    if start is None:
        raise ValueError("layout has no start cell (S)")
    if goal is None:
        raise ValueError("layout has no goal cell (G)")

    return width, height, start, goal, frozenset(walls)


@dataclass
class MazeEnv(BenchmarkEnvironment):
    """A small deterministic grid maze.

    The default layout is `DEFAULT_LAYOUT`; pass `layout=` to override.
    """

    layout: tuple[str, ...] = DEFAULT_LAYOUT
    width: int = field(init=False)
    height: int = field(init=False)
    start: Position = field(init=False)
    goal_pos: Position = field(init=False)
    walls: frozenset[Position] = field(init=False)
    _agent: Position = field(init=False)
    _last_delta: tuple[int, int] = field(init=False, default=(0, 0))

    def __post_init__(self) -> None:
        self.width, self.height, self.start, self.goal_pos, self.walls = _parse_layout(self.layout)
        self._agent = self.start

    def reset(self) -> Position:
        self._agent = self.start
        self._last_delta = (0, 0)
        return self._agent

    def step(self, action: Action) -> Position:
        if action not in VALID_ACTIONS:
            raise ValueError(f"unknown action: {action!r}")
        dx, dy = _delta(action)
        candidate: Position = (self._agent[0] + dx, self._agent[1] + dy)
        if self._blocked(candidate):
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
        if self._blocked(candidate):
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

    def _blocked(self, pos: Position) -> bool:
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return pos in self.walls

    def dynamics(self, state: Position, action: Action) -> Position:
        """Pure function form of `step`. Useful for plugging into a planner.

        The function does not mutate any state. In a learned world model, this
        would be replaced by a forward pass of the predictor.
        """
        dx, dy = _delta(action)
        candidate: Position = (state[0] + dx, state[1] + dy)
        if self._blocked(candidate):
            return state
        return candidate


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
