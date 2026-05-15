"""A simple greedy policy for grid-like toy environments.

The policy routes the agent toward a known waypoint (typically a doorway)
when the goal is not in the same room, then greedily toward the goal. It is
not a world model; it exists as a sane reference baseline for the scorecard.
"""

from __future__ import annotations

from typing import Callable

from wmel.adapters.base import PlannerPolicy

Position = tuple[int, int]


def _step_toward(src: Position, dst: Position) -> str:
    """One axis-aligned action moving `src` closer to `dst`. Prefers x then y."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx != 0:
        return "right" if dx > 0 else "left"
    if dy != 0:
        return "up" if dy > 0 else "down"
    return "up"


class GreedyGridPolicy(PlannerPolicy):
    """Greedy policy for axis-aligned 2D grid environments.

    Parameters
    ----------
    waypoint_fn
        Optional callable mapping `(observation, goal) -> Position | None`. If
        it returns a position, the planner routes through it first. This is the
        hook used by the two-room env to feed in the doorway location.
    """

    def __init__(
        self,
        waypoint_fn: Callable[[Position, Position], Position | None] | None = None,
    ) -> None:
        self._waypoint_fn = waypoint_fn

    @property
    def name(self) -> str:
        return "greedy"

    def plan(
        self,
        observation: Position,
        goal: Position,
        horizon: int,
    ) -> list[str]:
        if horizon <= 0:
            return []

        actions: list[str] = []
        pos = observation
        waypoint = self._waypoint_fn(observation, goal) if self._waypoint_fn else None

        while len(actions) < horizon:
            target = waypoint if waypoint is not None and pos != waypoint else goal
            if pos == target and target is goal:
                break
            if pos == waypoint:
                waypoint = None
                target = goal
                if pos == goal:
                    break
            action = _step_toward(pos, target)
            actions.append(action)
            pos = _apply(pos, action)

        return actions


def _apply(pos: Position, action: str) -> Position:
    x, y = pos
    if action == "up":
        return (x, y + 1)
    if action == "down":
        return (x, y - 1)
    if action == "right":
        return (x + 1, y)
    if action == "left":
        return (x - 1, y)
    return pos
