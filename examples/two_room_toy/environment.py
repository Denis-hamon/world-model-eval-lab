"""Re-export of `wmel.envs.two_room_toy` for backward compatibility.

The two-room env now lives at `wmel.envs.two_room_toy` so it ships with
the installed package and the `wmel` console script can use it.
Existing imports keep working via this re-export.
"""

from wmel.envs.two_room_toy import (
    TwoRoomEnv,
    VALID_ACTIONS,
    two_room_waypoint_for,
)

__all__ = ["TwoRoomEnv", "VALID_ACTIONS", "two_room_waypoint_for"]
