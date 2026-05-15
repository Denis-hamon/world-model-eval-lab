"""Planning-horizon sweep on the maze toy environment.

Usage:

    python -m examples.maze_toy.run_horizon_sweep

Runs the `TabularWorldModelPlanner` at several lookahead depths and prints
the resulting success-rate / latency curve. This is the experiment that
backs the "Planning Horizon" metric described in `docs/02_metric_taxonomy.md`:
performance improves with horizon up to a point, then plateaus while latency
keeps rising.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.experiments import horizon_sweep, print_horizon_sweep

from examples.maze_toy.environment import VALID_ACTIONS, MazeEnv


PLAN_HORIZONS = (5, 10, 15, 20, 30)
EPISODES_PER_POINT = 30
EPISODE_HORIZON = 80
SEED = 0


def _policy_factory(plan_horizon: int) -> TabularWorldModelPlanner:
    template = MazeEnv()
    return TabularWorldModelPlanner(
        dynamics=template.dynamics,
        action_space=VALID_ACTIONS,
        num_candidates=200,
        plan_horizon=plan_horizon,
        seed=SEED,
    )


def main() -> None:
    sweep = horizon_sweep(
        env_factory=MazeEnv,
        policy_factory=_policy_factory,
        plan_horizons=PLAN_HORIZONS,
        episodes_per_point=EPISODES_PER_POINT,
        episode_horizon=EPISODE_HORIZON,
        perturb_prob=0.0,
        seed=SEED,
    )

    print_horizon_sweep(sweep)

    report = {
        "environment": "maze_toy",
        "policy_name": sweep.policy_name,
        "episodes_per_point": EPISODES_PER_POINT,
        "episode_horizon": EPISODE_HORIZON,
        "seed": SEED,
        "points": [
            {
                "plan_horizon": point.plan_horizon,
                "scorecard": asdict(point.scorecard),
                "success_ci": [point.success_ci_low, point.success_ci_high],
                "latency_ci_ms": [point.latency_ci_low, point.latency_ci_high],
            }
            for point in sweep.points
        ],
    }
    out_path = Path(__file__).parent / "horizon_sweep_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote horizon sweep report to {out_path}")


if __name__ == "__main__":
    main()
