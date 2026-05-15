"""Horizon sweep with the learned PyTorch MLP dynamics, paired with the
oracle sweep for a side-by-side comparison.

Reads the existing `horizon_sweep_report.json` produced by
`run_horizon_sweep.py` (the oracle dynamics curve) and adds a second
curve under the learned MLP dynamics. Same horizons, same seed.

Output: `examples/maze_toy/learned_horizon_sweep_report.json`, which the
SVG renderer reads to produce `docs/assets/horizon_sweep_compare.svg`.

Usage:

    pip install -e ".[learned]"
    python -m examples.maze_toy.run_horizon_sweep
    python -m examples.maze_toy.run_learned_sweep
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

from wmel.adapters.learned_dynamics_torch import (
    torch_dynamics,
    train_maze_dynamics,
)
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.experiments import horizon_sweep, print_horizon_sweep
from wmel.report import report_envelope_metadata

from examples.maze_toy.environment import VALID_ACTIONS, MazeEnv


PLAN_HORIZONS = (5, 10, 15, 20, 30)
EPISODES_PER_POINT = 30
EPISODE_HORIZON = 80
SEED = 0


def main() -> None:
    template = MazeEnv()

    print("Training MLP dynamics on maze transitions (~64 samples, 800 epochs)...")
    model = train_maze_dynamics(template, epochs=800, seed=SEED)
    learned_dyn = torch_dynamics(model, template.width, template.height)
    print("Training done.")

    def policy_factory(plan_horizon: int) -> TabularWorldModelPlanner:
        return TabularWorldModelPlanner(
            dynamics=learned_dyn,
            action_space=VALID_ACTIONS,
            num_candidates=200,
            plan_horizon=plan_horizon,
            seed=SEED,
        )

    print(f"Running learned-dynamics sweep across horizons {PLAN_HORIZONS}...")
    sweep = horizon_sweep(
        env_factory=MazeEnv,
        policy_factory=policy_factory,
        plan_horizons=PLAN_HORIZONS,
        episodes_per_point=EPISODES_PER_POINT,
        episode_horizon=EPISODE_HORIZON,
        perturb_prob=0.0,
        seed=SEED,
    )

    print_horizon_sweep(sweep)

    # Distinguish this sweep's policy_name from the oracle sweep's report,
    # which also uses TabularWorldModelPlanner and therefore reports the
    # same `policy.name`. Without this suffix anyone diffing the two
    # JSONs would think they came from the same policy.
    report = {
        **report_envelope_metadata(),
        "environment": "maze_toy",
        "policy_name": f"{sweep.policy_name} (learned-mlp)",
        "dynamics_kind": "learned-mlp",
        "episodes_per_point": EPISODES_PER_POINT,
        "episode_horizon": EPISODE_HORIZON,
        "seed": SEED,
        "training": {
            "epochs": 800,
            "transitions": 64,
            "model": "MazeDynamicsMLP(hidden=32)",
        },
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
    out_path = Path(__file__).parent / "learned_horizon_sweep_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote learned horizon sweep report to {out_path}")


if __name__ == "__main__":
    main()
