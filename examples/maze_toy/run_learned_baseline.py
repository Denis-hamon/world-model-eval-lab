"""Compare an oracle stdlib dynamics with a PyTorch-learned MLP dynamics.

Same maze, same MPC planner, same evaluation contract. The point is the
contract: the learned model is a drop-in replacement for the oracle, and
its scorecard exposes the real cost of swapping a hand-written function
for a learned one (latency goes up, success rate stays at 100%).

Usage:

    pip install -e ".[learned]"
    python -m examples.maze_toy.run_learned_baseline
"""

from __future__ import annotations

import json
import sys
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
from wmel.benchmark_runner import BenchmarkRunner
from wmel.metrics import compute_scorecard
from wmel.report import print_scorecard, to_json_report

from examples.maze_toy.environment import VALID_ACTIONS, MazeEnv


EPISODES = 30
HORIZON = 60
PERTURB_PROB = 0.2
SEED = 0


def _env_factory() -> MazeEnv:
    return MazeEnv()


def main() -> None:
    template = MazeEnv()

    oracle_planner = TabularWorldModelPlanner(
        dynamics=template.dynamics,
        action_space=VALID_ACTIONS,
        num_candidates=200,
        plan_horizon=20,
        seed=SEED,
    )

    print("Training MLP dynamics on maze transitions (~64 samples, 800 epochs)...")
    model = train_maze_dynamics(template, epochs=800, seed=SEED)
    learned_dyn = torch_dynamics(model, template.width, template.height)
    learned_planner = TabularWorldModelPlanner(
        dynamics=learned_dyn,
        action_space=VALID_ACTIONS,
        num_candidates=200,
        plan_horizon=20,
        seed=SEED,
    )
    print("Training done.")

    oracle_results = BenchmarkRunner(
        env_factory=_env_factory,
        policy=oracle_planner,
        episodes=EPISODES,
        horizon=HORIZON,
        perturb_prob=PERTURB_PROB,
        seed=SEED,
    ).run()

    learned_results = BenchmarkRunner(
        env_factory=_env_factory,
        policy=learned_planner,
        episodes=EPISODES,
        horizon=HORIZON,
        perturb_prob=PERTURB_PROB,
        seed=SEED,
    ).run()

    oracle_card = compute_scorecard(
        oracle_results,
        policy_name="tabular-world-model (oracle dynamics)",
        compute_per_plan_call=oracle_planner.compute_per_plan_call,
        perturbation_name="env-default",
    )
    learned_card = compute_scorecard(
        learned_results,
        policy_name="tabular-world-model (learned MLP dynamics)",
        compute_per_plan_call=learned_planner.compute_per_plan_call,
        perturbation_name="env-default",
    )

    print_scorecard(oracle_card)
    print_scorecard(learned_card)

    report = {
        "environment": "maze_toy",
        "episodes": EPISODES,
        "horizon": HORIZON,
        "perturb_prob": PERTURB_PROB,
        "seed": SEED,
        "training": {
            "epochs": 800,
            "transitions": 64,
            "model": "MazeDynamicsMLP(hidden=32)",
        },
        "runs": {
            "oracle-dynamics": to_json_report(oracle_results, oracle_card),
            "learned-dynamics-mlp": to_json_report(learned_results, learned_card),
        },
    }
    out_path = Path(__file__).parent / "learned_baseline_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote learned baseline report to {out_path}")


if __name__ == "__main__":
    main()
