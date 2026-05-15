"""Run random, greedy, and tabular-world-model baselines on the maze env.

Usage:

    python -m examples.maze_toy.run_baseline

Prints three scorecards and writes a sample JSON report to
`examples/maze_toy/sample_report.json`.

This example is the smallest setup that demonstrates the full evaluation
contract end-to-end: a concrete subclass of `LeWMAdapterStub` is plugged
into the same `BenchmarkRunner` as the existing baselines, with no other
changes to the surrounding code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.adapters.greedy_policy import GreedyGridPolicy
from wmel.adapters.random_policy import RandomPolicy
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
    template_env = MazeEnv()

    random_results = BenchmarkRunner(
        env_factory=_env_factory,
        policy=RandomPolicy(action_space=VALID_ACTIONS, seed=SEED),
        episodes=EPISODES,
        horizon=HORIZON,
        perturb_prob=PERTURB_PROB,
        seed=SEED,
    ).run()

    greedy_results = BenchmarkRunner(
        env_factory=_env_factory,
        policy=GreedyGridPolicy(),
        episodes=EPISODES,
        horizon=HORIZON,
        perturb_prob=PERTURB_PROB,
        seed=SEED,
    ).run()

    wm_planner = TabularWorldModelPlanner(
        dynamics=template_env.dynamics,
        action_space=VALID_ACTIONS,
        num_candidates=200,
        plan_horizon=20,
        seed=SEED,
    )
    wm_results = BenchmarkRunner(
        env_factory=_env_factory,
        policy=wm_planner,
        episodes=EPISODES,
        horizon=HORIZON,
        perturb_prob=PERTURB_PROB,
        seed=SEED,
    ).run()

    random_card = compute_scorecard(random_results, policy_name="random")
    greedy_card = compute_scorecard(greedy_results, policy_name="greedy-no-waypoint")
    wm_card = compute_scorecard(
        wm_results,
        policy_name="tabular-world-model",
        compute_per_plan_call=wm_planner.compute_per_plan_call,
    )

    print_scorecard(random_card)
    print_scorecard(greedy_card)
    print_scorecard(wm_card)

    report = {
        "environment": "maze_toy",
        "episodes": EPISODES,
        "horizon": HORIZON,
        "perturb_prob": PERTURB_PROB,
        "seed": SEED,
        "runs": {
            "random": to_json_report(random_results, random_card),
            "greedy-no-waypoint": to_json_report(greedy_results, greedy_card),
            "tabular-world-model": to_json_report(wm_results, wm_card),
        },
    }
    out_path = Path(__file__).parent / "sample_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote sample report to {out_path}")


if __name__ == "__main__":
    main()
