"""Run random and greedy baselines on the two-room toy environment.

Usage:

    python -m examples.two_room_toy.run_baseline

Prints two scorecards and writes a sample JSON report to
`examples/two_room_toy/sample_report.json`.
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
from wmel.benchmark_runner import BenchmarkRunner
from wmel.metrics import compute_scorecard
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report

from examples.two_room_toy.environment import VALID_ACTIONS, TwoRoomEnv, two_room_waypoint_for


EPISODES = 50
HORIZON = 60
PERTURB_PROB = 0.3
SEED = 0


def _env_factory() -> TwoRoomEnv:
    return TwoRoomEnv()


def main() -> None:
    waypoint_fn = two_room_waypoint_for(TwoRoomEnv())

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
        policy=GreedyGridPolicy(waypoint_fn=waypoint_fn),
        episodes=EPISODES,
        horizon=HORIZON,
        perturb_prob=PERTURB_PROB,
        seed=SEED,
    ).run()

    random_card = compute_scorecard(
        random_results, policy_name="random", perturbation_name="env-default"
    )
    greedy_card = compute_scorecard(
        greedy_results, policy_name="greedy", perturbation_name="env-default"
    )

    print_scorecard(random_card)
    print_scorecard(greedy_card)

    report = {
        **report_envelope_metadata(),
        "environment": "two_room_toy",
        "episodes": EPISODES,
        "horizon": HORIZON,
        "perturb_prob": PERTURB_PROB,
        "seed": SEED,
        "runs": {
            "random": to_json_report(random_results, random_card),
            "greedy": to_json_report(greedy_results, greedy_card),
        },
    }
    out_path = Path(__file__).parent / "sample_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote sample report to {out_path}")


if __name__ == "__main__":
    main()
