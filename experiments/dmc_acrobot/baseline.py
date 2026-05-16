"""Random-policy baseline on DeepMind Control Suite Acrobot-swingup.

The first run of the framework against a non-toy environment. Pins the
floor: a uniformly random policy over the discretised 5-level torque space
does not swing the pendulum up. Future experiments under this directory
will plug in a learned world model and report against this baseline.

Usage:

    pip install -e ".[dev,control]"
    python -m experiments.dmc_acrobot.baseline

Writes:

    results/dmc_acrobot/baseline_random.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.adapters.random_policy import RandomPolicy
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_acrobot import DMCAcrobotEnv
from wmel.metrics import compute_scorecard
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report


EPISODES = 20
HORIZON = 500
SEED = 0


def main() -> None:
    env_template = DMCAcrobotEnv()
    policy = RandomPolicy(action_space=env_template.action_space, seed=SEED)

    results = BenchmarkRunner(
        env_factory=DMCAcrobotEnv,
        policy=policy,
        episodes=EPISODES,
        horizon=HORIZON,
        perturb_prob=0.0,
        seed=SEED,
    ).run()

    card = compute_scorecard(
        results,
        policy_name="random",
        perturbation_name="env-default",
    )
    print_scorecard(card)

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "policy": "random",
        "episodes": EPISODES,
        "horizon_per_episode": HORIZON,
        "action_discretisation": list(t[0] for t in env_template.action_space),
        "upright_threshold": 0.6,
        "seed": SEED,
        **to_json_report(results, card),
    }

    out_path = _REPO_ROOT / "results" / "dmc_acrobot" / "baseline_random.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
