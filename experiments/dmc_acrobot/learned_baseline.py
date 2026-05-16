"""Learned-dynamics baseline on DMC Acrobot-swingup.

Trains a small Markovian MLP world model on random rollouts of Acrobot,
plugs it into `TabularWorldModelPlanner` via the existing `dynamics=`
argument, and benchmarks. The result is a scorecard committed under
`results/dmc_acrobot/learned_random_rollouts.json` for reproducibility.

What this experiment exposes (deliberately)

- **Prediction quality**: the model's held-out MSE on random-rollout
  transitions. With 10 episodes x 200 steps and 200 epochs, we typically
  get val_mse ~ 0.02 on a 6-dim observation - the model fits well.
- **Decision quality**: the planner's success rate on swing-up. With
  the model trained on random rollouts only, we honestly expect ~0%
  success: the planner queries the model on high-energy states that
  random data does not visit. The two metrics decouple, which is exactly
  the distribution-shift story the framework is meant to expose. The
  Counterfactual Planning Gap metric (v1.0) will quantify this directly.

Usage:

    pip install -e ".[dev,control,learned]"
    python -m experiments.dmc_acrobot.learned_baseline

Smoke mode for CI (shorter training, fewer episodes, ~30 s wall-clock):

    python -m experiments.dmc_acrobot.learned_baseline --smoke

Writes:

    results/dmc_acrobot/learned_random_rollouts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.adapters.mlp_world_model import (
    acrobot_upright_score,
    collect_random_rollouts,
    learned_dynamics,
    train_world_model,
)
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_acrobot import DMCAcrobotEnv
from wmel.metrics import compute_scorecard
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smaller config for CI smoke-tests (~30 s wall-clock).",
    )
    return parser.parse_args()


def _config(smoke: bool) -> dict:
    if smoke:
        return {
            "train_episodes": 5,
            "train_steps_per_episode": 80,
            "epochs": 30,
            "num_candidates": 20,
            "plan_horizon": 10,
            "benchmark_episodes": 2,
            "benchmark_horizon": 80,
        }
    return {
        "train_episodes": 10,
        "train_steps_per_episode": 200,
        "epochs": 200,
        "num_candidates": 50,
        "plan_horizon": 15,
        "benchmark_episodes": 10,
        "benchmark_horizon": 500,
    }


def main() -> None:
    args = _parse_args()
    cfg = _config(smoke=args.smoke)
    seed = 0

    env_template = DMCAcrobotEnv()
    print(f"[1/4] Collecting {cfg['train_episodes']} random rollouts...")
    transitions = collect_random_rollouts(
        DMCAcrobotEnv,
        n_episodes=cfg["train_episodes"],
        max_steps_per_episode=cfg["train_steps_per_episode"],
        seed=seed,
    )
    print(f"      Collected {len(transitions)} transitions.")

    print(f"[2/4] Training MLP world model ({cfg['epochs']} epochs)...")
    model, train_log = train_world_model(
        transitions,
        obs_dim=6,
        n_actions=len(env_template.action_space),
        epochs=cfg["epochs"],
        seed=seed,
        verbose=False,
    )
    print(
        f"      Final train MSE: {train_log['final_train_mse']:.4f}, "
        f"val MSE: {train_log['final_val_mse']:.4f}"
    )

    print("[3/4] Plugging learned model into TabularWorldModelPlanner...")
    dyn = learned_dynamics(model, env_template.action_space)
    planner = TabularWorldModelPlanner(
        dynamics=dyn,
        action_space=env_template.action_space,
        num_candidates=cfg["num_candidates"],
        plan_horizon=cfg["plan_horizon"],
        score=acrobot_upright_score,
        seed=seed,
    )

    print(
        f"[4/4] Benchmarking ({cfg['benchmark_episodes']} episodes, horizon "
        f"{cfg['benchmark_horizon']})..."
    )
    results = BenchmarkRunner(
        env_factory=DMCAcrobotEnv,
        policy=planner,
        episodes=cfg["benchmark_episodes"],
        horizon=cfg["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()

    card = compute_scorecard(
        results,
        policy_name="tabular-world-model (learned MLP dynamics)",
        compute_per_plan_call=planner.compute_per_plan_call,
        perturbation_name="env-default",
    )
    print_scorecard(card)

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "policy": "tabular-world-model + learned MLP dynamics",
        "config": cfg,
        "training": train_log,
        "model": {
            "kind": "MLP",
            "obs_dim": 6,
            "n_actions": len(env_template.action_space),
            "hidden": train_log["hidden"],
        },
        "seed": seed,
        "smoke_mode": args.smoke,
        **to_json_report(results, card),
    }

    out_path = (
        _REPO_ROOT / "results" / "dmc_acrobot" / "learned_random_rollouts.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
