"""Counterfactual Planning Gap (CPG) on DMC Acrobot-swingup.

The decomposition the v0.8 learned-baseline left open:

- Phase 2 (v0.8) shipped a learned MLP world model with low held-out
  prediction error (val_mse ~0.026) but zero success rate when plugged into
  the planner. Three candidate explanations remained: distribution shift,
  planner capacity, score approximation.

- This experiment (v0.9) discriminates between them. It runs the SAME
  TabularWorldModelPlanner with the SAME acrobot_upright_score on the same
  benchmark, twice: once with the **learned MLP dynamics** trained on random
  rollouts (the v0.8 setup), once with **oracle dynamics** that step the real
  Acrobot physics under the hood. CPG = success_rate(oracle) - success_rate
  (learned). The full formal definition is in `wmel.metrics.CPGResult`.

Reading the result:

- **CPG > 0**: the oracle planner solves more episodes. Model error is the
  bottleneck. Improving the model should close the gap.
- **CPG ~ 0 with both at zero**: the planner cannot solve the task even with
  a perfect model. Planner capacity is the bottleneck. No amount of model
  training will help; the framework needs a better search procedure (CEM,
  iLQR, gradient-based MPC).
- **CPG ~ 0 with both at non-zero**: the learned model is as good as the
  oracle for planning purposes. The framework is consistent.

Usage:

    pip install -e ".[dev,control,learned]"
    python -m experiments.dmc_acrobot.cpg

Smoke mode for CI (~20 s):

    python -m experiments.dmc_acrobot.cpg --smoke

Writes:

    results/dmc_acrobot/cpg.json
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

from dataclasses import asdict

from wmel.adapters.mlp_world_model import (
    acrobot_upright_score,
    collect_random_rollouts,
    learned_dynamics,
    train_world_model,
)
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_acrobot import DMCAcrobotEnv, make_acrobot_oracle_dynamics
from wmel.metrics import compute_scorecard, counterfactual_planning_gap, cpg_verdict
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smaller config for CI (~20 s wall-clock).",
    )
    return p.parse_args()


def _config(smoke: bool) -> dict:
    if smoke:
        return {
            "train_episodes": 3,
            "train_steps_per_episode": 60,
            "epochs": 20,
            "num_candidates": 15,
            "plan_horizon": 8,
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

    print(f"[1/5] Collecting {cfg['train_episodes']} random rollouts for training...")
    transitions = collect_random_rollouts(
        DMCAcrobotEnv,
        n_episodes=cfg["train_episodes"],
        max_steps_per_episode=cfg["train_steps_per_episode"],
        seed=seed,
    )
    print(f"      {len(transitions)} transitions.")

    print(f"[2/5] Training MLP world model ({cfg['epochs']} epochs)...")
    model, train_log = train_world_model(
        transitions,
        obs_dim=6,
        n_actions=len(env_template.action_space),
        epochs=cfg["epochs"],
        seed=seed,
    )
    print(
        f"      val_mse={train_log['final_val_mse']:.4f}, "
        f"train_mse={train_log['final_train_mse']:.4f}"
    )

    def make_planner(dyn) -> TabularWorldModelPlanner:
        return TabularWorldModelPlanner(
            dynamics=dyn,
            action_space=env_template.action_space,
            num_candidates=cfg["num_candidates"],
            plan_horizon=cfg["plan_horizon"],
            score=acrobot_upright_score,
            seed=seed,
        )

    print("[3/5] Benchmarking with ORACLE dynamics...")
    oracle_planner = make_planner(make_acrobot_oracle_dynamics())
    oracle_results = BenchmarkRunner(
        env_factory=DMCAcrobotEnv,
        policy=oracle_planner,
        episodes=cfg["benchmark_episodes"],
        horizon=cfg["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()
    oracle_card = compute_scorecard(
        oracle_results,
        policy_name="tabular-world-model (oracle dynamics)",
        compute_per_plan_call=oracle_planner.compute_per_plan_call,
        perturbation_name="env-default",
    )
    print_scorecard(oracle_card)

    print("[4/5] Benchmarking with LEARNED MLP dynamics...")
    learned_planner = make_planner(learned_dynamics(model, env_template.action_space))
    learned_results = BenchmarkRunner(
        env_factory=DMCAcrobotEnv,
        policy=learned_planner,
        episodes=cfg["benchmark_episodes"],
        horizon=cfg["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()
    learned_card = compute_scorecard(
        learned_results,
        policy_name="tabular-world-model (learned MLP dynamics)",
        compute_per_plan_call=learned_planner.compute_per_plan_call,
        perturbation_name="env-default",
    )
    print_scorecard(learned_card)

    print("[5/5] Computing Counterfactual Planning Gap...")
    cpg = counterfactual_planning_gap(oracle_results, learned_results)
    verdict = cpg_verdict(cpg)
    print(
        f"  oracle  success = {cpg.oracle_success_rate:.3f}  (n={cpg.n_episodes_oracle})"
    )
    print(
        f"  learned success = {cpg.learned_success_rate:.3f}  (n={cpg.n_episodes_learned})"
    )
    print(
        f"  CPG = {cpg.gap:+.3f}   95% AC CI [{cpg.gap_ci_low:+.3f}, "
        f"{cpg.gap_ci_high:+.3f}]"
    )
    if args.smoke:
        # The smoke config is too small for a meaningful verdict; printing
        # one would mislead a maintainer comparing it to the committed
        # full-run result. See `experiments/dmc_acrobot/README.md`.
        print(
            "  Verdict: SMOKE MODE (config too small for diagnosis; verdict suppressed)"
        )
    else:
        print(f"  Verdict: {verdict}")

    # Headline first, full per-episode payloads last - so `cat | head -40`
    # shows the conclusion rather than a wall of latency arrays.
    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "metric": "counterfactual_planning_gap",
        "cpg": {**asdict(cpg), "verdict": verdict},
        "config": cfg,
        "training": train_log,
        "seed": seed,
        "smoke_mode": args.smoke,
        "oracle_scorecard": {
            "policy_name": oracle_card.policy_name,
            "success_rate": oracle_card.success_rate,
            "average_steps_to_success": oracle_card.average_steps_to_success,
            "average_planning_latency_ms": oracle_card.average_planning_latency_ms,
            "average_compute_per_decision": oracle_card.average_compute_per_decision,
            "episodes": oracle_card.episodes,
        },
        "learned_scorecard": {
            "policy_name": learned_card.policy_name,
            "success_rate": learned_card.success_rate,
            "average_steps_to_success": learned_card.average_steps_to_success,
            "average_planning_latency_ms": learned_card.average_planning_latency_ms,
            "average_compute_per_decision": learned_card.average_compute_per_decision,
            "episodes": learned_card.episodes,
        },
        "oracle_full": to_json_report(oracle_results, oracle_card),
        "learned_full": to_json_report(learned_results, learned_card),
    }
    out_path = _REPO_ROOT / "results" / "dmc_acrobot" / "cpg.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
