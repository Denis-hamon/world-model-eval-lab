"""Recurrence-truncation ablation on DMC Acrobot (Task 9).

The DreamerV3 adapter's documented limitation is the Markovian projection:
`make_dreamerv3_dynamics` re-encodes from the observation and truncates the
recurrent state to a one-frame posterior on *every* dynamics call. DreamerV3's
native planning mode instead imagines forward *in latent space* from a single
encode. This experiment runs both as planning arms next to the oracle and
reports the CPG difference between them -- that difference *is* the measured
cost of the truncation (experiments/GPU_ROADMAP.md, Task 9).

Three arms, same random-shooting planner, same env/oracle/score/seed:

  1. oracle dynamics                       (TabularWorldModelPlanner + oracle)
  2. DreamerV3 Markovian projection        (TabularWorldModelPlanner + make_dreamerv3_dynamics)
  3. DreamerV3 latent rollout              (DreamerV3LatentPlanner: encode once, imagine in latent)

Arms 2 and 3 share the *same* trained checkpoint (Task 6); they differ only
in how the world model is unrolled, so CPG(2) - CPG(3) isolates the
recurrence-truncation cost.

Usage
-----
    python -m experiments.dmc_acrobot.dreamerv3_latent_cpg --smoke
    python -m experiments.dmc_acrobot.dreamerv3_latent_cpg --varied-init --seed 0
    python -m experiments.dmc_acrobot.dreamerv3_latent_cpg \
        --agent-checkpoint results/dmc_acrobot/dreamerv3_acrobot.pt

Writes:
    results/dmc_acrobot/dreamerv3_latent_cpg.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    s = str(_entry)
    if _entry.is_dir() and s not in sys.path:
        sys.path.insert(0, s)

import torch

from wmel.adapters.dreamerv3_adapter import (
    make_dreamerv3_dynamics,
    make_dreamerv3_latent_planner,
)
from wmel.adapters.mlp_world_model import acrobot_upright_score
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_acrobot import (
    DEFAULT_DISCRETE_LEVELS,
    DMCAcrobotEnv,
    make_acrobot_oracle_dynamics,
)
from wmel.metrics import compute_scorecard, counterfactual_planning_gap, cpg_verdict
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report

from experiments._seeding import eval_varied_factory


_RESULTS_DIR = _REPO_ROOT / "results" / "dmc_acrobot"
DEFAULT_CHECKPOINT = _RESULTS_DIR / "dreamerv3_acrobot.pt"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Tiny config for end-to-end validation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--agent-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Ported DreamerV3 adapter checkpoint (from Task 6).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for the DreamerV3 arms (CPU is usually faster for the scalar planner).",
    )
    p.add_argument("--out-suffix", default="", help="Suffix for the output JSON, e.g. _seed1.")
    p.add_argument(
        "--varied-init",
        action="store_true",
        help="Vary the initial state per episode (seed shared across arms).",
    )
    return p.parse_args()


def _config(smoke: bool) -> dict:
    if smoke:
        return {
            "num_candidates": 15,
            "plan_horizon": 8,
            "benchmark_episodes": 2,
            "benchmark_horizon": 80,
        }
    return {
        "num_candidates": 50,
        "plan_horizon": 15,
        "benchmark_episodes": 10,
        "benchmark_horizon": 500,
    }


def _make_eval_factory(seed: int, levels, varied_init: bool):
    if varied_init:
        return eval_varied_factory(DMCAcrobotEnv, seed, discrete_levels=levels)
    return lambda: DMCAcrobotEnv(discrete_levels=levels)


def _run(policy, *, name: str, cfg: dict, seed: int, levels, varied_init: bool):
    results = BenchmarkRunner(
        env_factory=_make_eval_factory(seed, levels, varied_init),
        policy=policy,
        episodes=cfg["benchmark_episodes"],
        horizon=cfg["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()
    card = compute_scorecard(
        results,
        policy_name=name,
        compute_per_plan_call=policy.compute_per_plan_call,
        perturbation_name="env-default",
    )
    print_scorecard(card)
    return results, card


def _card_to_dict(card) -> dict:
    return {
        "policy_name": card.policy_name,
        "success_rate": card.success_rate,
        "average_steps_to_success": card.average_steps_to_success,
        "average_planning_latency_ms": card.average_planning_latency_ms,
        "average_compute_per_decision": card.average_compute_per_decision,
        "episodes": card.episodes,
    }


def main() -> None:
    args = _parse_args()
    cfg = _config(smoke=args.smoke)
    seed = args.seed
    levels = DEFAULT_DISCRETE_LEVELS

    if not args.agent_checkpoint.exists():
        raise FileNotFoundError(
            f"DreamerV3 checkpoint not found at {args.agent_checkpoint}. "
            "Run experiments.dmc_acrobot.dreamerv3_cpg (Task 6) first."
        )

    env_template = DMCAcrobotEnv(discrete_levels=levels)

    def tabular(dyn):
        return TabularWorldModelPlanner(
            dynamics=dyn,
            action_space=env_template.action_space,
            num_candidates=cfg["num_candidates"],
            plan_horizon=cfg["plan_horizon"],
            score=acrobot_upright_score,
            seed=seed,
        )

    print("[1/3] Random-shooting with ORACLE dynamics...")
    oracle_results, oracle_card = _run(
        tabular(make_acrobot_oracle_dynamics()),
        name="oracle dynamics", cfg=cfg, seed=seed, levels=levels, varied_init=args.varied_init,
    )

    print("\n[2/3] DreamerV3 Markovian projection (re-encode every step)...")
    markov_results, markov_card = _run(
        tabular(make_dreamerv3_dynamics(args.agent_checkpoint, device=args.device)),
        name="DreamerV3 Markovian projection", cfg=cfg, seed=seed, levels=levels, varied_init=args.varied_init,
    )

    print("\n[3/3] DreamerV3 latent rollout (encode once, imagine in latent)...")
    latent_planner = make_dreamerv3_latent_planner(
        args.agent_checkpoint,
        action_space=env_template.action_space,
        num_candidates=cfg["num_candidates"],
        plan_horizon=cfg["plan_horizon"],
        score=acrobot_upright_score,
        seed=seed,
        device=args.device,
    )
    latent_results, latent_card = _run(
        latent_planner,
        name="DreamerV3 latent rollout", cfg=cfg, seed=seed, levels=levels, varied_init=args.varied_init,
    )

    cpg_markov = counterfactual_planning_gap(oracle_results, markov_results)
    cpg_latent = counterfactual_planning_gap(oracle_results, latent_results)
    v_markov, v_latent = cpg_verdict(cpg_markov), cpg_verdict(cpg_latent)
    truncation_cost = cpg_markov.gap - cpg_latent.gap

    print("\n[CPG] vs oracle:")
    print(f"  Markovian projection: CPG={cpg_markov.gap:+.3f}  CI [{cpg_markov.gap_ci_low:+.3f}, {cpg_markov.gap_ci_high:+.3f}]  {'SMOKE' if args.smoke else v_markov}")
    print(f"  Latent rollout      : CPG={cpg_latent.gap:+.3f}  CI [{cpg_latent.gap_ci_low:+.3f}, {cpg_latent.gap_ci_high:+.3f}]  {'SMOKE' if args.smoke else v_latent}")
    print(f"  Recurrence-truncation cost (CPG_markov - CPG_latent) = {truncation_cost:+.3f}")

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "metric": "counterfactual_planning_gap",
        "planner": "random-shooting",
        "learned_model": "dreamerv3",
        "ablation": "recurrence_truncation",
        "cpg_markovian": {**asdict(cpg_markov), "verdict": v_markov},
        "cpg_latent": {**asdict(cpg_latent), "verdict": v_latent},
        "recurrence_truncation_cost": truncation_cost,
        "config": cfg,
        "seed": seed,
        "smoke_mode": args.smoke,
        "varied_init": args.varied_init,
        "checkpoint": str(args.agent_checkpoint.relative_to(_REPO_ROOT)) if args.agent_checkpoint.is_relative_to(_REPO_ROOT) else str(args.agent_checkpoint),
        "oracle_scorecard": _card_to_dict(oracle_card),
        "markovian_scorecard": _card_to_dict(markov_card),
        "latent_scorecard": _card_to_dict(latent_card),
        "oracle_full": to_json_report(oracle_results, oracle_card),
        "markovian_full": to_json_report(markov_results, markov_card),
        "latent_full": to_json_report(latent_results, latent_card),
    }
    json_path = _RESULTS_DIR / f"dreamerv3_latent_cpg{args.out_suffix}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {json_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
