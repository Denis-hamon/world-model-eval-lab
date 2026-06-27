"""DreamerV3 under CEM on DMC Acrobot-swingup (Task 8: planner-capacity axis).

The v0.13 result showed the planner axis flips verdicts for TD-MPC2
(random-shooting INCONCLUSIVE -> CEM MODEL BOTTLENECK at the fixed init).
This experiment completes the planner column of the multi-model table for
DreamerV3: it *re-plans* the Task 6 DreamerV3 checkpoint with CEM instead of
random shooting -- no retraining, only planning compute.

Why batched CEM (experiments/GPU_ROADMAP.md, Task 8, note 2): the dmc_proprio
RSSM is ~an order of magnitude slower per dynamics call than the TD-MPC2
adapter, and `wmel.adapters.cem_planner.CEMPlanner` issues one scalar
`dynamics(state, action)` per (sample, horizon-step). We therefore reuse the
`BatchedCEMPlanner` from `experiments.dmc_acrobot._batched_cem` (identical CEM
math, deterministic at a fixed seed) with the batched DreamerV3 dynamics
(`make_dreamerv3_batched_dynamics`) on `device="cuda"`, so CEM's candidate
batch amortises the launch overhead. The oracle arm uses the trivial batched
wrapper around the dm_control oracle, so both arms share one planner type.

Usage
-----
    python -m experiments.dmc_acrobot.dreamerv3_cem_cpg --smoke
    python -m experiments.dmc_acrobot.dreamerv3_cem_cpg --varied-init --seed 0
    python -m experiments.dmc_acrobot.dreamerv3_cem_cpg \
        --agent-checkpoint results/dmc_acrobot/dreamerv3_acrobot.pt

Writes:
    results/dmc_acrobot/dreamerv3_cem_cpg.json
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

from wmel.adapters.dreamerv3_adapter import make_dreamerv3_batched_dynamics
from wmel.adapters.mlp_world_model import acrobot_upright_score
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_acrobot import (
    DEFAULT_DISCRETE_LEVELS,
    DMCAcrobotEnv,
    make_acrobot_oracle_dynamics,
)
from wmel.metrics import compute_scorecard, counterfactual_planning_gap, cpg_verdict
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report

from experiments._seeding import eval_varied_factory
from experiments.dmc_acrobot._batched_cem import (
    BatchedCEMPlanner,
    make_oracle_batched_dynamics,
)


_RESULTS_DIR = _REPO_ROOT / "results" / "dmc_acrobot"
DEFAULT_CHECKPOINT = _RESULTS_DIR / "dreamerv3_acrobot.pt"
JSON_PATH = _RESULTS_DIR / "dreamerv3_cem_cpg.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Tiny CEM config for end-to-end validation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--agent-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Ported DreamerV3 adapter checkpoint (from Task 6).",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the batched DreamerV3 dynamics (CEM candidate batch).",
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
            "cem_iters": 2,
            "cem_samples": 8,
            "cem_elites": 3,
            "plan_horizon": 8,
            "smoothing": 0.1,
            "benchmark_episodes": 2,
            "benchmark_horizon": 80,
        }
    return {
        "cem_iters": 3,
        "cem_samples": 24,
        "cem_elites": 6,
        "plan_horizon": 15,
        "smoothing": 0.1,
        "benchmark_episodes": 10,
        "benchmark_horizon": 500,
    }


def _run_arm(*, name: str, batched_dynamics, cfg: dict, seed: int, levels, varied_init: bool):
    env_template = DMCAcrobotEnv(discrete_levels=levels)
    planner = BatchedCEMPlanner(
        dynamics=batched_dynamics,
        action_space=env_template.action_space,
        num_iterations=cfg["cem_iters"],
        num_samples=cfg["cem_samples"],
        num_elites=cfg["cem_elites"],
        plan_horizon=cfg["plan_horizon"],
        smoothing=cfg["smoothing"],
        score=acrobot_upright_score,
        seed=seed,
    )
    env_factory = (
        eval_varied_factory(DMCAcrobotEnv, seed, discrete_levels=levels)
        if varied_init
        else (lambda: DMCAcrobotEnv(discrete_levels=levels))
    )
    results = BenchmarkRunner(
        env_factory=env_factory,
        policy=planner,
        episodes=cfg["benchmark_episodes"],
        horizon=cfg["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()
    card = compute_scorecard(
        results,
        policy_name=f"cem ({name})",
        compute_per_plan_call=planner.compute_per_plan_call,
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

    print("[1/2] CEM with ORACLE dynamics...")
    oracle_results, oracle_card = _run_arm(
        name="oracle dynamics",
        batched_dynamics=make_oracle_batched_dynamics(make_acrobot_oracle_dynamics()),
        cfg=cfg, seed=seed, levels=levels, varied_init=args.varied_init,
    )

    print("\n[2/2] CEM with DreamerV3 dynamics (batched, %s)..." % args.device)
    dreamer_results, dreamer_card = _run_arm(
        name="DreamerV3 dynamics",
        batched_dynamics=make_dreamerv3_batched_dynamics(args.agent_checkpoint, device=args.device),
        cfg=cfg, seed=seed, levels=levels, varied_init=args.varied_init,
    )

    cpg = counterfactual_planning_gap(oracle_results, dreamer_results)
    verdict = cpg_verdict(cpg)
    print("\n[CPG] vs oracle:")
    print(f"  oracle    success = {cpg.oracle_success_rate:.3f} (n={cpg.n_episodes_oracle})")
    print(f"  DreamerV3 success = {cpg.learned_success_rate:.3f} (n={cpg.n_episodes_learned})")
    print(f"  CPG = {cpg.gap:+.3f}  95% AC CI [{cpg.gap_ci_low:+.3f}, {cpg.gap_ci_high:+.3f}]")
    print(f"  Verdict: {'SMOKE MODE' if args.smoke else verdict}")

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "metric": "counterfactual_planning_gap",
        "planner": "cem-batched",
        "learned_model": "dreamerv3",
        "cpg": {**asdict(cpg), "verdict": verdict},
        "config": cfg,
        "seed": seed,
        "smoke_mode": args.smoke,
        "varied_init": args.varied_init,
        "checkpoint": str(args.agent_checkpoint.relative_to(_REPO_ROOT)) if args.agent_checkpoint.is_relative_to(_REPO_ROOT) else str(args.agent_checkpoint),
        "oracle_scorecard": _card_to_dict(oracle_card),
        "learned_scorecard": _card_to_dict(dreamer_card),
        "oracle_full": to_json_report(oracle_results, oracle_card),
        "learned_full": to_json_report(dreamer_results, dreamer_card),
    }
    json_path = _RESULTS_DIR / f"dreamerv3_cem_cpg{args.out_suffix}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {json_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
