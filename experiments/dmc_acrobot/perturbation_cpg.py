"""Perturbation-aware CPG on the three v0.12 dynamics arms under CEM.

Tests whether the v0.13 \\textsc{model bottleneck} verdict survives a
perturbed deployment, and whether the size of the gap moves with the
perturbation magnitude. The perturbation library
(\\texttt{wmel.perturbations}) plugs into \\texttt{BenchmarkRunner}
without touching the planner or the dynamics callable, so the same
CEM + oracle / TD-MPC2 / MLP-on-TD-MPC2-data triple from phase-5f and
phase-5h benchmarks against perturbed episodes by setting
\\texttt{perturb\\_prob = 1.0} and varying \\texttt{DropNextActions(k)}.

Why DropNextActions
-------------------
DMC Acrobot's \\texttt{env.perturb()} is a no-op (continuous-control
state-level perturbations are out of scope for the env wrapper that
shipped in v0.8). The remaining action-level perturbation in the
library, \\texttt{DropNextActions(k)}, drops the next k queued actions
mid-episode and forces the planner to re-plan from the resulting
state. This models actuator drops, network gaps, or a debouncing
layer swallowing a burst of commands. Varying k between $0$ (control)
and a value comparable to the CEM plan horizon ($15$) sweeps from no
perturbation to a near-full plan invalidation.

Reading the result
------------------
- Oracle drops, learned drops, gap unchanged: the verdict is robust
  to perturbation; the dynamics-quality bottleneck dominates.
- Oracle drops, learned stays at 0: the gap shrinks but verdict
  survives; the perturbation hurts the only arm that could succeed.
- Both unchanged: the perturbation is too mild relative to the
  CEM-replan margin; pick a larger k.

Usage
-----
    python -m experiments.dmc_acrobot.perturbation_cpg --smoke
    python -m experiments.dmc_acrobot.perturbation_cpg

Writes:
    results/dmc_acrobot/perturbation_cpg.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LAZY_LEGACY_OP", "0")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TDMPC2_PKG = _REPO_ROOT / "third_party" / "tdmpc2" / "tdmpc2"
for _entry in (_REPO_ROOT, _REPO_ROOT / "src", _TDMPC2_PKG):
    s = str(_entry)
    if _entry.is_dir() and s not in sys.path:
        sys.path.insert(0, s)

import torch

from wmel.adapters.cem_planner import CEMPlanner
from wmel.adapters.mlp_world_model import (
    acrobot_upright_score,
    collect_random_rollouts,
    learned_dynamics,
    train_world_model,
)
from wmel.adapters.tdmpc2_adapter import make_tdmpc2_dynamics
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_acrobot import DEFAULT_DISCRETE_LEVELS, DMCAcrobotEnv, make_acrobot_oracle_dynamics
from wmel.metrics import compute_scorecard, counterfactual_planning_gap, cpg_verdict
from wmel.perturbations import DropNextActions
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report

from experiments.dmc_acrobot.coverage_mlp_on_tdmpc2 import (
    _collect_tdmpc2_rollouts,
    _coverage_stats,
    _load_tdmpc2_agent,
)


TDMPC2_AGENT_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "tdmpc2_agent.pt"
TDMPC2_DYNAMICS_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "tdmpc2_acrobot.pt"
JSON_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "perturbation_cpg.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Tiny config, random data; validates wiring.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ks", default="0,1,5", help="Comma-separated DropNextActions k values to sweep.")
    p.add_argument("--n-mlp-transitions", type=int, default=20_000)
    return p.parse_args()


def _config(smoke: bool, n_mlp_transitions: int) -> dict:
    if smoke:
        return {
            "n_mlp_transitions": 200,
            "mlp_epochs": 20,
            "cem_iters": 2,
            "cem_samples": 8,
            "cem_elites": 3,
            "plan_horizon": 8,
            "smoothing": 0.1,
            "benchmark_episodes": 4,
            "benchmark_horizon": 80,
            "perturb_prob": 1.0,
        }
    return {
        "n_mlp_transitions": n_mlp_transitions,
        "mlp_epochs": 200,
        "cem_iters": 3,
        "cem_samples": 24,
        "cem_elites": 6,
        "plan_horizon": 15,
        "smoothing": 0.1,
        "benchmark_episodes": 50,
        "benchmark_horizon": 500,
        "perturb_prob": 1.0,
    }


def _run_cell(*, name: str, dynamics, cfg: dict, seed: int, levels, k: int):
    env_template = DMCAcrobotEnv(discrete_levels=levels)
    planner = CEMPlanner(
        dynamics=dynamics,
        action_space=env_template.action_space,
        num_iterations=cfg["cem_iters"],
        num_samples=cfg["cem_samples"],
        num_elites=cfg["cem_elites"],
        plan_horizon=cfg["plan_horizon"],
        smoothing=cfg["smoothing"],
        score=acrobot_upright_score,
        seed=seed,
    )
    # k=0 means no DropNextActions perturbation; pass None so the runner
    # falls back to EnvPerturbation (which is a no-op for Acrobot).
    perturbation = DropNextActions(k=k) if k > 0 else None
    results = BenchmarkRunner(
        env_factory=lambda: DMCAcrobotEnv(discrete_levels=levels),
        policy=planner,
        episodes=cfg["benchmark_episodes"],
        horizon=cfg["benchmark_horizon"],
        perturb_prob=(cfg["perturb_prob"] if k > 0 else 0.0),
        perturbation=perturbation,
        seed=seed,
    ).run()
    pert_name = perturbation.name if perturbation is not None else "no-op"
    card = compute_scorecard(
        results,
        policy_name=f"cem ({name})",
        compute_per_plan_call=planner.compute_per_plan_call,
        perturbation_name=pert_name,
    )
    print_scorecard(card)
    return results, card


def _card_to_dict(card) -> dict:
    if card is None:
        return None
    return {
        "policy_name": card.policy_name,
        "success_rate": card.success_rate,
        "perturbation_recovery_rate": card.perturbation_recovery_rate,
        "average_steps_to_success": card.average_steps_to_success,
        "average_planning_latency_ms": card.average_planning_latency_ms,
        "average_compute_per_decision": card.average_compute_per_decision,
        "episodes": card.episodes,
    }


def main() -> None:
    args = _parse_args()
    cfg = _config(smoke=args.smoke, n_mlp_transitions=args.n_mlp_transitions)
    ks = [int(k) for k in args.ks.split(",")]
    seed = args.seed
    levels = DEFAULT_DISCRETE_LEVELS

    use_tdmpc2_data = (not args.smoke) and TDMPC2_AGENT_PATH.exists() and TDMPC2_DYNAMICS_PATH.exists()
    if (not args.smoke) and not use_tdmpc2_data:
        raise FileNotFoundError(
            f"Missing {TDMPC2_AGENT_PATH} or {TDMPC2_DYNAMICS_PATH}; run experiments.dmc_acrobot.tdmpc2_cpg first or pass --smoke."
        )

    # Collect transitions and train MLP once (shared across all k values).
    if use_tdmpc2_data:
        print(f"[data] Loading TD-MPC2 agent and collecting {cfg['n_mlp_transitions']} eval-mode transitions...")
        agent = _load_tdmpc2_agent(TDMPC2_AGENT_PATH, seed=seed)
        transitions = _collect_tdmpc2_rollouts(
            agent=agent,
            env_factory=lambda: DMCAcrobotEnv(discrete_levels=levels),
            n_transitions=cfg["n_mlp_transitions"],
            levels=levels,
            seed=seed,
        )
    else:
        n_eps = max(1, cfg["n_mlp_transitions"] // 200)
        print(f"[data] Collecting {cfg['n_mlp_transitions']} random-policy transitions ({n_eps} eps)...")
        transitions = collect_random_rollouts(
            lambda: DMCAcrobotEnv(discrete_levels=levels),
            n_episodes=n_eps,
            max_steps_per_episode=200,
            seed=seed,
        )
        transitions = transitions[: cfg["n_mlp_transitions"]]
    coverage = _coverage_stats(transitions, levels)

    print(f"[mlp] Training MLP world model ({cfg['mlp_epochs']} epochs)...")
    env_template = DMCAcrobotEnv(discrete_levels=levels)
    mlp_model, mlp_log = train_world_model(
        transitions,
        obs_dim=6,
        n_actions=len(env_template.action_space),
        epochs=cfg["mlp_epochs"],
        seed=seed,
    )
    print(f"      val_mse={mlp_log['final_val_mse']:.6f}")

    cells: list[dict] = []
    for k in ks:
        print(f"\n========== k = {k} ({'no perturbation' if k == 0 else f'DropNextActions(k={k}) at every episode'}) ==========")

        print(f"[k={k}, arm 1/3] CEM on ORACLE dynamics")
        oracle_results, oracle_card = _run_cell(
            name="oracle dynamics", dynamics=make_acrobot_oracle_dynamics(),
            cfg=cfg, seed=seed, levels=levels, k=k,
        )

        print(f"[k={k}, arm 2/3] CEM on MLP-on-{'tdmpc2' if use_tdmpc2_data else 'random'} dynamics")
        mlp_results, mlp_card = _run_cell(
            name=f"MLP on {'tdmpc2' if use_tdmpc2_data else 'random'} data",
            dynamics=learned_dynamics(mlp_model, env_template.action_space),
            cfg=cfg, seed=seed, levels=levels, k=k,
        )

        if TDMPC2_DYNAMICS_PATH.exists():
            print(f"[k={k}, arm 3/3] CEM on TD-MPC2 dynamics")
            tdmpc2_results, tdmpc2_card = _run_cell(
                name="TD-MPC2 dynamics",
                dynamics=make_tdmpc2_dynamics(TDMPC2_DYNAMICS_PATH, device="cpu"),
                cfg=cfg, seed=seed, levels=levels, k=k,
            )
        else:
            tdmpc2_results, tdmpc2_card = None, None

        cpgs = {}
        for arm_name, results in [("mlp_on_data", mlp_results), ("tdmpc2", tdmpc2_results)]:
            if results is None:
                continue
            cpg = counterfactual_planning_gap(oracle_results, results)
            v = cpg_verdict(cpg)
            cpgs[arm_name] = {**asdict(cpg), "verdict": v}

        cells.append(
            {
                "k": k,
                "perturbation_name": ("no-op" if k == 0 else f"drop-next-{k}"),
                "cpgs": cpgs,
                "oracle_scorecard": _card_to_dict(oracle_card),
                "mlp_scorecard": _card_to_dict(mlp_card),
                "tdmpc2_scorecard": _card_to_dict(tdmpc2_card),
                "oracle_full": to_json_report(oracle_results, oracle_card),
                "mlp_full": to_json_report(mlp_results, mlp_card),
                "tdmpc2_full": to_json_report(tdmpc2_results, tdmpc2_card) if tdmpc2_results else None,
            }
        )

    print("\n========== SUMMARY ==========")
    print(f"{'k':>4s}  {'pert':>14s}  {'oracle':>7s}  {'mlp':>5s}  {'tdmpc2':>7s}  {'cpg_mlp':>18s}  {'cpg_tdmpc2':>18s}")
    for cell in cells:
        k = cell["k"]
        pn = cell["perturbation_name"]
        o = cell["oracle_scorecard"]["success_rate"]
        m = cell["mlp_scorecard"]["success_rate"]
        t = cell["tdmpc2_scorecard"]["success_rate"] if cell["tdmpc2_scorecard"] else None
        cpg_m = cell["cpgs"].get("mlp_on_data", {})
        cpg_t = cell["cpgs"].get("tdmpc2", {})
        cpg_m_s = f"{cpg_m.get('gap', 0):+.2f} [{cpg_m.get('gap_ci_low', 0):+.2f},{cpg_m.get('gap_ci_high', 0):+.2f}]"
        cpg_t_s = f"{cpg_t.get('gap', 0):+.2f} [{cpg_t.get('gap_ci_low', 0):+.2f},{cpg_t.get('gap_ci_high', 0):+.2f}]" if cpg_t else "n/a"
        t_s = f"{t:.3f}" if t is not None else "n/a"
        print(f"{k:>4d}  {pn:>14s}  {o:>7.3f}  {m:>5.3f}  {t_s:>7s}  {cpg_m_s:>18s}  {cpg_t_s:>18s}")

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "metric": "perturbation_aware_cpg",
        "planner": "cem",
        "mlp_data_source": "tdmpc2" if use_tdmpc2_data else "random",
        "tdmpc2_agent_seed_held_fixed": use_tdmpc2_data,
        "ks": ks,
        "config": cfg,
        "mlp_training": mlp_log,
        "coverage": coverage,
        "seed": seed,
        "smoke_mode": args.smoke,
        "cells": cells,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {JSON_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
