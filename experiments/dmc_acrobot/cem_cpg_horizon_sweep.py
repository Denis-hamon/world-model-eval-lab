"""Horizon-of-planning ablation under CEM on Acrobot.

The v0.13 pooled-150 cem_cpg cell holds plan_horizon fixed at 15 and asks
whether learned dynamics beats oracle dynamics under CEM. This script
sweeps `plan_horizon in {1, 5, 10, 15, 20, 30}` at the same pooled-150
budget per (H, arm) cell and asks the orthogonal question: as the
planner's effective rollout depth changes, does the CPG move?

Two diagnostic regimes:

- If CPG decreases monotonically with smaller H, compounding error of
  the learned dynamics is confirmed as the driver (short-horizon rollouts
  do not accumulate enough prediction error to matter).
- If CPG stays roughly flat across H, the bottleneck is independent of
  planning depth; the diagnosis shifts to off-manifold distribution shift
  at the first step or score-function mismatch.

The TD-MPC2 agent is loaded once and held fixed across all (H, seed)
cells, same convention as `cem_cpg_sweep.py`. Per-seed variance enters
through env init, CEM sampling, and MLP training; per-H variance comes
from the CEM planner's plan_horizon argument only.

Usage
-----
    python -m experiments.dmc_acrobot.cem_cpg_horizon_sweep --smoke
    python -m experiments.dmc_acrobot.cem_cpg_horizon_sweep
    python -m experiments.dmc_acrobot.cem_cpg_horizon_sweep --horizons 1,5,10
    python -m experiments.dmc_acrobot.cem_cpg_horizon_sweep --seeds 0,1,2,3

Writes:
    results/dmc_acrobot/cem_cpg_horizon_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from wmel.adapters.mlp_world_model import (
    acrobot_upright_score,
    collect_random_rollouts,
    train_world_model,
)
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_acrobot import DEFAULT_DISCRETE_LEVELS, DMCAcrobotEnv, make_acrobot_oracle_dynamics
from wmel.metrics import compute_scorecard, counterfactual_planning_gap, cpg_verdict
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report

from experiments.dmc_acrobot.coverage_mlp_on_tdmpc2 import (
    _collect_tdmpc2_rollouts,
    _coverage_stats,
    _load_tdmpc2_agent,
)
from experiments.dmc_acrobot._batched_cem import (
    BatchedCEMPlanner,
    make_mlp_batched_dynamics,
    make_oracle_batched_dynamics,
    make_tdmpc2_batched_dynamics,
)


TDMPC2_AGENT_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "tdmpc2_agent.pt"
TDMPC2_DYNAMICS_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "tdmpc2_acrobot.pt"
JSON_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "cem_cpg_horizon_sweep.json"

DEFAULT_HORIZONS = (1, 5, 10, 15, 20, 30)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Tiny CEM config, random data (no TD-MPC2 ckpt needed).")
    p.add_argument("--seeds", default="0,1,2", help="Comma-separated list of seeds (default 0,1,2).")
    p.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
                   help="Comma-separated list of plan_horizon values (default 1,5,10,15,20,30).")
    p.add_argument("--episodes", type=int, default=50, help="Episodes per (seed, H, arm) cell. Default 50 -> pooled 150 across 3 seeds.")
    p.add_argument("--n-mlp-transitions", type=int, default=20_000)
    p.add_argument("--output", default=None, help="Output JSON path (default results/dmc_acrobot/cem_cpg_horizon_sweep.json). Use a unique value when running parallel partial sweeps; merge via experiments.dmc_acrobot.pool_horizon_sweep.")
    return p.parse_args()


def _smoke_cfg() -> dict:
    return {
        "n_mlp_transitions": 200,
        "mlp_epochs": 20,
        "cem_iters": 2,
        "cem_samples": 8,
        "cem_elites": 3,
        "smoothing": 0.1,
        "benchmark_episodes": 2,
        "benchmark_horizon": 80,
    }


def _full_cfg(n_mlp_transitions: int, episodes: int) -> dict:
    return {
        "n_mlp_transitions": n_mlp_transitions,
        "mlp_epochs": 200,
        "cem_iters": 3,
        "cem_samples": 24,
        "cem_elites": 6,
        "smoothing": 0.1,
        "benchmark_episodes": episodes,
        "benchmark_horizon": 500,
    }


def _run_cem_arm(*, name: str, dynamics, cfg: dict, plan_horizon: int, seed: int, levels):
    env_template = DMCAcrobotEnv(discrete_levels=levels)
    planner = BatchedCEMPlanner(
        dynamics=dynamics,
        action_space=env_template.action_space,
        num_iterations=cfg["cem_iters"],
        num_samples=cfg["cem_samples"],
        num_elites=cfg["cem_elites"],
        plan_horizon=plan_horizon,
        smoothing=cfg["smoothing"],
        score=acrobot_upright_score,
        seed=seed,
    )
    results = BenchmarkRunner(
        env_factory=lambda: DMCAcrobotEnv(discrete_levels=levels),
        policy=planner,
        episodes=cfg["benchmark_episodes"],
        horizon=cfg["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()
    card = compute_scorecard(
        results,
        policy_name=f"cem H={plan_horizon} ({name})",
        compute_per_plan_call=planner.compute_per_plan_call,
        perturbation_name="env-default",
    )
    print_scorecard(card)
    return results, card


def _card_to_dict(card) -> dict | None:
    if card is None:
        return None
    return {
        "policy_name": card.policy_name,
        "success_rate": card.success_rate,
        "average_steps_to_success": card.average_steps_to_success,
        "average_planning_latency_ms": card.average_planning_latency_ms,
        "average_compute_per_decision": card.average_compute_per_decision,
        "episodes": card.episodes,
    }


def _train_seed_mlp(*, seed: int, cfg: dict, levels, use_tdmpc2_data: bool, agent_loaded):
    """Collect data + train the MLP for one seed. H-independent."""
    if use_tdmpc2_data:
        agent = agent_loaded()
        print(f"[seed {seed}] Collecting {cfg['n_mlp_transitions']} transitions from TD-MPC2 eval policy...")
        transitions = _collect_tdmpc2_rollouts(
            agent=agent,
            env_factory=lambda: DMCAcrobotEnv(discrete_levels=levels),
            n_transitions=cfg["n_mlp_transitions"],
            levels=levels,
            seed=seed,
        )
    else:
        n_eps = max(1, cfg["n_mlp_transitions"] // 200)
        print(f"[seed {seed}] Collecting {cfg['n_mlp_transitions']} random transitions ({n_eps} eps)...")
        transitions = collect_random_rollouts(
            lambda: DMCAcrobotEnv(discrete_levels=levels),
            n_episodes=n_eps,
            max_steps_per_episode=200,
            seed=seed,
        )
        transitions = transitions[: cfg["n_mlp_transitions"]]

    coverage = _coverage_stats(transitions, levels)
    print(f"[seed {seed}] coverage u_max={coverage['uprightness_max']:.3f}  frac(u>1)={coverage['fraction_u_gt_1']:.3f}")

    print(f"[seed {seed}] Training MLP ({cfg['mlp_epochs']} epochs)...")
    env_template = DMCAcrobotEnv(discrete_levels=levels)
    mlp_model, mlp_log = train_world_model(
        transitions,
        obs_dim=6,
        n_actions=len(env_template.action_space),
        epochs=cfg["mlp_epochs"],
        seed=seed,
    )
    print(f"[seed {seed}]   val_mse={mlp_log['final_val_mse']:.6f}")
    return mlp_model, mlp_log, coverage, env_template.action_space


def _pool_episodes(per_seed_results):
    pooled = []
    for r in per_seed_results:
        if r is None:
            return None
        pooled.extend(r)
    return pooled


def main() -> None:
    args = _parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]
    cfg = _smoke_cfg() if args.smoke else _full_cfg(args.n_mlp_transitions, args.episodes)
    levels = DEFAULT_DISCRETE_LEVELS

    use_tdmpc2_data = (not args.smoke) and TDMPC2_AGENT_PATH.exists() and TDMPC2_DYNAMICS_PATH.exists()
    if (not args.smoke) and not use_tdmpc2_data:
        raise FileNotFoundError(
            f"Missing {TDMPC2_AGENT_PATH} or {TDMPC2_DYNAMICS_PATH}; run experiments.dmc_acrobot.tdmpc2_cpg first or pass --smoke."
        )

    _agent_cache: dict = {}
    def agent_loaded():
        if "agent" not in _agent_cache:
            print(f"[setup] Loading TD-MPC2 agent from {TDMPC2_AGENT_PATH.name} (held fixed across (H, seed))...")
            _agent_cache["agent"] = _load_tdmpc2_agent(TDMPC2_AGENT_PATH, seed=seeds[0])
        return _agent_cache["agent"]

    print(f"[setup] seeds={seeds}, horizons={horizons}, episodes_per_cell={cfg['benchmark_episodes']}, smoke={args.smoke}, use_tdmpc2_data={use_tdmpc2_data}")

    per_seed: list = []
    for s in seeds:
        print(f"\n========== SEED {s} (data + MLP) ==========")
        mlp_model, mlp_log, coverage, action_space = _train_seed_mlp(
            seed=s, cfg=cfg, levels=levels, use_tdmpc2_data=use_tdmpc2_data, agent_loaded=agent_loaded,
        )

        # Batched dynamics: build once per seed, reuse across all H values.
        # The TD-MPC2 model is loaded once into the batched adapter.
        oracle_dyn_batched = make_oracle_batched_dynamics(make_acrobot_oracle_dynamics())
        mlp_dyn_batched = make_mlp_batched_dynamics(mlp_model, action_space)
        if use_tdmpc2_data:
            tdmpc2_dyn_batched = make_tdmpc2_batched_dynamics(TDMPC2_DYNAMICS_PATH, device="cpu")
        else:
            tdmpc2_dyn_batched = None

        per_H: dict[int, dict] = {}
        for H in horizons:
            print(f"\n--- SEED {s}, H={H} ---")
            t0 = time.time()
            oracle_results, oracle_card = _run_cem_arm(
                name="oracle dynamics", dynamics=oracle_dyn_batched,
                cfg=cfg, plan_horizon=H, seed=s, levels=levels,
            )
            mlp_results, mlp_card = _run_cem_arm(
                name=f"MLP on {'tdmpc2' if use_tdmpc2_data else 'random'} data",
                dynamics=mlp_dyn_batched,
                cfg=cfg, plan_horizon=H, seed=s, levels=levels,
            )
            if tdmpc2_dyn_batched is not None:
                tdmpc2_results, tdmpc2_card = _run_cem_arm(
                    name="TD-MPC2 dynamics",
                    dynamics=tdmpc2_dyn_batched,
                    cfg=cfg, plan_horizon=H, seed=s, levels=levels,
                )
            else:
                tdmpc2_results, tdmpc2_card = None, None
            dt = time.time() - t0
            print(f"[seed {s}, H={H}] elapsed {dt:.1f}s")
            per_H[H] = {
                "oracle_results": oracle_results,
                "mlp_results": mlp_results,
                "tdmpc2_results": tdmpc2_results,
                "oracle_card": oracle_card,
                "mlp_card": mlp_card,
                "tdmpc2_card": tdmpc2_card,
            }

        per_seed.append({
            "seed": s,
            "coverage": coverage,
            "mlp_training": mlp_log,
            "per_H": per_H,
        })

    # Pool across seeds for each H.
    print("\n========== POOLED ==========")
    cells: list[dict] = []
    for H in horizons:
        oracle_pool = _pool_episodes([out["per_H"][H]["oracle_results"] for out in per_seed])
        mlp_pool = _pool_episodes([out["per_H"][H]["mlp_results"] for out in per_seed])
        tdmpc2_pool = _pool_episodes([out["per_H"][H]["tdmpc2_results"] for out in per_seed])

        oracle_pooled_card = compute_scorecard(
            oracle_pool,
            policy_name=f"cem H={H} (oracle dynamics) pooled",
            compute_per_plan_call=per_seed[0]["per_H"][H]["oracle_card"].average_compute_per_decision,
            perturbation_name="env-default",
        )
        mlp_pooled_card = compute_scorecard(
            mlp_pool,
            policy_name=f"cem H={H} (MLP on {'tdmpc2' if use_tdmpc2_data else 'random'} data) pooled",
            compute_per_plan_call=per_seed[0]["per_H"][H]["mlp_card"].average_compute_per_decision,
            perturbation_name="env-default",
        )
        if tdmpc2_pool is not None:
            tdmpc2_pooled_card = compute_scorecard(
                tdmpc2_pool,
                policy_name=f"cem H={H} (TD-MPC2 dynamics) pooled",
                compute_per_plan_call=per_seed[0]["per_H"][H]["tdmpc2_card"].average_compute_per_decision,
                perturbation_name="env-default",
            )
        else:
            tdmpc2_pooled_card = None

        cpgs: dict[str, dict] = {}
        for name, results in [("mlp_on_data", mlp_pool), ("tdmpc2", tdmpc2_pool)]:
            if results is None:
                continue
            cpg = counterfactual_planning_gap(oracle_pool, results)
            v = cpg_verdict(cpg)
            cpgs[name] = {**asdict(cpg), "verdict": v}
            print(f"  H={H:>2d} oracle vs {name:>15s}: oracle={cpg.oracle_success_rate:.3f}  learned={cpg.learned_success_rate:.3f}  "
                  f"CPG={cpg.gap:+.3f}  CI [{cpg.gap_ci_low:+.3f}, {cpg.gap_ci_high:+.3f}]  "
                  f"{'SMOKE' if args.smoke else v}")

        cells.append({
            "plan_horizon": H,
            "pooled_n_per_arm": len(oracle_pool),
            "cpgs": cpgs,
            "pooled_oracle_scorecard": _card_to_dict(oracle_pooled_card),
            "pooled_mlp_scorecard": _card_to_dict(mlp_pooled_card),
            "pooled_tdmpc2_scorecard": _card_to_dict(tdmpc2_pooled_card),
            "oracle_full": to_json_report(oracle_pool, oracle_pooled_card),
            "mlp_full": to_json_report(mlp_pool, mlp_pooled_card),
            "tdmpc2_full": to_json_report(tdmpc2_pool, tdmpc2_pooled_card) if tdmpc2_pool else None,
        })

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "metric": "counterfactual_planning_gap",
        "planner": "cem",
        "sweep_axis": "plan_horizon",
        "mlp_data_source": "tdmpc2" if use_tdmpc2_data else "random",
        "tdmpc2_agent_seed_held_fixed": use_tdmpc2_data,
        "seeds": seeds,
        "horizons": horizons,
        "episodes_per_cell": cfg["benchmark_episodes"],
        "smoke_mode": args.smoke,
        "cfg": cfg,
        "per_seed": [
            {
                "seed": out["seed"],
                "coverage": out["coverage"],
                "mlp_training": out["mlp_training"],
                "per_H_scorecards": {
                    H: {
                        "oracle": _card_to_dict(out["per_H"][H]["oracle_card"]),
                        "mlp": _card_to_dict(out["per_H"][H]["mlp_card"]),
                        "tdmpc2": _card_to_dict(out["per_H"][H]["tdmpc2_card"]),
                    }
                    for H in horizons
                },
            }
            for out in per_seed
        ],
        "cells": cells,
    }
    out_path = Path(args.output) if args.output else JSON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
