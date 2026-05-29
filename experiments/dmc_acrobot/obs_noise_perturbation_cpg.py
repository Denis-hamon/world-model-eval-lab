"""Observation-noise perturbation: differential CPG fragility test on Acrobot.

The v0.13 perturbation section flagged that the available DMC perturbation
(`DropNextActions(k)`) hurts every arm roughly equally — it perturbs the
*planner* by invalidating queued actions, not the *dynamics* by shifting
the observation distribution. A genuine differential fragility test needs
a perturbation that touches the model's input distribution: if the
learned dynamics has been trained on a particular observation manifold,
small Gaussian noise on the observation should *widen* the CPG (oracle
robust, learned arms fall over) — that is the cleanest evidence that
distribution shift, not planner capacity, is the fragility mode.

The `obs_noise_std` hook on `DMCAcrobotEnv` (added in this branch) adds
zero-mean Gaussian noise of configurable stdev to every returned
observation, leaving the underlying physics state untouched. The
planner — oracle, MLP, or TD-MPC2 — sees the noisy observation and
rolls out its own dynamics from there. Per-component drift of the
observation grows linearly with sigma; planner accuracy can degrade
non-linearly, which is the signal we are after.

Protocol
--------
Sweep sigma in {0.0, 0.01, 0.05, 0.1} on a fixed CEM config
(num_iterations=3, num_samples=24, plan_horizon=15) — the same config
the v0.13 `cem_cpg_sweep` cell used. Per sigma:

1. Build a DMCAcrobotEnv with `obs_noise_std=sigma`. BenchmarkRunner
   instantiates the env once per episode, and each episode gets a
   distinct (but reproducible) noise seed, so the per-cell success rate
   averages over `benchmark_episodes` independent noise realizations
   rather than replaying a single one.
2. Run a 50-episode CEM bench on the **oracle** dynamics arm.
3. Run a 50-episode CEM bench on the **MLP-on-TD-MPC2-data** arm.
4. Run a 50-episode CEM bench on the **TD-MPC2** dynamics arm.
5. Compute CPG vs oracle for the two learned arms.

Single seed (sigma sweep is the variation axis; not pooled across
RL seeds). Total wall: 4 sigma * 3 arms * 50 episodes.

Usage
-----
    python -m experiments.dmc_acrobot.obs_noise_perturbation_cpg --smoke

The two committed result JSONs were produced by these exact invocations
(H=1 is the informative regime; H=15 is the floored control):

    python -m experiments.dmc_acrobot.obs_noise_perturbation_cpg \
        --plan-horizon 1 \
        --output results/dmc_acrobot/obs_noise_perturbation_cpg_h1.json
    python -m experiments.dmc_acrobot.obs_noise_perturbation_cpg \
        --plan-horizon 15 \
        --output results/dmc_acrobot/obs_noise_perturbation_cpg_h15.json

Writes (default, when --output is omitted):
    results/dmc_acrobot/obs_noise_perturbation_cpg.json
"""

from __future__ import annotations

import argparse
import itertools
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
from wmel.envs.dmc_acrobot import (
    DEFAULT_DISCRETE_LEVELS,
    DMCAcrobotEnv,
    make_acrobot_oracle_dynamics,
)
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
JSON_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "obs_noise_perturbation_cpg.json"

DEFAULT_SIGMAS = (0.0, 0.01, 0.05, 0.1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Tiny config, 2 eps per arm, random data (no TD-MPC2 ckpt needed).")
    p.add_argument("--sigmas", default=",".join(str(s) for s in DEFAULT_SIGMAS),
                   help="Comma-separated list of obs-noise stdev values. Default 0.0,0.01,0.05,0.1.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the planner / MLP / env init.")
    p.add_argument("--episodes", type=int, default=50, help="Episodes per (sigma, arm) cell.")
    p.add_argument("--n-mlp-transitions", type=int, default=20_000)
    p.add_argument("--plan-horizon", type=int, default=None,
                   help="Override CEM plan_horizon. Default 15 (smoke 8). Use 1 to give the "
                        "TD-MPC2 arm headroom (it scores ~0.89 at H=1 in phase-5o), which is "
                        "the regime where differential obs-noise fragility can actually show.")
    p.add_argument("--output", default=None, help="Output JSON path (default results/dmc_acrobot/obs_noise_perturbation_cpg.json).")
    return p.parse_args()


def _smoke_cfg() -> dict:
    return {
        "n_mlp_transitions": 200,
        "mlp_epochs": 20,
        "cem_iters": 2,
        "cem_samples": 8,
        "cem_elites": 3,
        "plan_horizon": 8,
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
        "plan_horizon": 15,
        "smoothing": 0.1,
        "benchmark_episodes": episodes,
        "benchmark_horizon": 500,
    }


def _run_cem_arm(*, name: str, dynamics, cfg: dict, sigma: float, seed: int, levels):
    # IMPORTANT: BenchmarkRunner calls env_factory() once per episode. We must
    # therefore give each episode a DISTINCT obs-noise seed, otherwise all
    # `benchmark_episodes` episodes in a cell replay the same noise
    # realization and the cell measures n=1 in noise-space (point estimate
    # tied to a single noise draw, with planner-sample variance only). A
    # per-episode counter derived from (seed, sigma) gives reproducible but
    # distinct noise realizations, so the success rate averages over the
    # noise distribution as intended.
    base = int(seed * 1_000_000 + int(round(sigma * 1000)) * 1000)
    _noise_counter = itertools.count(base)
    env_factory = lambda: DMCAcrobotEnv(
        discrete_levels=levels,
        obs_noise_std=sigma,
        obs_noise_seed=next(_noise_counter),
    )
    env_template = env_factory()
    planner = BatchedCEMPlanner(
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
        policy_name=f"cem sigma={sigma:g} ({name})",
        compute_per_plan_call=planner.compute_per_plan_call,
        perturbation_name=f"obs_noise(sigma={sigma:g})",
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


def main() -> None:
    args = _parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    cfg = _smoke_cfg() if args.smoke else _full_cfg(args.n_mlp_transitions, args.episodes)
    if args.plan_horizon is not None:
        cfg["plan_horizon"] = args.plan_horizon
    levels = DEFAULT_DISCRETE_LEVELS
    seed = args.seed

    use_tdmpc2_data = (not args.smoke) and TDMPC2_AGENT_PATH.exists() and TDMPC2_DYNAMICS_PATH.exists()
    if (not args.smoke) and not use_tdmpc2_data:
        raise FileNotFoundError(
            f"Missing {TDMPC2_AGENT_PATH} or {TDMPC2_DYNAMICS_PATH}; run experiments.dmc_acrobot.tdmpc2_cpg first or pass --smoke."
        )

    # 1. Collect MLP data + train MLP, using a CLEAN env (sigma=0) so the
    #    MLP fits the canonical observation manifold. The obs_noise hook
    #    applies only at planner-time, not at MLP training-time. This is the
    #    intended setup: the MLP is "deployed" against a noisy observation
    #    distribution; if its capacity is brittle to distribution shift, the
    #    CPG should widen with sigma.
    if use_tdmpc2_data:
        agent = _load_tdmpc2_agent(TDMPC2_AGENT_PATH, seed=seed)
        print(f"[data] Collecting {cfg['n_mlp_transitions']} transitions from TD-MPC2 eval policy (sigma=0)...")
        transitions = _collect_tdmpc2_rollouts(
            agent=agent,
            env_factory=lambda: DMCAcrobotEnv(discrete_levels=levels),
            n_transitions=cfg["n_mlp_transitions"],
            levels=levels,
            seed=seed,
        )
    else:
        n_eps = max(1, cfg["n_mlp_transitions"] // 200)
        print(f"[data] Collecting {cfg['n_mlp_transitions']} random transitions ({n_eps} eps)...")
        transitions = collect_random_rollouts(
            lambda: DMCAcrobotEnv(discrete_levels=levels),
            n_episodes=n_eps,
            max_steps_per_episode=200,
            seed=seed,
        )
        transitions = transitions[: cfg["n_mlp_transitions"]]

    coverage = _coverage_stats(transitions, levels)
    print(f"[coverage] u_max={coverage['uprightness_max']:.3f}  frac(u>1)={coverage['fraction_u_gt_1']:.3f}")

    print(f"[mlp] Training MLP ({cfg['mlp_epochs']} epochs)...")
    env_template = DMCAcrobotEnv(discrete_levels=levels)
    mlp_model, mlp_log = train_world_model(
        transitions,
        obs_dim=6,
        n_actions=len(env_template.action_space),
        epochs=cfg["mlp_epochs"],
        seed=seed,
    )
    print(f"      val_mse={mlp_log['final_val_mse']:.6f}")

    # 2. Build batched dynamics callables once. These are sigma-independent;
    #    sigma only changes the env that BenchmarkRunner steps through.
    oracle_dyn = make_oracle_batched_dynamics(make_acrobot_oracle_dynamics())
    mlp_dyn = make_mlp_batched_dynamics(mlp_model, env_template.action_space)
    if use_tdmpc2_data:
        tdmpc2_dyn = make_tdmpc2_batched_dynamics(TDMPC2_DYNAMICS_PATH, device="cpu")
    else:
        tdmpc2_dyn = None

    # 3. Per-sigma cell.
    cells: list[dict] = []
    for sigma in sigmas:
        print(f"\n========== sigma = {sigma:g} ==========")
        t0 = time.time()
        oracle_results, oracle_card = _run_cem_arm(
            name="oracle dynamics", dynamics=oracle_dyn,
            cfg=cfg, sigma=sigma, seed=seed, levels=levels,
        )
        mlp_results, mlp_card = _run_cem_arm(
            name=f"MLP on {'tdmpc2' if use_tdmpc2_data else 'random'} data",
            dynamics=mlp_dyn,
            cfg=cfg, sigma=sigma, seed=seed, levels=levels,
        )
        if tdmpc2_dyn is not None:
            tdmpc2_results, tdmpc2_card = _run_cem_arm(
                name="TD-MPC2 dynamics",
                dynamics=tdmpc2_dyn,
                cfg=cfg, sigma=sigma, seed=seed, levels=levels,
            )
        else:
            tdmpc2_results, tdmpc2_card = None, None
        dt = time.time() - t0
        print(f"[sigma {sigma:g}] elapsed {dt:.1f}s")

        cpgs: dict[str, dict] = {}
        for name, results in [
            ("mlp_on_data", mlp_results),
            ("tdmpc2", tdmpc2_results),
        ]:
            if results is None:
                continue
            cpg = counterfactual_planning_gap(oracle_results, results)
            v = cpg_verdict(cpg)
            cpgs[name] = {**asdict(cpg), "verdict": v}
            print(f"  sigma={sigma:g} oracle vs {name:>15s}: oracle={cpg.oracle_success_rate:.3f}  learned={cpg.learned_success_rate:.3f}  "
                  f"CPG={cpg.gap:+.3f}  CI [{cpg.gap_ci_low:+.3f}, {cpg.gap_ci_high:+.3f}]  "
                  f"{'SMOKE' if args.smoke else v}")

        cells.append({
            "sigma": sigma,
            "n_per_arm": len(oracle_results),
            "cpgs": cpgs,
            "oracle_scorecard": _card_to_dict(oracle_card),
            "mlp_scorecard": _card_to_dict(mlp_card),
            "tdmpc2_scorecard": _card_to_dict(tdmpc2_card),
            "oracle_full": to_json_report(oracle_results, oracle_card),
            "mlp_full": to_json_report(mlp_results, mlp_card),
            "tdmpc2_full": to_json_report(tdmpc2_results, tdmpc2_card) if tdmpc2_results else None,
        })

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "metric": "counterfactual_planning_gap",
        "planner": "cem",
        "sweep_axis": "obs_noise_std",
        "plan_horizon": cfg["plan_horizon"],
        "mlp_data_source": "tdmpc2" if use_tdmpc2_data else "random",
        "mlp_trained_at_sigma": 0.0,
        "tdmpc2_agent_seed_held_fixed": use_tdmpc2_data,
        "seed": seed,
        "sigmas": sigmas,
        "episodes_per_cell": cfg["benchmark_episodes"],
        "smoke_mode": args.smoke,
        "cfg": cfg,
        "mlp_training": mlp_log,
        "coverage": coverage,
        "cells": cells,
    }
    out_path = Path(args.output) if args.output else JSON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
