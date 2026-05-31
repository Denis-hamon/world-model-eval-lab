"""Multi-seed pooled-150 CEM CPG on the three v0.12 dynamics arms.

Action (3) of the post-v0.12 roadmap. At $n = 10$ under CEM (phase-5f,
PR #5) the verdict was already \textsc{model bottleneck} with CI
$[+0.49, +1.01]$ on a +0.900 gap; this experiment pools 3 seeds at 50
episodes per seed per arm to mirror the v0.11 sweep protocol and either
hardens the CI further or surfaces seed-level variability.

Honest scoping
--------------
Three independent variables vary across seeds 0, 1, 2:
  - the benchmark env init (DMCAcrobotEnv reset seed),
  - the CEM sampler (per-timestep categorical sampling),
  - the MLP retraining (initialisation, batch sampling, and the
    TD-MPC2 rollout subset gathered with this seed's env init).

One variable is *held fixed* across seeds: the trained TD-MPC2 agent
itself, loaded from the single seed-0 checkpoint at
`results/dmc_acrobot/tdmpc2_agent.pt`. Retraining TD-MPC2 from seed 1
and seed 2 would cost roughly $2 \times 30$~h on an RTX 5000 for a
marginal benefit; at the $n = 10$ point CEM already returned a decisive
verdict, so the marginal value of also varying the agent's training
seed is small. The JSON output flags this as
`tdmpc2_agent_seed_held_fixed: true` so a future tightening can be done
without ambiguity about what was already varied.

Usage
-----
    python -m experiments.dmc_acrobot.cem_cpg_sweep --smoke
    python -m experiments.dmc_acrobot.cem_cpg_sweep                # default: seeds 0,1,2
    python -m experiments.dmc_acrobot.cem_cpg_sweep --seeds 0,1,2,3
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
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report

from experiments._seeding import eval_varied_factory, train_varied_factory

from experiments.dmc_acrobot.coverage_mlp_on_tdmpc2 import (
    _collect_tdmpc2_rollouts,
    _coverage_stats,
    _load_tdmpc2_agent,
)


TDMPC2_AGENT_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "tdmpc2_agent.pt"
TDMPC2_DYNAMICS_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "tdmpc2_acrobot.pt"
JSON_PATH = _REPO_ROOT / "results" / "dmc_acrobot" / "cem_cpg_sweep.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="2 seeds, 2 eps per arm, random data; validates wiring.")
    p.add_argument("--seeds", default="0,1,2", help="Comma-separated list of seeds (default 0,1,2).")
    p.add_argument("--n-mlp-transitions", type=int, default=20_000)
    p.add_argument(
        "--varied-init",
        action="store_true",
        help=(
            "Vary the initial state per episode (seed shared across arms, "
            "training drawn from a disjoint block) so success rates sample the "
            "task distribution. Off by default to reproduce committed results."
        ),
    )
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
            "benchmark_episodes": 2,
            "benchmark_horizon": 80,
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
    }


def _run_cem_arm(*, name: str, dynamics, cfg: dict, seed: int, levels, varied_init: bool = False):
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
    # All arms at this seed share base_seed=seed, so episode k starts from the
    # same state in every arm (paired). Off by default to reproduce committed
    # results.
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


def _run_one_seed(*, seed: int, cfg: dict, levels, use_tdmpc2_data: bool, agent_loaded, varied_init: bool = False):
    """Run all three CEM arms for a single seed and return the per-arm results.

    `agent_loaded` is a callable that, when called, returns a TDMPC2 agent
    already loaded from `tdmpc2_agent.pt`. We pass it in so we only pay the
    agent-load cost once across all seeds; the per-seed variance from the
    agent's stochastic policy at eval-mode is zero (TD-MPC2 eval-mode is
    deterministic), so the cross-seed variation enters only through the env
    init / sampler.
    """
    if use_tdmpc2_data:
        agent = agent_loaded()
        print(f"[seed {seed}] Collecting {cfg['n_mlp_transitions']} transitions from TD-MPC2 eval policy...")
        transitions = _collect_tdmpc2_rollouts(
            agent=agent,
            env_factory=(
                train_varied_factory(DMCAcrobotEnv, seed, discrete_levels=levels)
                if varied_init
                else (lambda: DMCAcrobotEnv(discrete_levels=levels))
            ),
            n_transitions=cfg["n_mlp_transitions"],
            levels=levels,
            seed=seed,
        )
    else:
        n_eps = max(1, cfg["n_mlp_transitions"] // 200)
        print(f"[seed {seed}] Collecting {cfg['n_mlp_transitions']} random transitions ({n_eps} eps)...")
        transitions = collect_random_rollouts(
            train_varied_factory(DMCAcrobotEnv, seed, discrete_levels=levels)
            if varied_init
            else (lambda: DMCAcrobotEnv(discrete_levels=levels)),
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

    print(f"\n[seed {seed}, arm 1/3] CEM on ORACLE dynamics")
    oracle_results, oracle_card = _run_cem_arm(
        name="oracle dynamics", dynamics=make_acrobot_oracle_dynamics(), cfg=cfg, seed=seed, levels=levels, varied_init=varied_init,
    )
    print(f"\n[seed {seed}, arm 2/3] CEM on MLP-on-{'tdmpc2' if use_tdmpc2_data else 'random'} dynamics")
    mlp_results, mlp_card = _run_cem_arm(
        name=f"MLP on {'tdmpc2' if use_tdmpc2_data else 'random'} data",
        dynamics=learned_dynamics(mlp_model, env_template.action_space),
        cfg=cfg, seed=seed, levels=levels, varied_init=varied_init,
    )

    if TDMPC2_DYNAMICS_PATH.exists():
        print(f"\n[seed {seed}, arm 3/3] CEM on TD-MPC2 dynamics")
        tdmpc2_results, tdmpc2_card = _run_cem_arm(
            name="TD-MPC2 dynamics",
            dynamics=make_tdmpc2_dynamics(TDMPC2_DYNAMICS_PATH, device="cpu"),
            cfg=cfg, seed=seed, levels=levels, varied_init=varied_init,
        )
    else:
        tdmpc2_results, tdmpc2_card = None, None
        print(f"\n[seed {seed}, arm 3/3] TD-MPC2 dynamics skipped (no checkpoint).")

    return {
        "seed": seed,
        "coverage": coverage,
        "mlp_training": mlp_log,
        "oracle_results": oracle_results,
        "mlp_results": mlp_results,
        "tdmpc2_results": tdmpc2_results,
        "oracle_card": oracle_card,
        "mlp_card": mlp_card,
        "tdmpc2_card": tdmpc2_card,
    }


def _pool_episodes(per_seed_outputs, key: str):
    """Concatenate per-seed episode lists into a single pooled list."""
    pooled = []
    for out in per_seed_outputs:
        results = out[key]
        if results is None:
            return None
        pooled.extend(results)
    return pooled


def _card_to_dict(card) -> dict:
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
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = _config(smoke=args.smoke, n_mlp_transitions=args.n_mlp_transitions)
    levels = DEFAULT_DISCRETE_LEVELS

    use_tdmpc2_data = (not args.smoke) and TDMPC2_AGENT_PATH.exists() and TDMPC2_DYNAMICS_PATH.exists()
    if (not args.smoke) and not use_tdmpc2_data:
        raise FileNotFoundError(
            f"Missing {TDMPC2_AGENT_PATH} or {TDMPC2_DYNAMICS_PATH}; run experiments.dmc_acrobot.tdmpc2_cpg first or pass --smoke."
        )

    # Load the TD-MPC2 agent ONCE and share across seeds. The per-seed
    # variation through eval-mode rollouts comes from the env init (seed),
    # not from the agent's policy (deterministic at eval).
    _agent_cache: dict = {}
    def agent_loaded():
        if "agent" not in _agent_cache:
            print(f"[setup] Loading TD-MPC2 agent from {TDMPC2_AGENT_PATH.name} (held fixed across seeds)...")
            _agent_cache["agent"] = _load_tdmpc2_agent(TDMPC2_AGENT_PATH, seed=seeds[0])
        return _agent_cache["agent"]

    print(f"[setup] seeds={seeds}, episodes_per_seed_per_arm={cfg['benchmark_episodes']}, "
          f"smoke={args.smoke}, use_tdmpc2_data={use_tdmpc2_data}")

    per_seed: list = []
    for s in seeds:
        print(f"\n========== SEED {s} ==========")
        out = _run_one_seed(seed=s, cfg=cfg, levels=levels, use_tdmpc2_data=use_tdmpc2_data, agent_loaded=agent_loaded, varied_init=args.varied_init)
        per_seed.append(out)

    # Pool episodes across seeds, recompute the scorecards on the pool,
    # then compute pooled CPGs.
    pooled_oracle = _pool_episodes(per_seed, "oracle_results")
    pooled_mlp = _pool_episodes(per_seed, "mlp_results")
    pooled_tdmpc2 = _pool_episodes(per_seed, "tdmpc2_results")

    print("\n========== POOLED ==========")
    print(f"Pooled n per arm = {len(pooled_oracle)}")

    oracle_pooled_card = compute_scorecard(
        pooled_oracle,
        policy_name="cem (oracle dynamics) pooled",
        compute_per_plan_call=per_seed[0]["oracle_card"].average_compute_per_decision,
        perturbation_name="env-default",
    )
    print_scorecard(oracle_pooled_card)

    mlp_pooled_card = compute_scorecard(
        pooled_mlp,
        policy_name=f"cem (MLP on {'tdmpc2' if use_tdmpc2_data else 'random'} data) pooled",
        compute_per_plan_call=per_seed[0]["mlp_card"].average_compute_per_decision,
        perturbation_name="env-default",
    )
    print_scorecard(mlp_pooled_card)

    if pooled_tdmpc2 is not None:
        tdmpc2_pooled_card = compute_scorecard(
            pooled_tdmpc2,
            policy_name="cem (TD-MPC2 dynamics) pooled",
            compute_per_plan_call=per_seed[0]["tdmpc2_card"].average_compute_per_decision,
            perturbation_name="env-default",
        )
        print_scorecard(tdmpc2_pooled_card)
    else:
        tdmpc2_pooled_card = None

    print("\n========== CPG (pooled vs oracle) ==========")
    cpgs = {}
    for name, results in [("mlp_on_data", pooled_mlp), ("tdmpc2", pooled_tdmpc2)]:
        if results is None:
            continue
        cpg = counterfactual_planning_gap(pooled_oracle, results)
        v = cpg_verdict(cpg)
        cpgs[name] = {**asdict(cpg), "verdict": v}
        print(f"  oracle vs {name:>15s}: oracle={cpg.oracle_success_rate:.3f}  learned={cpg.learned_success_rate:.3f}  "
              f"CPG={cpg.gap:+.3f}  CI [{cpg.gap_ci_low:+.3f}, {cpg.gap_ci_high:+.3f}]  "
              f"{'SMOKE' if args.smoke else v}")

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "metric": "counterfactual_planning_gap",
        "planner": "cem",
        "mlp_data_source": "tdmpc2" if use_tdmpc2_data else "random",
        "tdmpc2_agent_seed_held_fixed": use_tdmpc2_data,
        "seeds": seeds,
        "pooled_n_per_arm": len(pooled_oracle),
        "cpgs": cpgs,
        "config": cfg,
        "smoke_mode": args.smoke,
        "varied_init": args.varied_init,
        "per_seed": [
            {
                "seed": out["seed"],
                "coverage": out["coverage"],
                "mlp_training": out["mlp_training"],
                "oracle_scorecard": _card_to_dict(out["oracle_card"]),
                "mlp_scorecard": _card_to_dict(out["mlp_card"]),
                "tdmpc2_scorecard": _card_to_dict(out["tdmpc2_card"]),
            }
            for out in per_seed
        ],
        "pooled_oracle_scorecard": _card_to_dict(oracle_pooled_card),
        "pooled_mlp_scorecard": _card_to_dict(mlp_pooled_card),
        "pooled_tdmpc2_scorecard": _card_to_dict(tdmpc2_pooled_card),
        "oracle_full": to_json_report(pooled_oracle, oracle_pooled_card),
        "mlp_full": to_json_report(pooled_mlp, mlp_pooled_card),
        "tdmpc2_full": to_json_report(pooled_tdmpc2, tdmpc2_pooled_card) if pooled_tdmpc2 else None,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {JSON_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
