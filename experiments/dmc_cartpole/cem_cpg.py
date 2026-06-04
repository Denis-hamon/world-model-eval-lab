"""CEM planner on the three v0.12 dynamics arms (oracle / TD-MPC2 / MLP-on-TD-MPC2).

The v0.12 result (PR #3, merged) returned three identical CPG numbers
(+0.300 INCONCLUSIVE at n=10) for three very different learned dynamics.
That confound between *dynamics quality* and *planner capacity* is what
this experiment is built to resolve: swap the random-shooting MPC for
CEM (`wmel.adapters.cem_planner.CEMPlanner`) and rerun the same
BenchmarkRunner / oracle / score / seed protocol on each arm.

The CEM budget here is intentionally close to the random-shoot budget
the v0.12 numbers were measured against:
  CEM:          num_iterations * num_samples * plan_horizon = 3 * 24 * 15 = 1 080 evals/plan
  random-shoot: num_candidates * plan_horizon              =     50 * 15 =   750 evals/plan
Same compute order of magnitude. Any verdict change is attributable to
the search strategy, not to a 100x compute uplift.

Reading the result:
  - If oracle+CEM jumps from 0.30 well above 0.30, random-shoot was leaving
    upside on the table even with perfect dynamics. Planner is *a*
    contributor; we cannot say it is the *primary* one yet.
  - If oracle+CEM stays near 0.30 and learned+CEM also stays at 0.00, the
    random-shoot ceiling is not artificially low. Planner is not the
    primary bottleneck; dynamics quality (or score-function mismatch) is.
  - If oracle+CEM stays near 0.30 but learned+CEM lifts to non-zero, CEM
    *exploits* learned dynamics that random-shoot did not. The dynamics
    was usable; the random-shoot was wasting it.

Usage
-----
    python -m experiments.dmc_cartpole.cem_cpg --smoke
    python -m experiments.dmc_cartpole.cem_cpg                  # uses trained TD-MPC2 at results/dmc_cartpole/tdmpc2_*.pt
    python -m experiments.dmc_cartpole.cem_cpg --seed 1

Writes:
    results/dmc_cartpole/cem_cpg.json
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
    collect_random_rollouts,
    learned_dynamics,
    train_world_model,
)
from wmel.adapters.tdmpc2_adapter import make_tdmpc2_dynamics
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_cartpole import (
    DEFAULT_DISCRETE_LEVELS,
    DMCCartpoleEnv,
    cartpole_upright_score,
    make_cartpole_oracle_dynamics,
)
from wmel.metrics import compute_scorecard, counterfactual_planning_gap, cpg_verdict
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report

from experiments._seeding import eval_varied_factory, train_varied_factory

# Reuse the TD-MPC2 agent loader and rollout collector from the v0.12
# coverage experiment to avoid duplicating ~50 lines of cfg / monkey-patch
# boilerplate. These are not part of the wmel public API; treat as
# experiment-internal helpers.
from experiments.dmc_cartpole.coverage_mlp_on_tdmpc2 import (
    _collect_tdmpc2_rollouts,
    _coverage_stats,
    _load_tdmpc2_agent,
)


TDMPC2_AGENT_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / "tdmpc2_agent.pt"
TDMPC2_DYNAMICS_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / "tdmpc2_cartpole.pt"
JSON_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / "cem_cpg.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Tiny CEM config, random data (no TD-MPC2 ckpt needed).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-size", type=int, default=1, help="TD-MPC2 size preset for the loaded agent / dynamics.")
    p.add_argument("--mlp-data-source", choices=["tdmpc2", "random"], default="tdmpc2",
                   help="Where to source MLP training data (mirrors coverage_mlp_on_tdmpc2).")
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
    p.add_argument("--out-suffix", default="", help="Extra suffix appended to the output result-JSON filename (e.g. _fixedinit) so a re-run does not overwrite existing results. Does NOT affect checkpoint paths.")
    p.add_argument("--episodes", type=int, default=None, help="Override benchmark episodes per arm per seed (default 10). Use 50 with three seeds + pool_cpg for an n=150 pooled estimate on the existing checkpoints.")
    return p.parse_args()


def _output_suffix(model_size: int, seed: int) -> str:
    if model_size == 1 and seed == 0:
        return ""
    if model_size == 1:
        return f"_seed{seed}"
    return f"_size{model_size}_seed{seed}"


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
        "benchmark_episodes": 10,
        "benchmark_horizon": 500,
    }


def _run_arm(
    *,
    name: str,
    dynamics_factory,
    cfg: dict,
    seed: int,
    levels: tuple[float, ...],
    varied_init: bool = False,
):
    env_template = DMCCartpoleEnv(discrete_levels=levels)
    planner = CEMPlanner(
        dynamics=dynamics_factory(),
        action_space=env_template.action_space,
        num_iterations=cfg["cem_iters"],
        num_samples=cfg["cem_samples"],
        num_elites=cfg["cem_elites"],
        plan_horizon=cfg["plan_horizon"],
        smoothing=cfg["smoothing"],
        score=cartpole_upright_score,
        seed=seed,
    )
    # Both arms share base_seed=seed, so episode k starts from the same state
    # in every arm (paired). Off by default to reproduce committed results.
    env_factory = (
        eval_varied_factory(DMCCartpoleEnv, seed, discrete_levels=levels)
        if varied_init
        else (lambda: DMCCartpoleEnv(discrete_levels=levels))
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


def main() -> None:
    args = _parse_args()
    cfg = _config(smoke=args.smoke, n_mlp_transitions=args.n_mlp_transitions)
    if args.episodes is not None:
        cfg["benchmark_episodes"] = args.episodes
    seed = args.seed
    model_size = args.model_size
    levels = DEFAULT_DISCRETE_LEVELS
    suffix = _output_suffix(model_size, seed)
    global TDMPC2_AGENT_PATH, TDMPC2_DYNAMICS_PATH, JSON_PATH
    TDMPC2_AGENT_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / f"tdmpc2_agent{suffix}.pt"
    TDMPC2_DYNAMICS_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / f"tdmpc2_cartpole{suffix}.pt"
    JSON_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / f"cem_cpg{suffix}{args.out_suffix}.json"

    use_tdmpc2 = (args.mlp_data_source == "tdmpc2") and not args.smoke

    # 1. MLP-on-{tdmpc2|random} dynamics: collect data, train MLP.
    if use_tdmpc2:
        if not TDMPC2_AGENT_PATH.exists():
            raise FileNotFoundError(
                f"TD-MPC2 agent checkpoint not found at {TDMPC2_AGENT_PATH}. "
                "Run experiments.dmc_cartpole.tdmpc2_cpg first or pass --mlp-data-source random."
            )
        if not TDMPC2_DYNAMICS_PATH.exists():
            raise FileNotFoundError(
                f"TD-MPC2 dynamics checkpoint not found at {TDMPC2_DYNAMICS_PATH}."
            )
        agent = _load_tdmpc2_agent(TDMPC2_AGENT_PATH, seed=seed, model_size=model_size)
        print(f"[data] Collecting {cfg['n_mlp_transitions']} transitions from TD-MPC2 eval policy...")
        transitions = _collect_tdmpc2_rollouts(
            agent=agent,
            env_factory=(
                train_varied_factory(DMCCartpoleEnv, seed, discrete_levels=levels)
                if args.varied_init
                else (lambda: DMCCartpoleEnv(discrete_levels=levels))
            ),
            n_transitions=cfg["n_mlp_transitions"],
            levels=levels,
            seed=seed,
        )
    else:
        n_eps = max(1, cfg["n_mlp_transitions"] // 200)
        print(f"[data] Collecting {cfg['n_mlp_transitions']} random-policy transitions ({n_eps} eps x 200 steps)...")
        transitions = collect_random_rollouts(
            train_varied_factory(DMCCartpoleEnv, seed, discrete_levels=levels)
            if args.varied_init
            else (lambda: DMCCartpoleEnv(discrete_levels=levels)),
            n_episodes=n_eps,
            max_steps_per_episode=200,
            seed=seed,
        )
        transitions = transitions[: cfg["n_mlp_transitions"]]

    coverage = _coverage_stats(transitions, levels)
    print(f"[coverage] n={coverage['n_transitions']}  u_max={coverage['uprightness_max']:.3f}  "
          f"frac(u>1)={coverage['fraction_u_gt_1']:.3f}")

    print(f"[mlp] Training MLP world model ({cfg['mlp_epochs']} epochs)...")
    env_template = DMCCartpoleEnv(discrete_levels=levels)
    mlp_model, mlp_log = train_world_model(
        transitions,
        obs_dim=5,
        n_actions=len(env_template.action_space),
        epochs=cfg["mlp_epochs"],
        seed=seed,
    )
    print(f"      val_mse={mlp_log['final_val_mse']:.6f}")

    # 2. Build the three dynamics callables.
    def make_oracle():
        return make_cartpole_oracle_dynamics()

    def make_mlp():
        return learned_dynamics(mlp_model, env_template.action_space)

    if (not args.smoke) and TDMPC2_DYNAMICS_PATH.exists():
        def make_tdmpc2():
            return make_tdmpc2_dynamics(TDMPC2_DYNAMICS_PATH, device="cpu")
        tdmpc2_available = True
    else:
        tdmpc2_available = False

    # 3. Run all available arms with the SAME CEM config and the same seed.
    print("\n[1/3] CEM with ORACLE dynamics...")
    oracle_results, oracle_card = _run_arm(
        name="oracle dynamics", dynamics_factory=make_oracle, cfg=cfg, seed=seed, levels=levels, varied_init=args.varied_init,
    )

    print(f"\n[2/3] CEM with MLP-on-{'tdmpc2' if use_tdmpc2 else 'random'} dynamics...")
    mlp_results, mlp_card = _run_arm(
        name=f"MLP on {'tdmpc2' if use_tdmpc2 else 'random'} data",
        dynamics_factory=make_mlp, cfg=cfg, seed=seed, levels=levels, varied_init=args.varied_init,
    )

    if tdmpc2_available:
        print("\n[3/3] CEM with TD-MPC2 dynamics...")
        tdmpc2_results, tdmpc2_card = _run_arm(
            name="TD-MPC2 dynamics", dynamics_factory=make_tdmpc2, cfg=cfg, seed=seed, levels=levels, varied_init=args.varied_init,
        )
    else:
        tdmpc2_results, tdmpc2_card = None, None
        print("\n[3/3] TD-MPC2 dynamics arm skipped (no checkpoint or smoke mode).")

    # 4. Compute CPGs vs oracle and report.
    print("\n[CPG] vs oracle:")
    cpgs = {}
    for name, results in [
        ("mlp_on_data", mlp_results),
        ("tdmpc2", tdmpc2_results),
    ]:
        if results is None:
            continue
        cpg = counterfactual_planning_gap(oracle_results, results)
        v = cpg_verdict(cpg)
        cpgs[name] = {**asdict(cpg), "verdict": v}
        print(f"  oracle vs {name:>15s}: oracle={cpg.oracle_success_rate:.3f}  learned={cpg.learned_success_rate:.3f}  "
              f"CPG={cpg.gap:+.3f}  CI [{cpg.gap_ci_low:+.3f}, {cpg.gap_ci_high:+.3f}]  "
              f"{'SMOKE' if args.smoke else v}")

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_cartpole_swingup",
        "metric": "counterfactual_planning_gap",
        "planner": "cem",
        "mlp_data_source": "tdmpc2" if use_tdmpc2 else "random",
        "cpgs": cpgs,
        "config": cfg,
        "mlp_training": mlp_log,
        "coverage": coverage,
        "seed": seed,
        "smoke_mode": args.smoke,
        "varied_init": args.varied_init,
        "oracle_scorecard": _card_to_dict(oracle_card),
        "mlp_scorecard": _card_to_dict(mlp_card),
        "tdmpc2_scorecard": _card_to_dict(tdmpc2_card) if tdmpc2_card else None,
        "oracle_full": to_json_report(oracle_results, oracle_card),
        "mlp_full": to_json_report(mlp_results, mlp_card),
        "tdmpc2_full": to_json_report(tdmpc2_results, tdmpc2_card) if tdmpc2_results else None,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {JSON_PATH.relative_to(_REPO_ROOT)}")


def _card_to_dict(card) -> dict:
    return {
        "policy_name": card.policy_name,
        "success_rate": card.success_rate,
        "average_steps_to_success": card.average_steps_to_success,
        "average_planning_latency_ms": card.average_planning_latency_ms,
        "average_compute_per_decision": card.average_compute_per_decision,
        "episodes": card.episodes,
    }


if __name__ == "__main__":
    main()
