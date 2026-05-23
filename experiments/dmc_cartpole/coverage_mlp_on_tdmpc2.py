"""Coverage hypothesis test: v0.11's MLP, trained on TD-MPC2 on-policy data.

The v0.11 paper argues that the learned arm's planning failure is a
*coverage* bottleneck (random-policy rollouts never visit the upright
regime), not a *capacity* bottleneck (at 20 000 transitions the MLP fits
the training distribution to ~4e-4 val MSE). This experiment is the
direct test:

- The architecture, optimiser, training schedule, planner, score function,
  and env are *exactly* v0.11's MLP cell at 20 000 transitions.
- The only change is the *data-collection policy*: instead of a uniform
  random torque, we use a trained TD-MPC2 agent (eval-mode actions snapped
  to the discrete torque set) to gather the same number of transitions.
- We then run the same CPG protocol and compare to the v0.11 paper's
  Table 2 row at 20 000 transitions (CPG = +0.267, verdict
  MODEL BOTTLENECK, learned success = 0/150).

Reading the result:

- If the learned arm's success rate goes from 0 to non-zero while
  everything except the data source is held fixed, the coverage hypothesis
  is confirmed: the model had ample capacity all along, it just needed
  data from the relevant regime.
- If the learned arm stays at 0 despite a measurably-shifted coverage
  axis, the coverage diagnosis is too strong; second-order contributors
  (planner capacity, score mismatch) deserve more weight.

Usage
-----
    ./scripts/setup_tdmpc2.sh
    python -m experiments.dmc_cartpole.coverage_mlp_on_tdmpc2 --smoke           # 2 eps, validates wiring (no TD-MPC2 ckpt needed)
    python -m experiments.dmc_cartpole.coverage_mlp_on_tdmpc2                   # 20k transitions from results/dmc_cartpole/tdmpc2_agent.pt
    python -m experiments.dmc_cartpole.coverage_mlp_on_tdmpc2 --data-source random  # control: rerun v0.11's setup on this code path

Writes:
    results/dmc_cartpole/coverage_mlp_on_tdmpc2_cpg.json
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

import numpy as np
import torch

from wmel.adapters.mlp_world_model import (
    MLPWorldModel,
    collect_random_rollouts,
    learned_dynamics,
    train_world_model,
)
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_cartpole import (
    DEFAULT_DISCRETE_LEVELS,
    DMCCartpoleEnv,
    cartpole_upright_score,
    make_cartpole_oracle_dynamics,
)
from wmel.metrics import compute_scorecard, counterfactual_planning_gap, cpg_verdict
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report


AGENT_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / "tdmpc2_agent.pt"
JSON_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / "coverage_mlp_on_tdmpc2_cpg.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="2 eps random data, tiny CPG. Validates wiring only.")
    p.add_argument("--data-source", choices=["tdmpc2", "random"], default="tdmpc2", help="Data-collection policy.")
    p.add_argument("--n-transitions", type=int, default=20_000, help="Target number of transitions to collect.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-size", type=int, default=1, help="TD-MPC2 size preset used by the loaded agent.")
    p.add_argument("--agent-ckpt", default=str(AGENT_PATH), help="Path to a tdmpc2 agent checkpoint (used iff --data-source=tdmpc2).")
    return p.parse_args()


def _output_suffix(model_size: int, seed: int) -> str:
    if model_size == 1 and seed == 0:
        return ""
    if model_size == 1:
        return f"_seed{seed}"
    return f"_size{model_size}_seed{seed}"


def _config(smoke: bool, n_transitions: int) -> dict:
    if smoke:
        return {
            "n_transitions": 200,
            "epochs": 20,
            "num_candidates": 15,
            "plan_horizon": 8,
            "benchmark_episodes": 2,
            "benchmark_horizon": 80,
        }
    return {
        "n_transitions": n_transitions,
        "epochs": 200,
        "num_candidates": 50,
        "plan_horizon": 15,
        "benchmark_episodes": 10,
        "benchmark_horizon": 500,
    }


def _snap_to_levels(value: float, levels: tuple[float, ...]) -> tuple[int, float]:
    """Return (index, level) of the discrete torque closest to `value`."""
    arr = np.asarray(levels)
    idx = int(np.argmin(np.abs(arr - value)))
    return idx, float(arr[idx])


def _coverage_stats(transitions, levels: tuple[float, ...]) -> dict:
    obs = np.asarray([t[0] for t in transitions], dtype=np.float32)
    # state = (sin_u, sin_l, cos_u, cos_l, v_s, v_e); u = cos_u + cos_l
    u = obs[:, 2] + obs[:, 3]
    action_idx = np.asarray([t[1] for t in transitions], dtype=np.int64)
    counts = np.bincount(action_idx, minlength=len(levels)).tolist()
    return {
        "n_transitions": int(len(transitions)),
        "uprightness_max": float(u.max()),
        "uprightness_p95": float(np.percentile(u, 95)),
        "uprightness_mean": float(u.mean()),
        "fraction_u_gt_1": float((u > 1.0).mean()),
        "fraction_u_gt_1_5": float((u > 1.5).mean()),
        "action_counts": counts,
    }


def _collect_tdmpc2_rollouts(agent, env_factory, n_transitions: int, levels: tuple[float, ...], seed: int) -> list:
    """Run eval-mode TD-MPC2 episodes against wmel's DMCCartpoleEnv, snapping
    continuous actions to the discrete torque set. Returns v0.11-compatible
    transitions: list of (obs_tuple, action_idx, next_obs_tuple).
    """
    transitions: list = []
    ep = 0
    while len(transitions) < n_transitions:
        env = env_factory()
        obs = env.reset()
        t = 0
        while t < 500 and len(transitions) < n_transitions:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device)
            with torch.no_grad():
                a_cont = agent.act(obs_t, t0=(t == 0), eval_mode=True)
            a_value = float(a_cont.detach().cpu().flatten()[0])
            a_idx, a_level = _snap_to_levels(a_value, levels)
            next_obs = env.step((a_level,))
            transitions.append((tuple(obs), a_idx, tuple(next_obs)))
            obs = next_obs
            t += 1
            if env.is_success():
                break
        ep += 1
    print(f"[collect] {len(transitions)} transitions from {ep} TD-MPC2 episodes")
    return transitions


def _load_tdmpc2_agent(ckpt_path: Path, seed: int, model_size: int = 1):
    """Load a TDMPC2 agent from the checkpoint produced by tdmpc2_cpg.py."""
    import hydra.utils
    hydra.utils.get_original_cwd = lambda: os.getcwd()
    from omegaconf import OmegaConf
    from common.parser import parse_cfg
    from envs import make_env as tdmpc2_make_env
    from envs import dmcontrol as tdmpc2_dmcontrol
    from tdmpc2 import TDMPC2

    # Mirror tdmpc2_cpg._patch_dmcontrol_no_frame_skip + _build_cfg so the
    # agent we instantiate matches the architecture and timescale of the
    # one whose weights we are loading.
    cls = tdmpc2_dmcontrol.DMControlWrapper
    def step_once(self, action):
        action = action.astype(self.action_spec_dtype)
        step = self.env.step(action)
        from collections import defaultdict
        return self._obs_to_array(step.observation), float(step.reward or 0.0), False, defaultdict(float)
    cls.step = step_once

    cfg = OmegaConf.load(str(_TDMPC2_PKG / "config.yaml"))
    cfg.task = "cartpole-swingup"
    cfg.obs = "state"
    cfg.steps = 1
    cfg.seed = int(seed)
    cfg.model_size = int(model_size)
    cfg.compile = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.save_csv = False
    cfg.enable_wandb = False
    cfg.exp_name = "coverage_mlp"
    cfg.data_dir = "/tmp/tdmpc2_data"
    cfg.eval_freq = 10_000_000
    cfg.eval_episodes = 0
    cfg.buffer_size = 100
    cfg.work_dir = str(_REPO_ROOT / "results" / "dmc_cartpole" / "tdmpc2_workdir")
    cfg.wandb_project = "wmel"
    cfg.wandb_entity = "noop"
    cfg = parse_cfg(cfg)
    _ = tdmpc2_make_env(cfg)  # populates obs_shape / action_dim on cfg
    agent = TDMPC2(cfg)
    state = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    agent.model.load_state_dict(state["model_state"])
    return agent


def main() -> None:
    args = _parse_args()
    cfg_dict = _config(smoke=args.smoke, n_transitions=args.n_transitions)
    seed = args.seed
    model_size = args.model_size
    levels = DEFAULT_DISCRETE_LEVELS
    suffix = _output_suffix(model_size, seed)
    global AGENT_PATH, JSON_PATH
    # If the caller did not override --agent-ckpt, point at the (size, seed)-
    # suffixed agent so each (size, seed) cell picks up its own training run.
    if args.agent_ckpt == str(AGENT_PATH):
        args.agent_ckpt = str(_REPO_ROOT / "results" / "dmc_cartpole" / f"tdmpc2_agent{suffix}.pt")
    AGENT_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / f"tdmpc2_agent{suffix}.pt"
    JSON_PATH = _REPO_ROOT / "results" / "dmc_cartpole" / f"coverage_mlp_on_tdmpc2_cpg{suffix}.json"

    # Validate code path with random data if smoke OR if the data-source is
    # random; otherwise load the TD-MPC2 agent and run its policy.
    use_tdmpc2 = (args.data_source == "tdmpc2") and not args.smoke
    if use_tdmpc2:
        ckpt = Path(args.agent_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"TD-MPC2 agent checkpoint not found at {ckpt}. Run experiments.dmc_cartpole.tdmpc2_cpg first or pass --data-source random.")
        agent = _load_tdmpc2_agent(ckpt, seed=seed, model_size=model_size)
        print(f"[1/5] Collecting {cfg_dict['n_transitions']} transitions from TD-MPC2 eval-mode policy...")
        transitions = _collect_tdmpc2_rollouts(
            agent=agent,
            env_factory=lambda: DMCCartpoleEnv(discrete_levels=levels),
            n_transitions=cfg_dict["n_transitions"],
            levels=levels,
            seed=seed,
        )
    else:
        # Random-policy baseline path: identical to v0.11.
        n_eps = max(1, cfg_dict["n_transitions"] // 200)
        print(f"[1/5] Collecting {cfg_dict['n_transitions']} transitions from random policy ({n_eps} episodes x 200 steps)...")
        transitions = collect_random_rollouts(
            lambda: DMCCartpoleEnv(discrete_levels=levels),
            n_episodes=n_eps,
            max_steps_per_episode=200,
            seed=seed,
        )
        transitions = transitions[: cfg_dict["n_transitions"]]

    coverage = _coverage_stats(transitions, levels)
    print(f"[coverage] n={coverage['n_transitions']}  u_max={coverage['uprightness_max']:.3f}  "
          f"frac(u>1)={coverage['fraction_u_gt_1']:.3f}  frac(u>1.5)={coverage['fraction_u_gt_1_5']:.3f}")

    print(f"[2/5] Training MLP world model ({cfg_dict['epochs']} epochs) on collected transitions...")
    env_template = DMCCartpoleEnv(discrete_levels=levels)
    model, train_log = train_world_model(
        transitions,
        obs_dim=5,
        n_actions=len(env_template.action_space),
        epochs=cfg_dict["epochs"],
        seed=seed,
    )
    print(f"      val_mse={train_log['final_val_mse']:.6f}  train_mse={train_log['final_train_mse']:.6f}")

    def make_planner(dyn):
        return TabularWorldModelPlanner(
            dynamics=dyn,
            action_space=env_template.action_space,
            num_candidates=cfg_dict["num_candidates"],
            plan_horizon=cfg_dict["plan_horizon"],
            score=cartpole_upright_score,
            seed=seed,
        )

    print("[3/5] Benchmarking with ORACLE dynamics...")
    oracle_planner = make_planner(make_cartpole_oracle_dynamics())
    oracle_results = BenchmarkRunner(
        env_factory=lambda: DMCCartpoleEnv(discrete_levels=levels),
        policy=oracle_planner,
        episodes=cfg_dict["benchmark_episodes"],
        horizon=cfg_dict["benchmark_horizon"],
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
        env_factory=lambda: DMCCartpoleEnv(discrete_levels=levels),
        policy=learned_planner,
        episodes=cfg_dict["benchmark_episodes"],
        horizon=cfg_dict["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()
    effective_source = "tdmpc2" if use_tdmpc2 else "random"
    learned_card = compute_scorecard(
        learned_results,
        policy_name=f"tabular-world-model (MLP on {effective_source} data)",
        compute_per_plan_call=learned_planner.compute_per_plan_call,
        perturbation_name="env-default",
    )
    print_scorecard(learned_card)

    print("[5/5] Computing CPG...")
    cpg = counterfactual_planning_gap(oracle_results, learned_results)
    verdict = cpg_verdict(cpg)
    print(f"  oracle  success = {cpg.oracle_success_rate:.3f} (n={cpg.n_episodes_oracle})")
    print(f"  learned success = {cpg.learned_success_rate:.3f} (n={cpg.n_episodes_learned})")
    print(f"  CPG = {cpg.gap:+.3f}  95% AC CI [{cpg.gap_ci_low:+.3f}, {cpg.gap_ci_high:+.3f}]")
    if args.smoke:
        print("  Verdict: SMOKE MODE (config too small for diagnosis; verdict suppressed)")
    else:
        print(f"  Verdict: {verdict}")

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_cartpole_swingup",
        "metric": "counterfactual_planning_gap",
        "learned_model": "mlp_world_model",
        "data_source": effective_source,
        "cpg": {**asdict(cpg), "verdict": verdict},
        "config": cfg_dict,
        "training": train_log,
        "coverage": coverage,
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
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {JSON_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
