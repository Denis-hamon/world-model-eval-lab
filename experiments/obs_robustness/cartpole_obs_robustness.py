"""Observation-richness stress test on DMC Cartpole (CPU).

Reframed T2.3 (see this directory's README and experiments/GPU_ROADMAP.md:92 for
why this is nuisance-augmented state, not pixels). Question: is the
prediction != decision dissociation an artifact of the clean, privileged
low-dimensional observation, or does it survive when the learned world model
must work from a richer, partly task-irrelevant observation?

Design. For each (nuisance kind, width, seed) we wrap DMC Cartpole so the
observation is ``true_state ++ nuisance`` (``observation_augmentation`` does the
algebra), train a small MLP world model on the augmented observation (CPU),
and run the CPG protocol with the planner scoring ONLY the true-state slice:

  - oracle arm : the augmented oracle (true Cartpole physics on the state slice,
    nuisance reproduced exactly);
  - learned arm: the trained MLP on the augmented observation.

Only the dynamics callable changes between arms, so CPG = success(oracle) -
success(learned) is a clean counterfactual. Alongside CPG we record the learned
model's one-step error split into the decision-relevant state slice and the
nuisance slice.

Falsifiable claim. CPG and mse_state stay ~flat across nuisance kind and width,
while mse_total moves a lot (down for ``redundant`` smooth features, up for the
``high_freq`` features). That would show a popular "model quality" proxy
(observation-space one-step MSE -- the M1 foil of the keystone) is movable by
task-irrelevant observation design without touching the closed-loop verdict,
sharpening prediction != decision. If instead CPG shifts materially with the
nuisance, the verdict is observation-form-dependent -- itself a reportable
finding. ``width=0`` is the no-nuisance control and must reproduce the baseline
Cartpole CPG.

Usage:
    pip install -e ".[control,learned,dev]"
    python -m experiments.obs_robustness.cartpole_obs_robustness            # full
    python -m experiments.obs_robustness.cartpole_obs_robustness --smoke    # tiny
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

# Stdlib + pure-algebra import at module scope; torch/dm_control are pulled in
# inside main() so a syntax/import check does not require the heavy extras.
from experiments.obs_robustness.observation_augmentation import (  # noqa: E402
    HighFrequencyFeatures,
    ObsAugmentedEnv,
    RedundantFeatures,
    make_augmented_oracle,
    make_augmented_score,
    one_step_mse_split,
)


def _config(smoke: bool) -> dict:
    if smoke:
        return {"kinds": ["redundant", "high_freq"], "widths": [0, 4], "seeds": [0],
                "n_mlp_episodes": 5, "mlp_max_steps": 80, "mlp_epochs": 20, "mlp_hidden": 64,
                "num_candidates": 30, "plan_horizon": 8,
                "benchmark_episodes": 2, "benchmark_horizon": 80,
                "mse_states": 16}
    return {"kinds": ["redundant", "high_freq"], "widths": [0, 4, 16, 64], "seeds": [0, 1, 2],
            "n_mlp_episodes": 40, "mlp_max_steps": 200, "mlp_epochs": 200, "mlp_hidden": 64,
            "num_candidates": 50, "plan_horizon": 15,
            "benchmark_episodes": 10, "benchmark_horizon": 500,
            "mse_states": 200}


def _make_spec(kind: str, width: int, base_dim: int):
    if width == 0:
        return None  # no-nuisance control: plain base env / oracle / score
    if kind == "redundant":
        return RedundantFeatures(width=width, base_dim=base_dim)
    if kind == "high_freq":
        return HighFrequencyFeatures(width=width, base_dim=base_dim)
    raise ValueError(f"unknown nuisance kind: {kind!r}")


def _collect_aug_states(env_factory, action_space, n_states, seed):
    """A fixed seeded random-action rollout, returning the augmented obs visited
    -- the sample on which the one-step MSE split is measured."""
    rng = random.Random(seed)
    env = env_factory()
    obs = env.reset()
    states = []
    for i in range(n_states):
        states.append(tuple(float(x) for x in obs))
        if i == n_states - 1:
            break
        if (i + 1) % 25 == 0:
            obs = env.reset()
        else:
            obs = env.step(rng.choice(action_space))
    return states


def run_cell(kind, width, seed, cfg, base_dim, base_oracle_factory, base_score, env_cls):
    from wmel.adapters.mlp_world_model import (
        collect_random_rollouts, learned_dynamics, train_world_model,
    )
    from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
    from wmel.benchmark_runner import BenchmarkRunner
    from wmel.metrics import counterfactual_planning_gap, cpg_verdict

    spec = _make_spec(kind, width, base_dim)
    if spec is None:
        env_factory = lambda: env_cls()
        oracle_dyn = base_oracle_factory()
        score = base_score
        obs_dim = base_dim
    else:
        env_factory = lambda: ObsAugmentedEnv(env_cls(), spec)
        oracle_dyn = make_augmented_oracle(base_oracle_factory(), base_dim, spec)
        score = make_augmented_score(base_score, base_dim)
        obs_dim = base_dim + width

    action_space = env_cls().action_space

    # Learned arm: MLP trained on the augmented observation.
    transitions = collect_random_rollouts(
        env_factory, n_episodes=cfg["n_mlp_episodes"],
        max_steps_per_episode=cfg["mlp_max_steps"], seed=seed)
    model, train_log = train_world_model(
        transitions, obs_dim=obs_dim, n_actions=len(action_space),
        epochs=cfg["mlp_epochs"], hidden=cfg["mlp_hidden"], seed=seed)
    learned_dyn = learned_dynamics(model, action_space)

    def make_planner(dyn):
        return TabularWorldModelPlanner(
            dynamics=dyn, action_space=action_space,
            num_candidates=cfg["num_candidates"], plan_horizon=cfg["plan_horizon"],
            score=score, seed=seed)

    def run_arm(dyn):
        return BenchmarkRunner(
            env_factory=env_factory, policy=make_planner(dyn),
            episodes=cfg["benchmark_episodes"], horizon=cfg["benchmark_horizon"],
            perturb_prob=0.0, seed=seed).run()

    oracle_results = run_arm(oracle_dyn)
    learned_results = run_arm(learned_dyn)
    cpg = counterfactual_planning_gap(oracle_results, learned_results)
    verdict = cpg_verdict(cpg)

    # One-step error split (decision-relevant state vs nuisance). At width=0 the
    # nuisance slice is empty, so mse_nuisance is 0 and mse_total == mse_state.
    states = _collect_aug_states(env_factory, action_space, cfg["mse_states"], seed)
    mse = one_step_mse_split(oracle_dyn, learned_dyn, states, action_space, base_dim)

    return {
        "kind": "none" if spec is None else kind,
        "width": width,
        "seed": seed,
        "gap": cpg.gap,
        "gap_ci_low": cpg.gap_ci_low,
        "gap_ci_high": cpg.gap_ci_high,
        "verdict": verdict,
        "mse_total": round(mse["mse_total"], 8),
        "mse_state": round(mse["mse_state"], 8),
        "mse_nuisance": round(mse["mse_nuisance"], 8) if width else None,
        "context": {
            "oracle_success_rate": cpg.oracle_success_rate,
            "learned_success_rate": cpg.learned_success_rate,
            "final_val_mse": train_log["final_val_mse"],
            "obs_dim": obs_dim,
            "n_episodes": cpg.n_episodes_oracle,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny config for wiring validation")
    args = ap.parse_args()
    cfg = _config(args.smoke)

    from wmel.envs.dmc_cartpole import (
        DMCCartpoleEnv, cartpole_upright_score, make_cartpole_oracle_dynamics,
    )
    from wmel.report import report_envelope_metadata

    base_dim = len(DMCCartpoleEnv().reset())
    print(f"[setup] base Cartpole obs dim = {base_dim}; "
          f"kinds={cfg['kinds']} widths={cfg['widths']} seeds={cfg['seeds']}")

    rows = []
    # width=0 is kind-independent; run it once per seed under kind "none".
    done_zero = set()
    plan = []
    for kind in cfg["kinds"]:
        for width in cfg["widths"]:
            for seed in cfg["seeds"]:
                if width == 0:
                    if seed in done_zero:
                        continue
                    done_zero.add(seed)
                plan.append((kind, width, seed))

    for kind, width, seed in plan:
        rec = run_cell(kind, width, seed, cfg, base_dim,
                       make_cartpole_oracle_dynamics, cartpole_upright_score, DMCCartpoleEnv)
        rows.append(rec)
        mn = "  n/a   " if rec["mse_nuisance"] is None else f"{rec['mse_nuisance']:.4f}"
        print(f"  {rec['kind']:<9} w={rec['width']:<3} seed={seed}  "
              f"gap={rec['gap']:+.3f}  mse_total={rec['mse_total']:.4f} "
              f"mse_state={rec['mse_state']:.4f} mse_nuis={mn}  [{rec['verdict']}]")

    report = {
        **report_envelope_metadata(),
        "metric": "observation_robustness_cpg",
        "environment": "dmc_cartpole",
        "note": (
            "Reframed T2.3: does prediction != decision survive a richer, partly "
            "task-irrelevant observation? Observation = true_state ++ nuisance; "
            "the sim stays oracle and the planner scores only the state slice. "
            "Compare gap and mse_state (should be ~flat across width/kind) against "
            "mse_total (expected down for redundant smooth features, up for "
            "high_freq features). width=0 is the no-nuisance control. Non-pixel "
            "by the line-92 non-affiliation guardrail; same scientific payload."
        ),
        "config": cfg,
        "cells": rows,
    }
    out = _REPO_ROOT / "results" / "obs_robustness" / "cartpole_obs_robustness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"\nWrote {out.relative_to(_REPO_ROOT)} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
