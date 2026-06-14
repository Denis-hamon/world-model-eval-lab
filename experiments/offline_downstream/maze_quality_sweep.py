"""Offline-metric vs downstream-success sweep on the maze toy (CPU).

Stage T2.1.b of the keystone, CPU-only proof of concept: does a cheap,
planner-free *offline* metric predict *downstream* planning success? We need
models that span a quality range with non-floor success; the learned MLP arms
on DMC sit at the planning floor (success 0), so we use the fast maze toy where
under-training the learned dynamics genuinely degrades planning.

For each (epochs, seed) we train the maze MLP dynamics (under-training is the
quality knob), then compute three planner-free offline metrics against the true
maze dynamics and the downstream planning success of the same learned dynamics:

  M1  one-step mismatch   : fraction of (state, action) whose predicted next
                            cell differs from the true next cell. The naive
                            "reconstruction" metric -- the foil.
  M2  k-step divergence   : mean Manhattan distance between learned and true
                            open-loop rollouts under a fixed action sequence
                            (compounding error).
  M3  action agreement    : mean Kendall tau between the true and learned
                            *ranking of actions by closeness-to-goal* at each
                            state -- the decision-aware metric (does the model
                            rank actions the way the planner needs?).
  downstream success_rate : TabularWorldModelPlanner using the learned dynamics.

Writes results/offline_downstream/maze_offline_scores.json; correlate.py turns
it into rank correlations. Heavy dep (torch) lives here, in experiments/.

Usage:
    pip install -e ".[learned]"
    python -m experiments.offline_downstream.maze_quality_sweep            # full
    python -m experiments.offline_downstream.maze_quality_sweep --smoke    # tiny
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

from wmel.adapters.learned_dynamics_torch import (  # noqa: E402
    collect_transitions,
    torch_dynamics,
    train_maze_dynamics,
)
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner  # noqa: E402
from wmel.benchmark_runner import BenchmarkRunner  # noqa: E402
from wmel.metrics import action_success_rate, kendall_tau  # noqa: E402
from wmel.report import report_envelope_metadata  # noqa: E402

from examples.maze_toy.environment import VALID_ACTIONS, MazeEnv  # noqa: E402

EPOCHS_GRID = [2, 4, 8, 16, 32, 64, 150, 400]
SEEDS = [0, 1, 2]
HIDDEN = 32
K_STEPS = 8
EPISODES = 20
EPISODE_HORIZON = 50
PLAN_HORIZON = 20
NUM_CANDIDATES = 100  # reduced from the learned-baseline's 200 to keep the sweep fast


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _offline_metrics(env: MazeEnv, learned_dyn) -> dict:
    """M1 / M2 / M3 for a learned dynamics vs the true maze dynamics (no planner)."""
    transitions = collect_transitions(env)  # (state, action, true_next_state)
    states = sorted({s for s, _, _ in transitions})
    goal = env.goal_pos

    # M1: one-step mismatch rate.
    mismatches = sum(
        1 for s, a, ns in transitions if learned_dyn(s, a) != ns
    )
    m1 = mismatches / len(transitions)

    # M2: k-step open-loop divergence under one fixed action sequence.
    seq_rng = random.Random(0)
    seq = [seq_rng.choice(VALID_ACTIONS) for _ in range(K_STEPS)]
    divs = []
    for s in states:
        st, sl = s, s
        for a in seq:
            st = env.dynamics(st, a)
            sl = learned_dyn(sl, a)
        divs.append(_manhattan(st, sl))
    m2 = sum(divs) / len(divs)

    # M3: action-ranking agreement (decision-aware). Score each action by
    # closeness-to-goal of its predicted successor; compare true vs learned
    # rankings with Kendall tau, averaged over states with a non-degenerate
    # true ranking.
    agreements = []
    for s in states:
        true_scores = [-_manhattan(env.dynamics(s, a), goal) for a in VALID_ACTIONS]
        learned_scores = [-_manhattan(learned_dyn(s, a), goal) for a in VALID_ACTIONS]
        try:
            agreements.append(kendall_tau(true_scores, learned_scores))
        except ValueError:
            # Degenerate ranking on EITHER side -- in practice the action-blind
            # worst models, whose learned scores are constant across actions
            # (so M3 is undefined for them, which is itself informative).
            continue
    # None (-> JSON null) when no state has a defined ranking, rather than NaN
    # (invalid JSON). correlate.py drops non-finite cells per metric.
    m3 = round(sum(agreements) / len(agreements), 4) if agreements else None

    return {"m1_mismatch": round(m1, 4), "m2_kstep_divergence": round(m2, 4),
            "m3_action_agreement": m3}


def _downstream_success(learned_dyn, seed: int, episodes: int, horizon: int) -> float:
    planner = TabularWorldModelPlanner(
        dynamics=learned_dyn,
        action_space=VALID_ACTIONS,
        num_candidates=NUM_CANDIDATES,
        plan_horizon=PLAN_HORIZON,
        seed=seed,
    )
    results = BenchmarkRunner(
        env_factory=MazeEnv,
        policy=planner,
        episodes=episodes,
        horizon=horizon,
        perturb_prob=0.0,
        seed=seed,
    ).run()
    return action_success_rate(results)


def run_cell(epochs: int, seed: int, episodes: int, horizon: int) -> dict:
    template = MazeEnv()
    model = train_maze_dynamics(template, epochs=epochs, hidden=HIDDEN, seed=seed)
    learned_dyn = torch_dynamics(model, template.width, template.height)
    offline = _offline_metrics(template, learned_dyn)
    success = _downstream_success(learned_dyn, seed, episodes, horizon)
    return {"epochs": epochs, "seed": seed, "hidden": HIDDEN,
            **offline, "success_rate": round(success, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run for CI")
    args = ap.parse_args()

    if args.smoke:
        grid, seeds, episodes, horizon = [2, 50], [0], 3, 20
    else:
        grid, seeds, episodes, horizon = EPOCHS_GRID, SEEDS, EPISODES, EPISODE_HORIZON

    cells = []
    for epochs in grid:
        for seed in seeds:
            rec = run_cell(epochs, seed, episodes, horizon)
            cells.append(rec)
            m3 = rec["m3_action_agreement"]
            m3s = "n/a  " if m3 is None else f"{m3:.3f}"
            print(f"  epochs={epochs:>4} seed={seed}  M1={rec['m1_mismatch']:.3f} "
                  f"M2={rec['m2_kstep_divergence']:.3f} M3={m3s} "
                  f"success={rec['success_rate']:.3f}")

    report = {
        **report_envelope_metadata(),
        "metric": "maze_offline_downstream_sweep",
        "env": "maze_toy",
        "note": (
            "CPU proof-of-concept for the offline->downstream keystone. Quality "
            "knob is training epochs (under-training degrades the learned maze "
            "dynamics). M1 one-step mismatch (naive), M2 k-step divergence "
            "(compounding), M3 action-ranking agreement (decision-aware); "
            "downstream is planning success_rate. correlate.py turns these into "
            "rank correlations. The DMC/TD-MPC2 cells (the headline) need GPU."
        ),
        "config": {"hidden": HIDDEN, "k_steps": K_STEPS, "episodes": episodes,
                   "episode_horizon": horizon, "plan_horizon": PLAN_HORIZON,
                   "num_candidates": NUM_CANDIDATES},
        "cells": cells,
    }
    out = _REPO_ROOT / "results" / "offline_downstream" / "maze_offline_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")  # null, never NaN
    print(f"\nWrote {out.relative_to(_REPO_ROOT)} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
