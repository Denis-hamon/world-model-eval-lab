"""State-space coverage analysis for DMC Acrobot-swingup.

The v0.11 paper (Section 5.6, "What CPG separates: capacity vs.\\ coverage")
argues that the flat CPG curve under monotonically improving prediction loss
points to a data-coverage bottleneck: random rollouts in Acrobot do not visit
the high-energy upright regime, so the MLP world model is extrapolating
during planning. This script provides the empirical receipt for that claim.

What it measures
----------------

The natural "uprightness" axis on Acrobot is

    uprightness(state) = cos(upper) + cos(lower) = state[2] + state[3]

(see `acrobot_upright_score` for the layout derivation). Range: -2 (both
arms hanging straight down) to +2 (both arms pointing straight up). The
DMC reward `r_t >= 0.6` ("is_success") corresponds to a tip height
around 1.2-1.5 on this axis.

We compute the empirical distribution of `uprightness(state)` on:

1. The random-policy rollout dataset that v0.11's MLP was trained on
   (10 episodes x 200 steps = 2000 transitions; same recipe as the
   v0.11 sweep's `data_size=2000` cell).
2. The oracle-planner trajectories that v0.11 benchmarks against
   (a small number of episodes is enough for the histogram; the
   oracle-planner *attempts* to swing up so its state distribution
   skews much higher).

Output:

  - A bucketed histogram of uprightness for each dataset, written to
    `results/dmc_acrobot/coverage.json`.
  - Two summary fractions per dataset: fraction of states with
    uprightness > 1.0 (approaching upright) and > 1.5 (near upright).

Stdlib + numpy + torch (already pulled by `[learned]` extra) +
dm-control (`[control]` extra) for the oracle dynamics. CPU only.

Usage:

    pip install -e ".[dev,control,learned]"
    python -m experiments.dmc_acrobot.coverage_analysis

Smoke (~10 s, smaller config):

    python -m experiments.dmc_acrobot.coverage_analysis --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.adapters.mlp_world_model import (
    acrobot_upright_score,
    collect_random_rollouts,
)
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.envs.dmc_acrobot import DMCAcrobotEnv, make_acrobot_oracle_dynamics
from wmel.report import report_envelope_metadata

# Bucket edges on the uprightness axis. `uprightness = cos_upper + cos_lower`
# ranges in [-2, +2]. "Near upright" is anything around +2 (both arms up);
# the DMC success threshold corresponds roughly to uprightness > 1.0-1.2.
DEFAULT_BUCKETS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def uprightness(state) -> float:
    """Convenience: positive-is-better, the negation of acrobot_upright_score."""
    return -acrobot_upright_score(state)


def histogram(values, edges) -> list[int]:
    """Stdlib bucket counter. Returns counts per (edges[i], edges[i+1]) bin.

    The first bin is half-open on the left (<= edges[0] folds into it);
    the last bin is half-open on the right (>= edges[-1] folds into it).
    Length: len(edges) - 1.
    """
    counts = [0] * (len(edges) - 1)
    for v in values:
        # Find the last edge index e_i such that v >= edges[e_i].
        # Clamp into [0, len(counts) - 1].
        idx = 0
        for i in range(len(edges) - 1):
            if v >= edges[i]:
                idx = i
        counts[idx] += 1
    return counts


def collect_random_states(n_episodes: int, max_steps: int, seed: int) -> list[float]:
    """Run random-policy rollouts and return the uprightness of every state visited."""
    transitions = collect_random_rollouts(
        DMCAcrobotEnv,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps,
        seed=seed,
    )
    # Each transition is (obs, action_idx, next_obs). To avoid double-counting
    # the same state, we take only `obs` from each transition; this matches the
    # "what did the dataset show the model" framing.
    return [uprightness(obs) for (obs, _a, _next) in transitions]


def collect_oracle_planner_states(
    n_episodes: int,
    max_steps: int,
    num_candidates: int,
    plan_horizon: int,
    seed: int,
) -> list[float]:
    """Run the oracle-planner for a few episodes and capture every state visited."""
    env_template = DMCAcrobotEnv()
    planner = TabularWorldModelPlanner(
        dynamics=make_acrobot_oracle_dynamics(),
        action_space=env_template.action_space,
        num_candidates=num_candidates,
        plan_horizon=plan_horizon,
        score=acrobot_upright_score,
        seed=seed,
    )
    states: list[float] = []
    for ep in range(n_episodes):
        env = DMCAcrobotEnv()
        obs = env.reset()
        states.append(uprightness(obs))
        # Manual one-action-per-step loop (no replanning between plan() calls
        # would mean using stale plans; we replan every step so the oracle
        # planner is exercised the same way as in the benchmark).
        for _step in range(max_steps):
            actions = planner.plan(obs, env.goal, plan_horizon)
            if not actions:
                break
            obs = env.step(actions[0])
            states.append(uprightness(obs))
            if env.is_success():
                break
    return states


def summarise(values, edges, label, episodes, max_steps):
    counts = histogram(values, edges)
    n = len(values)
    frac_gt_10 = sum(1 for v in values if v > 1.0) / max(n, 1)
    frac_gt_15 = sum(1 for v in values if v > 1.5) / max(n, 1)
    mean = sum(values) / max(n, 1)
    mx = max(values) if values else float("nan")
    return {
        "label": label,
        "n_states": n,
        "episodes": episodes,
        "max_steps_per_episode": max_steps,
        "bucket_edges": list(edges),
        "bucket_counts": counts,
        "mean_uprightness": mean,
        "max_uprightness": mx,
        "frac_above_1_0": frac_gt_10,
        "frac_above_1_5": frac_gt_15,
    }


def _format_row(summary):
    edges = summary["bucket_edges"]
    counts = summary["bucket_counts"]
    n = summary["n_states"]
    parts = []
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        pct = 100.0 * c / max(n, 1)
        parts.append(f"  [{lo:+.1f},{hi:+.1f}): {c:>6d} ({pct:>5.1f}%)")
    return "\n".join(parts)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Tiny config for CI (~10 s).")
    p.add_argument("--random-episodes", type=int, default=10)
    p.add_argument("--random-steps", type=int, default=200)
    p.add_argument("--oracle-episodes", type=int, default=5)
    p.add_argument("--oracle-steps", type=int, default=300)
    p.add_argument("--num-candidates", type=int, default=50)
    p.add_argument("--plan-horizon", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        type=str,
        default="results/dmc_acrobot/coverage.json",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        cfg = dict(
            random_episodes=2, random_steps=80,
            oracle_episodes=1, oracle_steps=60,
            num_candidates=10, plan_horizon=6,
        )
        if args.output == "results/dmc_acrobot/coverage.json":
            args.output = "results/dmc_acrobot/coverage_smoke.json"
    else:
        cfg = dict(
            random_episodes=args.random_episodes,
            random_steps=args.random_steps,
            oracle_episodes=args.oracle_episodes,
            oracle_steps=args.oracle_steps,
            num_candidates=args.num_candidates,
            plan_horizon=args.plan_horizon,
        )

    print(f"[coverage] config: {cfg}, seed={args.seed}")

    print(f"[coverage] collecting random states "
          f"({cfg['random_episodes']} ep x {cfg['random_steps']} steps)...")
    rand_states = collect_random_states(
        cfg["random_episodes"], cfg["random_steps"], args.seed
    )
    rand_summary = summarise(
        rand_states, DEFAULT_BUCKETS, "random_rollouts",
        cfg["random_episodes"], cfg["random_steps"],
    )

    print(f"[coverage] collecting oracle-planner states "
          f"({cfg['oracle_episodes']} ep x {cfg['oracle_steps']} steps)...")
    oracle_states = collect_oracle_planner_states(
        cfg["oracle_episodes"], cfg["oracle_steps"],
        cfg["num_candidates"], cfg["plan_horizon"], args.seed,
    )
    oracle_summary = summarise(
        oracle_states, DEFAULT_BUCKETS, "oracle_planner",
        cfg["oracle_episodes"], cfg["oracle_steps"],
    )

    print()
    print(f"=== {rand_summary['label']} (n={rand_summary['n_states']} states) ===")
    print(_format_row(rand_summary))
    print(f"  mean uprightness: {rand_summary['mean_uprightness']:+.3f}")
    print(f"  max  uprightness: {rand_summary['max_uprightness']:+.3f}")
    print(f"  frac > 1.0      : {rand_summary['frac_above_1_0']*100:.2f}%")
    print(f"  frac > 1.5      : {rand_summary['frac_above_1_5']*100:.2f}%")

    print()
    print(f"=== {oracle_summary['label']} (n={oracle_summary['n_states']} states) ===")
    print(_format_row(oracle_summary))
    print(f"  mean uprightness: {oracle_summary['mean_uprightness']:+.3f}")
    print(f"  max  uprightness: {oracle_summary['max_uprightness']:+.3f}")
    print(f"  frac > 1.0      : {oracle_summary['frac_above_1_0']*100:.2f}%")
    print(f"  frac > 1.5      : {oracle_summary['frac_above_1_5']*100:.2f}%")

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "metric": "state_space_coverage",
        "axis": "uprightness = cos(upper) + cos(lower)",
        "seed": args.seed,
        "config": cfg,
        "datasets": [rand_summary, oracle_summary],
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
