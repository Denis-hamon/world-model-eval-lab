"""Command-line interface for `wmel`.

Two subcommands so far:

  wmel run    -- run a single benchmark (one env + one policy + one perturbation)
  wmel sweep  -- run a planning-horizon sweep on the maze toy env

The CLI is stdlib-only (argparse, json), no third-party dependencies. It
imports the same `BenchmarkRunner`, `horizon_sweep`, and reporting helpers
that the example scripts do. Anything the CLI can compute, the Python API
can compute - the CLI is a convenience surface, not a separate runtime.

The exported `main` function is the entry point declared in
`pyproject.toml` as the `wmel` console script.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from wmel.adapters.base import BenchmarkEnvironment, PlannerPolicy
from wmel.adapters.greedy_policy import GreedyGridPolicy
from wmel.adapters.random_policy import RandomPolicy
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.benchmark_runner import BenchmarkRunner
from wmel.experiments import horizon_sweep, print_horizon_sweep
from wmel.metrics import compute_scorecard
from wmel.perturbations import (
    CompositePerturbation,
    DropNextActions,
    EnvPerturbation,
    Perturbation,
)
from wmel.report import REPORT_SCHEMA_VERSION, print_scorecard, to_json_report


# ---------------------------------------------------------------------------
# Env / policy / perturbation registries
# ---------------------------------------------------------------------------
#
# We register the small set of envs and policies that ship with this repo
# so the CLI can refer to them by short string keys. Out-of-tree consumers
# can either fork this module or build their own CLI on top of the public
# API - the registry is intentionally not a public extension point.

# Lazy imports keep the CLI fast to start when only one subcommand is used.

def _maze_env_factory() -> BenchmarkEnvironment:
    from wmel.envs.maze_toy import MazeEnv

    return MazeEnv()


def _two_room_env_factory() -> BenchmarkEnvironment:
    from wmel.envs.two_room_toy import TwoRoomEnv

    return TwoRoomEnv()


ENVS: dict[str, Callable[[], BenchmarkEnvironment]] = {
    "maze_toy": _maze_env_factory,
    "two_room_toy": _two_room_env_factory,
}


def _build_random_policy(env: BenchmarkEnvironment, *, seed: int) -> PlannerPolicy:
    return RandomPolicy(action_space=env.action_space, seed=seed)


def _build_greedy_policy(env: BenchmarkEnvironment, *, seed: int) -> PlannerPolicy:
    # The greedy policy needs a waypoint hint on envs with a topological
    # bottleneck (e.g., a doorway through a wall). Without it, two-room
    # success rate silently regresses to 0%. We special-case the two-room
    # env to inject its doorway-pointing waypoint function.
    from wmel.envs.two_room_toy import TwoRoomEnv, two_room_waypoint_for

    if isinstance(env, TwoRoomEnv):
        return GreedyGridPolicy(waypoint_fn=two_room_waypoint_for(env))
    return GreedyGridPolicy()


def _build_tabular_wm_policy(
    env: BenchmarkEnvironment, *, seed: int, plan_horizon: int = 20
) -> PlannerPolicy:
    # The tabular world model needs a dynamics function. For the in-tree
    # envs that expose `dynamics`, use it directly. Other envs are not yet
    # supported by this policy in the CLI.
    if not hasattr(env, "dynamics"):
        raise SystemExit(
            f"env {type(env).__name__} does not expose a `dynamics` callable; "
            f"the tabular-world-model policy needs one. Try --policy random "
            f"or --policy greedy instead."
        )
    return TabularWorldModelPlanner(
        dynamics=env.dynamics,
        action_space=env.action_space,
        num_candidates=200,
        plan_horizon=plan_horizon,
        seed=seed,
    )


PolicyBuilder = Callable[..., PlannerPolicy]

POLICIES: dict[str, PolicyBuilder] = {
    "random": _build_random_policy,
    "greedy": _build_greedy_policy,
    "tabular-world-model": _build_tabular_wm_policy,
}


def _build_perturbation(spec: str) -> Perturbation:
    """Resolve a `--perturbation` argument to a concrete `Perturbation`.

    Accepted spec strings:
      env-default         delegate to env.perturb() (the runner's default)
      drop-next-K         drop next K queued actions (K is a positive int)
      composite:A+B       chain the two strategies above (env-default + drop-next-K)
    """
    if spec == "env-default":
        return EnvPerturbation()
    if spec.startswith("drop-next-"):
        try:
            k = int(spec[len("drop-next-") :])
        except ValueError as exc:
            raise SystemExit(f"invalid drop-next spec {spec!r}: {exc}") from exc
        return DropNextActions(k=k)
    if spec.startswith("composite:"):
        parts = [_build_perturbation(p.strip()) for p in spec[len("composite:") :].split("+")]
        return CompositePerturbation(*parts)
    raise SystemExit(
        f"unknown perturbation spec {spec!r}. "
        "Accepted: env-default | drop-next-<K> | composite:A+B"
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _validate_probability(value: float, flag: str) -> None:
    """Reject probabilities outside [0, 1] with a friendly exit message."""
    if not (0.0 <= value <= 1.0):
        raise SystemExit(
            f"{flag} must be in [0.0, 1.0]; got {value!r}. "
            "A negative or >1 probability would silently bias the run."
        )


def _parse_plan_horizons(spec: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive ints, with a clean error."""
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if not items:
        raise SystemExit(
            "--plan-horizons must be a non-empty comma-separated list of ints "
            f"(got {spec!r})."
        )
    parsed: list[int] = []
    for item in items:
        try:
            n = int(item)
        except ValueError as exc:
            raise SystemExit(
                f"--plan-horizons must contain only integers; got {item!r} "
                f"in {spec!r}."
            ) from exc
        if n <= 0:
            raise SystemExit(
                f"--plan-horizons must contain only positive integers; "
                f"got {n} in {spec!r}."
            )
        parsed.append(n)
    return tuple(parsed)


def cmd_run(args: argparse.Namespace) -> int:
    _validate_probability(args.perturb_prob, "--perturb-prob")
    env = ENVS[args.env]()
    policy_builder = POLICIES[args.policy]
    policy_kwargs: dict[str, Any] = {"seed": args.seed}
    if args.policy == "tabular-world-model":
        policy_kwargs["plan_horizon"] = args.plan_horizon
    policy = policy_builder(env, **policy_kwargs)

    perturbation = _build_perturbation(args.perturbation)

    results = BenchmarkRunner(
        env_factory=ENVS[args.env],
        policy=policy,
        episodes=args.episodes,
        horizon=args.horizon,
        perturb_prob=args.perturb_prob,
        perturbation=perturbation,
        seed=args.seed,
    ).run()

    scorecard = compute_scorecard(
        results,
        policy_name=policy.name,
        compute_per_plan_call=policy.compute_per_plan_call,
        perturbation_name=perturbation.name,
    )
    print_scorecard(scorecard)

    if args.output:
        envelope = to_json_report(
            results,
            scorecard,
            extra_metadata={
                "env": args.env,
                "policy": args.policy,
                "episodes": args.episodes,
                "horizon": args.horizon,
                "perturb_prob": args.perturb_prob,
                "perturbation": perturbation.name,
                "seed": args.seed,
            },
        )
        Path(args.output).write_text(json.dumps(envelope, indent=2) + "\n")
        print(f"Wrote report to {args.output}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    # The argparse `--policy` choices restrict this to the only supported
    # sweep policy, so we don't need a body-level guard. Leave the check
    # to argparse so the user sees its canonical error message.
    _validate_probability(args.perturb_prob, "--perturb-prob")
    horizons = _parse_plan_horizons(args.plan_horizons)
    perturbation = _build_perturbation(args.perturbation)

    def policy_factory(plan_horizon: int) -> PlannerPolicy:
        env = ENVS[args.env]()
        return _build_tabular_wm_policy(env, seed=args.seed, plan_horizon=plan_horizon)

    sweep = horizon_sweep(
        env_factory=ENVS[args.env],
        policy_factory=policy_factory,
        plan_horizons=horizons,
        episodes_per_point=args.episodes_per_point,
        episode_horizon=args.episode_horizon,
        perturb_prob=args.perturb_prob,
        perturbation=perturbation,
        seed=args.seed,
    )

    print_horizon_sweep(sweep)

    if args.output:
        from wmel import __version__ as _wmel_version
        from datetime import datetime, timezone

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "wmel_version": _wmel_version,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "policy_name": sweep.policy_name,
            "metadata": {
                "env": args.env,
                "policy": args.policy,
                "perturb_prob": args.perturb_prob,
                "perturbation": perturbation.name,
                "seed": args.seed,
                "episodes_per_point": args.episodes_per_point,
                "episode_horizon": args.episode_horizon,
            },
            "points": [
                {
                    "plan_horizon": point.plan_horizon,
                    "scorecard": asdict(point.scorecard),
                    "success_ci": [point.success_ci_low, point.success_ci_high],
                    "latency_ci_ms": [point.latency_ci_low, point.latency_ci_high],
                }
                for point in sweep.points
            ],
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote sweep report to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wmel",
        description=(
            "World Model Evaluation Lab - decision-oriented benchmark CLI. "
            "Run a benchmark, sweep a planning horizon, write a versioned JSON report."
        ),
    )
    subs = parser.add_subparsers(dest="cmd", required=True)

    # --- run ---
    p_run = subs.add_parser(
        "run",
        help="Run one benchmark (one env + one policy + one perturbation strategy).",
    )
    p_run.add_argument("--env", required=True, choices=sorted(ENVS.keys()))
    p_run.add_argument("--policy", required=True, choices=sorted(POLICIES.keys()))
    p_run.add_argument("--episodes", type=int, default=30)
    p_run.add_argument("--horizon", type=int, default=60, help="Episode horizon (max steps).")
    p_run.add_argument(
        "--plan-horizon",
        type=int,
        default=20,
        help="Planner lookahead. Used only by tabular-world-model.",
    )
    p_run.add_argument("--perturb-prob", type=float, default=0.0)
    p_run.add_argument(
        "--perturbation",
        default="env-default",
        help="Perturbation strategy: env-default | drop-next-K | composite:A+B.",
    )
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--output", help="Write a versioned JSON report to this path.")
    p_run.set_defaults(func=cmd_run)

    # --- sweep ---
    p_sweep = subs.add_parser(
        "sweep",
        help="Run a planning-horizon sweep on one env with one policy.",
    )
    p_sweep.add_argument("--env", required=True, choices=sorted(ENVS.keys()))
    p_sweep.add_argument(
        "--policy",
        default="tabular-world-model",
        choices=["tabular-world-model"],
        help="Sweep currently supports only tabular-world-model.",
    )
    p_sweep.add_argument(
        "--plan-horizons",
        default="5,10,15,20,30",
        help="Comma-separated list of plan horizons to sweep over.",
    )
    p_sweep.add_argument("--episodes-per-point", type=int, default=30)
    p_sweep.add_argument("--episode-horizon", type=int, default=80)
    p_sweep.add_argument("--perturb-prob", type=float, default=0.0)
    p_sweep.add_argument(
        "--perturbation",
        default="env-default",
        help="Perturbation strategy: env-default | drop-next-K | composite:A+B.",
    )
    p_sweep.add_argument("--seed", type=int, default=0)
    p_sweep.add_argument("--output", help="Write a versioned JSON sweep report to this path.")
    p_sweep.set_defaults(func=cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
