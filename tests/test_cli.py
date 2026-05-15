"""Tests for the `wmel` console script.

Coverage targets:
- argparse wiring (subcommands, required flags)
- env / policy / perturbation registry resolution
- end-to-end run command produces a scorecard and a valid JSON envelope
- sweep command writes a versioned JSON sweep report
- error paths (unknown perturbation spec, tabular-WM on env without dynamics)
"""

from __future__ import annotations

import json

import pytest

from wmel.cli import _build_perturbation, build_parser, main


def test_build_parser_lists_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "run" in help_text
    assert "sweep" in help_text


def test_build_perturbation_env_default() -> None:
    from wmel.perturbations import EnvPerturbation

    p = _build_perturbation("env-default")
    assert isinstance(p, EnvPerturbation)


def test_build_perturbation_drop_next() -> None:
    from wmel.perturbations import DropNextActions

    p = _build_perturbation("drop-next-3")
    assert isinstance(p, DropNextActions)
    assert p.name == "drop-next-3"


def test_build_perturbation_composite() -> None:
    from wmel.perturbations import CompositePerturbation

    p = _build_perturbation("composite:env-default+drop-next-2")
    assert isinstance(p, CompositePerturbation)
    assert "env-default" in p.name
    assert "drop-next-2" in p.name


def test_build_perturbation_rejects_unknown_spec() -> None:
    with pytest.raises(SystemExit):
        _build_perturbation("does-not-exist")


def test_build_perturbation_rejects_malformed_drop_next() -> None:
    with pytest.raises(SystemExit):
        _build_perturbation("drop-next-not-an-int")


def test_cli_run_writes_versioned_json(tmp_path) -> None:
    """End-to-end: `wmel run` on the maze produces a schema-versioned JSON."""
    out_path = tmp_path / "report.json"
    rc = main([
        "run",
        "--env", "maze_toy",
        "--policy", "random",
        "--episodes", "3",
        "--horizon", "20",
        "--seed", "0",
        "--output", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    envelope = json.loads(out_path.read_text())
    assert envelope["schema_version"] == "1.0"
    assert "wmel_version" in envelope
    assert "generated_at" in envelope
    assert envelope["scorecard"]["policy_name"] == "random"
    assert envelope["scorecard"]["episodes"] == 3
    assert envelope["metadata"]["env"] == "maze_toy"
    assert envelope["metadata"]["perturbation"] == "env-default"


def test_cli_run_with_drop_next_perturbation(tmp_path) -> None:
    """Custom perturbation string round-trips through to the report metadata."""
    out_path = tmp_path / "report.json"
    rc = main([
        "run",
        "--env", "maze_toy",
        "--policy", "random",
        "--episodes", "3",
        "--horizon", "20",
        "--perturb-prob", "1.0",
        "--perturbation", "drop-next-2",
        "--seed", "0",
        "--output", str(out_path),
    ])
    assert rc == 0
    envelope = json.loads(out_path.read_text())
    assert envelope["metadata"]["perturbation"] == "drop-next-2"
    assert envelope["scorecard"]["perturbation_name"] == "drop-next-2"


def test_cli_run_tabular_wm_on_two_room_fails_cleanly() -> None:
    """TwoRoomEnv does not expose a pure `dynamics` callable; the CLI must
    refuse to build a TabularWorldModelPlanner on it, not crash silently."""
    with pytest.raises(SystemExit):
        main([
            "run",
            "--env", "two_room_toy",
            "--policy", "tabular-world-model",
            "--episodes", "2",
        ])


def test_cli_sweep_writes_versioned_json(tmp_path) -> None:
    out_path = tmp_path / "sweep.json"
    rc = main([
        "sweep",
        "--env", "maze_toy",
        "--plan-horizons", "5,15",
        "--episodes-per-point", "3",
        "--episode-horizon", "30",
        "--seed", "0",
        "--output", str(out_path),
    ])
    assert rc == 0
    envelope = json.loads(out_path.read_text())
    assert envelope["schema_version"] == "1.0"
    assert "wmel_version" in envelope
    assert envelope["policy_name"] == "tabular-world-model"
    assert len(envelope["points"]) == 2
    assert envelope["points"][0]["plan_horizon"] == 5
    assert envelope["points"][1]["plan_horizon"] == 15
    assert envelope["metadata"]["perturbation"] == "env-default"


def test_cli_sweep_rejects_unsupported_policy() -> None:
    """argparse rejects --policy random on the sweep subcommand (sweep is
    tabular-world-model only). Asserting the argparse exit code (2) makes
    sure the rejection happens at the parser layer, not in dead body code."""
    with pytest.raises(SystemExit) as excinfo:
        main([
            "sweep",
            "--env", "maze_toy",
            "--policy", "random",
            "--plan-horizons", "5,10",
        ])
    assert excinfo.value.code == 2


def test_cli_run_rejects_perturb_prob_out_of_range() -> None:
    """A probability above 1.0 (or below 0.0) is a silent-bias trap.
    The CLI must refuse it, not pass it to the runner."""
    with pytest.raises(SystemExit) as excinfo:
        main([
            "run",
            "--env", "maze_toy",
            "--policy", "random",
            "--episodes", "2",
            "--perturb-prob", "1.5",
        ])
    assert "perturb-prob" in str(excinfo.value)

    with pytest.raises(SystemExit) as excinfo:
        main([
            "run",
            "--env", "maze_toy",
            "--policy", "random",
            "--episodes", "2",
            "--perturb-prob", "-0.5",
        ])
    assert "perturb-prob" in str(excinfo.value)


def test_cli_sweep_rejects_malformed_plan_horizons() -> None:
    """A typo in --plan-horizons should produce a clean CLI error, not a
    Python ValueError traceback."""
    with pytest.raises(SystemExit) as excinfo:
        main([
            "sweep",
            "--env", "maze_toy",
            "--plan-horizons", "5,a,15",
            "--episodes-per-point", "2",
        ])
    assert "plan-horizons" in str(excinfo.value)
    assert "'a'" in str(excinfo.value)


def test_cli_sweep_rejects_non_positive_horizon() -> None:
    """Horizons must be positive; zero or negatives produce no useful run."""
    with pytest.raises(SystemExit):
        main([
            "sweep",
            "--env", "maze_toy",
            "--plan-horizons", "5,0,15",
            "--episodes-per-point", "2",
        ])


def test_cli_run_greedy_solves_two_room_with_waypoint() -> None:
    """Regression: building greedy for two_room_toy must inject the doorway
    waypoint hint, or the policy silently regresses to 0% success."""
    from wmel.cli import _build_greedy_policy
    from wmel.envs.two_room_toy import TwoRoomEnv

    env = TwoRoomEnv()
    policy = _build_greedy_policy(env, seed=0)
    # Run a short benchmark; greedy with the waypoint should solve the
    # two-room maze almost always.
    from wmel.benchmark_runner import BenchmarkRunner
    from wmel.metrics import action_success_rate

    results = BenchmarkRunner(
        env_factory=TwoRoomEnv,
        policy=policy,
        episodes=5,
        horizon=40,
        perturb_prob=0.0,
        seed=0,
    ).run()
    assert action_success_rate(results) == 1.0, (
        "greedy without waypoint regresses to 0% on two_room; the CLI "
        "must inject the doorway hint when building the policy"
    )
