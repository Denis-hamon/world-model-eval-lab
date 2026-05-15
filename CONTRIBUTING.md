# Contributing

Thanks for your interest. This repository is an independent study of evaluation methodology for action-conditioned world models. Contributions that sharpen the **evaluation layer** (better metrics, better benchmarks, better reporting) are very welcome. Contributions that turn it into a training framework or a model zoo are out of scope.

If you have not read it yet, `AGENTS.md` lists the hard rules: no affiliation claims, no reimplementation of LeWorldModel, no heavyweight ML dependencies in the runtime, no GPU requirement.

## Dev setup

Requires Python 3.11+.

```bash
git clone https://github.com/Denis-hamon/world-model-eval-lab.git
cd world-model-eval-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m examples.maze_toy.run_horizon_sweep
```

If the test suite is green and the maze sweep prints a scorecard, you are set up.

## Code style

- Python 3.11+, type hints on every public function and class.
- `dataclass` and `typing` over hand-rolled classes.
- 4-space indent, no tabs, no trailing whitespace.
- Module docstrings on every file. One-line docstrings on small helpers, paragraph docstrings on public APIs.
- No emojis in code, output, or docs.
- No third-party runtime dependencies (`pytest` is dev-only). If you genuinely need one, justify it in the PR description.

## Adding a metric

1. Open `docs/02_metric_taxonomy.md` and add a row to the summary table: **definition**, **why it matters**, **example measurement**.
2. Implement the metric in `src/wmel/metrics.py` as a function over `Sequence[EpisodeResult]`. Return `None` when the metric is meaningless (no episodes, no successes, etc.) rather than fabricating a number.
3. Wire it into `compute_scorecard` if it belongs on the standard scorecard.
4. Add a unit test in `tests/test_metrics.py` against synthetic `EpisodeResult` data. Include the edge case where the metric should be `None`.
5. If the metric needs new fields on `EpisodeResult`, add them as keyword-only dataclass fields with sensible defaults so existing callers keep working.

## Adding a benchmark card

1. Open `docs/03_benchmark_cards.md`. Use the existing card structure: **task type**, **applied interpretation**, **relevant industries**, **world model value hypothesis**, **candidate metrics**, **applied question**.
2. If you also implement the environment, drop it in `examples/<env_name>/environment.py`. The environment must subclass `wmel.adapters.base.BenchmarkEnvironment` and expose `action_space`.
3. Add a `run_baseline.py` that compares at least a random and one other policy, and writes a `sample_report.json`.
4. Whitelist the new sample report in `.gitignore` if you want it tracked.
5. Add tests in `tests/test_<env_name>.py`. At minimum: reset works, success is reachable, the non-trivial baseline beats random under a fixed seed.

## Adding a perturbation

`wmel.perturbations.Perturbation` has two override hooks:

- `apply_to_env(env) -> None` — mutate environment state at the trigger moment. Default: no-op.
- `transform_actions(remaining_actions) -> list[Action]` — return a possibly-transformed copy of the action queue. Default: pass-through copy.

A new perturbation subclasses `Perturbation`, sets `name`, and overrides whichever hook is relevant. State-level perturbations override `apply_to_env`; action-level perturbations (actuator drops, delays, replacements) override `transform_actions`; combined failure modes can either be a custom class or a `CompositePerturbation` of existing parts.

Checklist:

1. Subclass `wmel.perturbations.Perturbation`. Set `name` and override at least one hook.
2. If your perturbation has parameters (e.g., `DropNextActions(k=...)`), validate them in `__init__` and raise `ValueError` on nonsense.
3. Add unit tests in `tests/test_perturbations.py` covering: the hook(s) you override, the unchanged hook, parameter validation, and (if action-level) the "drop more than what's there" edge case.
4. If your perturbation needs env-specific support (e.g., blocked-cell needs an env method), document the contract clearly. Do not silently no-op when the env lacks the method - raise or fall back explicitly.

## Adding an adapter (planner policy)

1. Subclass `wmel.adapters.base.PlannerPolicy`. Implement `plan(observation, goal, horizon)` and `name`.
2. If your adapter is a learned world model, subclass `LeWMAdapterStub` instead and implement `encode`, `rollout`, `score`. See `TabularWorldModelPlanner` for the smallest concrete example.
3. If you can declare a `compute_per_plan_call` estimate, set it on the class or instance. The scorecard will derive a `compute_per_decision` figure automatically.
4. Add unit tests covering `encode` (identity-or-not), `rollout` (uses the dynamics), `score` (defaults to Manhattan unless overridden), and `plan` (reaches a goal in a trivial open environment).

## Testing requirements

- All new code paths must have at least one test.
- Use deterministic seeds. No randomness without a `seed` parameter.
- Tests must run in under a second each. The full suite must stay under 5 seconds on CPU.
- No network access from tests. No filesystem access outside `examples/<env>/sample_report.json`.

## Pull request process

1. Branch off `main`.
2. Run `pytest -q` and `python -m examples.maze_toy.run_horizon_sweep` locally before opening the PR.
3. In the PR body, include the rendered Markdown of any scorecard the PR affects (use `wmel.report.to_markdown_scorecard`).
4. CI must be green.
5. If the PR introduces a new metric, benchmark card, adapter, or doc page, mention it in the next release notes draft.

## Disclaimer

This is an independent project. Contributions must preserve the non-affiliation stance toward AMI, Meta, the LeWorldModel project, and any of their authors. JEPA-style and LeWorldModel references are conceptual only.
