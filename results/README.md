# results/

JSON / Markdown / SVG artifacts produced by the scripts under [`experiments/`](../experiments/). Each result is committed alongside the script that produced it so the artifact is reproducible from a clean checkout.

## Convention

Each result subdirectory matches an `experiments/<name>/` directory and contains:

- `scorecard.json` — the run's `Scorecard`, produced by `compute_scorecard(...)`. Carries a `schema_version` envelope (v0.7+).
- `sweep.json` (when applicable) — a `HorizonSweep` report.
- `*.md` — Markdown exports produced by `wmel.report.to_markdown_scorecard` and `wmel.experiments.to_markdown_horizon_sweep`, ready to drop into a paper or a PR.
- `*.svg` — figures regenerated from the JSON by `scripts/render_visuals.py` (or its successor).

## Status

**Currently empty.** Toy artifacts produced by the example scripts live alongside their scripts under [`examples/maze_toy/`](../examples/maze_toy/) and [`examples/two_room_toy/`](../examples/two_room_toy/) — see `sample_report.json`, `horizon_sweep_report.json`, `learned_baseline_report.json`, `learned_horizon_sweep_report.json`.

The first real artifact will land here once [Phase 1 of the post-v0.7 plan](../docs/05_30_day_prototype_plan.html) ships a DMC Acrobot adapter.
