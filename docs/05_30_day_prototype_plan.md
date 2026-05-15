# 05 - 30-Day Prototype Plan

A focused four-week plan that takes the repository from "credible scaffolding" to "shareable product narrative".

The plan assumes one contributor, part-time. Every week ends with a runnable artifact, not just a doc.

---

## Week 1 - Repository and toy benchmark

**Goal**: a public-facing scaffold a stranger can clone, install, and run on a laptop.

- Land the repository structure, README, AGENTS.md, MIT license, and 5 doc pages.
- Implement the `BenchmarkEnvironment` and `PlannerPolicy` interfaces.
- Implement the Two-Room toy environment with optional perturbation.
- Implement random and greedy baselines.
- Ship `examples/two_room_toy/run_baseline.py` and a sample JSON report.
- Tests cover metrics on synthetic results and the toy environment.

**Exit criterion**: `pytest` is green and `python -m examples.two_room_toy.run_baseline` prints a scorecard from a clean checkout.

---

## Week 2 - Adapter interface and scorecards

**Goal**: prove the evaluation layer is model-agnostic.

- Tighten `PlannerPolicy` and `BenchmarkEnvironment` to be the only contract a model needs to implement.
- Add the `LeWMAdapterStub` as a documented contract example (does **not** import or reimplement any specific model).
- Add a second toy environment that exercises planning horizon (a small maze).
- Introduce the `Scorecard` dataclass and a human-readable text formatter.
- Add the `to_json_report` reporter and snapshot one example for the docs.

**Exit criterion**: a developer can implement a new `PlannerPolicy` in under 50 lines and produce a scorecard without touching the rest of the codebase.

**Status**: shipped in v0.2 - see `examples/maze_toy/` and `src/wmel/adapters/tabular_world_model.py`. `BenchmarkEnvironment` now exposes `action_space`. `TabularWorldModelPlanner` is a concrete `LeWMAdapterStub` subclass that fills in `encode`, `rollout`, `score`, and `plan` end-to-end with no third-party dependency.

---

## Week 3 - Baseline comparison and reporting

**Goal**: make the comparison story sharp.

- Add Compute per Decision and Planning Horizon to the scorecard.
- Add deterministic seeds and basic confidence-interval reporting on success metrics.
- Add a perturbation library: at minimum displacement, blocked-cell, and delayed-action perturbations.
- Add a Markdown report exporter so a scorecard can land in a doc page directly.
- Author one benchmark card in code, not just in prose - the toy maze with full scorecard output.

**Exit criterion**: a single command produces a Markdown report comparing two policies on two environments with confidence intervals.

**Status (partial, v0.3)**: planning-horizon sweep with Wilson and normal confidence intervals is shipped - see `wmel.experiments.horizon_sweep`, `examples/maze_toy/run_horizon_sweep.py`, and the worked example in `docs/02_metric_taxonomy.md`. The maze run reproduces the expected textbook curve: 0% at horizon 5, 100% at horizon 15+, with latency rising again past the plateau. Markdown export, compute-per-decision profiling, and a broader perturbation library remain.

---

## Week 4 - Public demo and product narrative

**Goal**: make the artifact persuasive to a non-researcher.

- Write a short blog-style page (`docs/06_demo.md`) walking through one scorecard and what it implies for a product decision.
- Record a 90-second screen capture of the toy benchmark running and the report being read. (Optional, do not block on it.)
- Tighten the README into a 60-second pitch.
- Add a `CONTRIBUTING.md` describing how to add a benchmark card, a metric, and an adapter.
- Tag `v0.1.0` and write release notes that explicitly state the non-affiliation disclaimer.

**Exit criterion**: a product person can read the README, run the demo, and articulate the thesis without help.

---

## What is explicitly out of scope for the first 30 days

- Training any model.
- Downloading any dataset or checkpoint.
- Adding any GPU dependency.
- Implementing Push-T, Reacher, or OGBench Cube fully - they remain benchmark cards until v0.2.
- Hosting a public scoreboard.

These are deliberate omissions. The whole point of the wedge is that the **evaluation layer** is what is missing, not yet another model.
