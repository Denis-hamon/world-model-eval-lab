---
prev:
  title: "04 - Industrial use cases"
  url: 04_industrial_use_cases.html
next:
  title: "06 - Reading a scorecard"
  url: 06_demo.html
---
# 05 - 30-Day Prototype Plan

A focused four-week plan that takes the repository from "credible scaffolding" to "shareable applied narrative".

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

**Status (v0.4)**:

- Planning-horizon sweep with Wilson and normal confidence intervals - see `wmel.experiments.horizon_sweep`, `examples/maze_toy/run_horizon_sweep.py`, and the worked example in `docs/02_metric_taxonomy.md`.
- Markdown report exporters: `wmel.report.to_markdown_scorecard`, `to_markdown_report`, and `wmel.experiments.to_markdown_horizon_sweep`. The output is paste-ready in a PR body or doc page.
- Compute-per-decision wired: `PlannerPolicy.compute_per_plan_call` is a class attribute that subclasses set. `TabularWorldModelPlanner` declares it as `num_candidates * plan_horizon`, and `compute_scorecard` derives an average over the run. The maze baseline reports ~256 rollout-units per decision for the world-model planner versus n/a for random / greedy.

What remains for v0.5: a pluggable perturbation library (displacement, blocked-cell, delayed-action) and a small CLI front-end.

**Status (v0.5)**: perturbation library shipped. `wmel.perturbations` defines a `Perturbation` ABC with two override hooks (`apply_to_env`, `transform_actions`) and three concrete subclasses: `EnvPerturbation` (delegates to `env.perturb()`, the runner's default), `DropNextActions(k)` (action-level drop), and `CompositePerturbation(*parts)` (composable). The runner's inner loop was refactored to a `deque`-based action queue so action-level perturbations are O(1). `Scorecard.perturbation_name` records the strategy. The CLI front-end remains for a later release.

---

## Week 4 - Public demo and applied narrative

**Goal**: make the artifact persuasive to a non-researcher.

- Write a short blog-style page (`docs/06_demo.md`) walking through one scorecard and what it implies for an applied decision.
- Record a 90-second screen capture of the toy benchmark running and the report being read. (Optional, do not block on it.)
- Tighten the README into a 60-second pitch.
- Add a `CONTRIBUTING.md` describing how to add a benchmark card, a metric, and an adapter.
- Tag `v0.1.0` and write release notes that explicitly state the non-affiliation disclaimer.

**Exit criterion**: a non-researcher can read the README, run the demo, and articulate the thesis without help.

**Status**: `docs/06_demo.md` is shipped (a row-by-row product walkthrough of the maze horizon sweep). `CONTRIBUTING.md` is shipped. Tagged releases are at v0.3.1 and v0.4.0 with explicit non-affiliation disclaimers. CI runs the suite plus a smoke test of the three example scripts on Python 3.11/3.12/3.13. Screen capture remains optional and is not done.

---

## What is explicitly out of scope for the first 30 days

- Training any model.
- Downloading any dataset or checkpoint.
- Adding any GPU dependency.
- Implementing Push-T, Reacher, or OGBench Cube fully - they remain benchmark cards until v0.2.
- Hosting a public scoreboard.

These are deliberate omissions. The whole point of this study is that the **evaluation layer** is what is missing, not yet another model.

---

## Recipe: executing this plan with an LLM coding agent

This repository was built end-to-end with an LLM coding agent in the loop (Claude Code in this case; Codex, Cursor, Aider, or any equivalent agent that can read files, run shell, edit code, and spawn sub-agents works the same way). The bottleneck was never "the model cannot code". It was **specification clarity** and **review discipline**.

The recipe below is reproducible. Each step has been used to ship the v0.1 -> v0.5 cycle of this repo.

### 0. Setup

- Pick a coding agent that can read files, run shell, edit code, and spawn sub-agents.
- Make sure the agent has read access to `AGENTS.md` (hard rules) and `CONTRIBUTING.md` (style + extension procedure). Both files exist for the agent as much as for a human contributor.
- Set up a working directory and confine the agent to it. Do not let it modify `~` or system files.
- Decide upfront: which user actions you will run by hand (creating remote repos, force-pushes, releases) versus which the agent may take autonomously (file edits, commits, pushes to a feature branch).

### 1. Hand it the week, not the task

Each week of this plan is a stand-alone brief. Drop the entire week's section into the agent verbatim and ask:

> Plan the implementation. List every file you will create or modify. Do not write code yet. When you list a metric or a test, state the invariant it locks in.

Approve the plan only when it lists the same files you would. If the agent proposes "improvements" outside the week's scope, push back. **Scope creep is the failure mode**, not under-delivery.

### 2. Implementation in one round

Once the plan is approved:

> Implement the plan. Write tests alongside each new module. After every file is written, run `pytest -q` and any example scripts the README mentions. Report failures rather than working around them.

A good agent will use a visible todo list, run tests before claiming success, and surface unexpected behaviour rather than papering over it.

### 3. Pre-tag adversarial review (the step that paid off 3-for-3)

Before tagging any release, spawn an **independent** review agent with a fresh context. The independence is what makes this work; the agent does not inherit your implementation rationalisations.

Prompt template:

> Adversarially review the diff at HEAD against the brief above. The previous review caught [insert prior failure modes; for this repo it was per-call vs per-episode latency confusion, perturbation accounting overcounting the denominator, dead reporting paths, fragile test heuristics, missing invariants]. Look specifically for: math errors, doc/code mismatches, missing invariants, silent fallbacks, performance regressions. If you find nothing, say so explicitly. Output `severity (critical / major / minor) | one-line description | file:line evidence | suggested fix`.

Track record on this repo:

- **v0.3 review**: 2 real bugs (per-call latency, perturbation accounting) + 2 minors. Both bugs fixed in v0.3.1.
- **v0.4 review**: 2 majors (test docstring arithmetic wrong, missing compute column in sweep markdown) + 5 minors. All addressed before tag.
- **v0.5 review**: 6 minors, including a dead `Perturbation.name` in the reporting path and an O(n) `list.pop(0)` regression in the runner. All addressed before tag.

Zero releases shipped with metric-correctness bugs after this pattern was adopted.

### 4. Tag, push, release

Only after review findings are addressed. Each tag corresponds to a working green-CI state. Release notes are drafted by the agent from the commit history plus the relevant doc-page updates, then read over by a human before publishing.

### 5. Soft passes (no version bump)

Documentation polishing, vocabulary tightening, rebranding, design-system changes, and Pages-only updates are **doc-only passes**. They do not need a version bump. They still deserve a single self-contained commit message that explains the why.

This repo has shipped several: the "soften framing" pass, the GitHub Pages landing, the IBM Plex design system, the interactive hero. None bumped the version.

### What to keep human-in-the-loop

The agent should not autonomously decide:

- **Strategic scope**: ship X or cut X. Especially "should this be a side project or be folded into the day job?".
- **Vocabulary that signals intent**: a repo positioned as a "product wedge" reads very differently from one positioned as a "methodology study". The agent will not catch this for you unless you ask it to.
- **Tagging decisions** and the human-facing release narrative.
- **Going public**: making the repo public, posting on social media, sharing internally. The agent can prepare the artifact; the human owns the broadcast.

### Anti-patterns this recipe explicitly avoids

- Asking the agent to "improve" the repo without a concrete target.
- Accepting code without tests for the invariants that actually matter (i.e. tests that would fail if the metric were silently wrong, not tests that just exercise the happy path).
- Skipping the adversarial review because `pytest` is green. Green tests do not catch doc/code mismatches or numerical confusion in unmeasured directions.
- Letting the agent rationalise a doc/code mismatch instead of fixing one side. The doc is usually right; if it is wrong, fix the doc and re-test the code.
- Bumping the version on doc-only passes. Releases should mean a real surface change.

### Reading order if you are setting this recipe up on a new project

1. This page (the four weekly briefs plus this recipe).
2. `AGENTS.md` (hard rules: scope, dependencies, non-affiliation, style).
3. `CONTRIBUTING.md` (how a new metric / benchmark card / adapter / perturbation should be structured).
4. One existing release commit message in `git log` (for example `v0.4.0` or `v0.5.0`) to calibrate the level of detail the agent should produce.

### How long does it actually take?

Per week, one focused sitting end-to-end (plan, implement, review, fix, commit, tag). Most of the wall-clock time is test runs and waiting on review cycles, not model think-time. The recipe favours **fewer, deeper cycles** over many shallow ones - exactly the discipline that catches metric-correctness bugs early.