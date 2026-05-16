# World Model Evaluation Lab

[![tests](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![live site](https://img.shields.io/badge/live%20site-%E2%86%92-0f5fbf)](https://denis-hamon.github.io/world-model-eval-lab/)

> A decision-oriented benchmark framework for evaluating action-conditioned world models beyond static AI benchmarks.

## Thesis

World models should not only be evaluated on prediction quality. They should be evaluated on their ability to **plan, recover, generalize, and make useful decisions under compute constraints**.

The next bottleneck for world models is not only model quality. It is proof of usefulness.

**Where to start**: if you have ten minutes, read [docs/06_demo.md](docs/06_demo.md) - a walkthrough of one scorecard and what a non-researcher should read out of it. If you have more, the rest of [docs/](docs/) covers the thesis, the metric taxonomy, the benchmark cards, and a 30-day study plan.

## Why this matters

Research on action-conditioned world models (JEPA-style predictors, latent dynamics models, video world models) is advancing quickly. Most public evaluation, however, still relies on:

- pixel reconstruction error,
- frame-level FID/PSNR,
- offline next-frame prediction loss,
- or fixed static AI benchmarks.

These metrics tell us how well a model **predicts**. They do not tell us whether the model is **useful** for downstream tasks like robotic manipulation, industrial control, datacenter operations, or warehouse routing.

This repository proposes a thin, opinionated evaluation layer that closes that gap.

## What this repo is

- A lightweight, CPU-only benchmark scaffolding for action-conditioned world models.
- A clear taxonomy of **decision-grade** metrics (success rate, planning latency, compute per decision, perturbation recovery, etc.).
- Reusable benchmark cards that map academic tasks to industrial applied questions.
- An adapter interface so any world model (research or proprietary) can be plugged in and scored.
- A runnable toy benchmark that demonstrates the full evaluation loop end-to-end.

## What this repo is not

- Not a reimplementation of any existing world model (including LeWorldModel).
- Not affiliated with AMI, Meta, Yann LeCun, or the LeWorldModel authors.
- Not a training pipeline. No checkpoints, no datasets, no GPU required.
- Not a production benchmark suite. It is an early, opinionated scaffold meant to grow.

## Architecture

![architecture](docs/assets/architecture.svg)

The repo treats every world model as a black box exposing this contract. The evaluation layer measures what happens **after** the action is taken, in an environment, under decision-relevant conditions.

## Evaluation levels

1. **Representation** - does the latent state preserve task-relevant structure?
2. **Planning** - does the model produce action sequences that solve the task within a horizon and a compute budget?
3. **Product value** - does the model recover from perturbations, generalize across tasks, and operate within latency and cost constraints that a real product would require?

## Quickstart

Requires Python 3.11+.

```bash
pip install -e ".[dev]"
python -m examples.two_room_toy.run_baseline
python -m examples.maze_toy.run_baseline
python -m examples.maze_toy.run_horizon_sweep
pytest
```

Or via the `wmel` console script (installed alongside the package):

```bash
wmel run --env maze_toy --policy tabular-world-model --episodes 30 --output run.json
wmel sweep --env maze_toy --plan-horizons 5,10,15,20,30 --output sweep.json
```

Both write versioned JSON reports (`schema_version: "1.0"`) carrying the wmel version, an ISO-8601 timestamp, and a metadata block describing the run's env, policy, perturbation, and seed.

The two-room run compares a random policy and a greedy policy on a 2D two-room environment.

The maze run pits random and naive greedy against a `TabularWorldModelPlanner` - a concrete subclass of `LeWMAdapterStub` that fills in `encode`, `rollout`, `score`, and `plan` end-to-end. Naive greedy fails (gets stuck on walls), random fails (too slow), and the world-model planner reaches the goal at the cost of higher planning latency.

The horizon sweep takes the same world-model planner and runs it at several lookahead depths, producing the success-rate / latency curve that operationalises the "Planning Horizon" metric. See [docs/02_metric_taxonomy.md](docs/02_metric_taxonomy.md) for the worked example.

JSON reports are saved next to each script.

## Planning-horizon sweep on the maze toy

![horizon sweep](docs/assets/horizon_sweep.svg)

Success rate plateaus at horizon 15. Per-call latency keeps rising past the plateau. Compute per decision is bounded around 250-370 rollout-units. The same scorecard structure applies to every benchmark card in [docs/03_benchmark_cards.md](docs/03_benchmark_cards.md).

## Initial metrics

- Action Success Rate
- Average Steps to Success
- Average Planning Latency (ms)
- Compute per Decision
- Planning Horizon
- Perturbation Recovery Rate
- Sample Efficiency
- Surprise Detection
- Latent Interpretability

See [docs/02_metric_taxonomy.md](docs/02_metric_taxonomy.md) for definitions and example measurements.

## Repository layout

```
world-model-eval-lab/
├── README.md
├── AGENTS.md
├── CONTRIBUTING.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── index.md                  # Pages landing
│   ├── _config.yml
│   ├── _layouts/default.html
│   ├── assets/                   # SVG illustrations + CSS + JS
│   ├── 00_thesis.md
│   ├── 01_evaluation_gap.md
│   ├── 02_metric_taxonomy.md
│   ├── 03_benchmark_cards.md
│   ├── 04_industrial_use_cases.md
│   ├── 05_30_day_prototype_plan.md
│   └── 06_demo.md
├── src/wmel/
│   ├── cli.py                    # `wmel run`, `wmel sweep`
│   ├── metrics.py
│   ├── benchmark_runner.py
│   ├── experiments.py
│   ├── perturbations.py          # Perturbation, EnvPerturbation, ...
│   ├── report.py
│   └── adapters/
│       ├── base.py
│       ├── random_policy.py
│       ├── greedy_policy.py
│       ├── lewm_adapter_stub.py
│       ├── tabular_world_model.py
│       └── learned_dynamics_torch.py   # optional, needs [learned] extra
├── examples/
│   ├── two_room_toy/
│   │   ├── environment.py
│   │   ├── run_baseline.py
│   │   └── README.md
│   └── maze_toy/
│       ├── environment.py
│       ├── run_baseline.py
│       ├── run_horizon_sweep.py
│       ├── run_learned_baseline.py     # optional, [learned] extra
│       ├── run_learned_sweep.py        # optional, [learned] extra
│       └── README.md
├── scripts/
│   └── render_visuals.py         # stdlib SVG generator for the Pages site
├── tests/                        # 80+ tests; learned ones import-or-skip
└── .github/
    ├── workflows/tests.yml
    ├── CODEOWNERS
    ├── SECURITY.md
    ├── dependabot.yml
    ├── PULL_REQUEST_TEMPLATE.md
    └── ISSUE_TEMPLATE/
```

## Roadmap

- **v0.1**: two-room benchmark, random and greedy baselines, scorecard, JSON report.
- **v0.2**: maze benchmark, concrete `TabularWorldModelPlanner` subclass of `LeWMAdapterStub`, tightened `BenchmarkEnvironment` contract (`action_space`).
- **v0.3.1**: `horizon_sweep` experiment with Wilson and normal confidence intervals; per-call planning latency; honest perturbation accounting; regression tests for both metric invariants.
- **v0.4**: Markdown reporting (`to_markdown_scorecard`, `to_markdown_report`, `to_markdown_horizon_sweep`); compute-per-decision wired via `PlannerPolicy.compute_per_plan_call`.
- **v0.5**: pluggable perturbation library (`Perturbation`, `EnvPerturbation`, `DropNextActions`, `CompositePerturbation`); runner accepts custom perturbations via a `perturbation` kwarg; `Scorecard.perturbation_name` records which strategy was used; runner inner loop switched to `deque` for O(1) action-queue pops.
- **v0.6**: proof-of-contract for learned dynamics. `wmel.adapters.learned_dynamics_torch` ships a PyTorch MLP that fits the maze's transition table and plugs into `TabularWorldModelPlanner` as a drop-in `dynamics` callable. Identical success rate to the oracle, 76x higher per-call latency - exactly the trade-off the framework is built to expose. PyTorch is an optional dependency (`pip install -e ".[learned]"`); core runtime stays stdlib-only.
- **v0.7**: `wmel` CLI (`wmel run`, `wmel sweep`) installed as a console script; `horizon_sweep` accepts a `Perturbation` argument; JSON reports versioned via a `schema_version` envelope with `wmel_version`, `generated_at`, and an extensible `metadata` block; second CI job `test-stdlib-only` locks in the no-torch runtime promise.
- **v0.8**: first non-toy environment - DeepMind Control Suite Acrobot-swingup wired in via `wmel.envs.dmc_acrobot`, with a `make_acrobot_oracle_dynamics()` factory and a Markovian MLP learned dynamics, both plugging into the same random-shooting MPC planner. `dm-control` is an optional extra (`pip install -e ".[control]"`).
- **v0.9**: Counterfactual Planning Gap (CPG) metric: a five-branch verdict gated on an Agresti--Caffo plus-4 $95\%$ CI rather than the raw point estimate, computed by running the same planner against the oracle and the learned dynamics. The Acrobot worked example reports the honest `INCONCLUSIVE` verdict at $n = 10$ instead of over-claiming.
- **v0.10** (current): short paper. `paper/` ships the LaTeX source, BibTeX bibliography, a stdlib `build_figures.py` that reproduces the table values from `results/dmc_acrobot/cpg.json`, and a Makefile. The paper is intentionally narrow: one env, one model, one seed, ten episodes per arm; the methodological contribution is decoupled from any of those choices.

## Paper

A short paper accompanies the framework: **"Counterfactual Planning Gap: A Decision-Grade Metric for Decoupling Model Error from Planner Capacity in World Model Evaluation"**. LaTeX source in [`paper/`](paper/). It formalises the CPG metric (definition, Agresti--Caffo confidence interval, gated verdict), defines the four-method evaluation contract, and reports the worked example on DMC Acrobot-swingup with the honest `INCONCLUSIVE` verdict at $n = 10$. Build with `make` inside `paper/` after installing `texlive-latex-extra` and `texlive-bibtex-extra`.

## Related work

This repository sits next to a substantial body of published work on world models (Dreamer-V3, MuZero, IRIS, Genie / Genie 2), joint-embedding predictive architectures (I-JEPA, V-JEPA / V-JEPA 2), action-conditioned benchmarks (DeepMind Control Suite, OGBench, LIBERO), and evaluation methodology (rliable, Wilson intervals, "Deep RL that Matters"). See the full list with one-line annotations under [docs/00_thesis.md#related-work](docs/00_thesis.md#related-work).

## Affiliation

Independent study. See the [disclaimer at the bottom of the Pages site](https://denis-hamon.github.io/world-model-eval-lab/#disclaimer) for the canonical statement.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, code style, and the workflow for adding a metric, benchmark card, or adapter.

## License

MIT - see [LICENSE](LICENSE).
