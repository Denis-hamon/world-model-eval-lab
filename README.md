# World Model Evaluation Lab

[![tests](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> A product-oriented benchmark framework for evaluating action-conditioned world models beyond static AI benchmarks.

## Thesis

World models should not only be evaluated on prediction quality. They should be evaluated on their ability to **plan, recover, generalize, and make useful decisions under compute constraints**.

The next bottleneck for world models is not only model quality. It is proof of usefulness.

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
- A clear taxonomy of **product-grade** metrics (success rate, planning latency, compute per decision, perturbation recovery, etc.).
- Reusable benchmark cards that map academic tasks to industrial product questions.
- An adapter interface so any world model (research or proprietary) can be plugged in and scored.
- A runnable toy benchmark that demonstrates the full evaluation loop end-to-end.

## What this repo is not

- Not a reimplementation of any existing world model (including LeWorldModel).
- Not affiliated with AMI, Meta, Yann LeCun, or the LeWorldModel authors.
- Not a training pipeline. No checkpoints, no datasets, no GPU required.
- Not a production benchmark suite. It is an early, opinionated scaffold meant to grow.

## Architecture

```
 Observation -> Encoder -> Latent State -> Action-conditioned Predictor -> Future Latent State -> Planner -> Action
```

The repo treats every world model as a black box exposing this contract. The evaluation layer measures what happens **after** the action is taken, in an environment, under product-relevant conditions.

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

The two-room run compares a random policy and a greedy policy on a 2D two-room environment.

The maze run pits random and naive greedy against a `TabularWorldModelPlanner` - a concrete subclass of `LeWMAdapterStub` that fills in `encode`, `rollout`, `score`, and `plan` end-to-end. Naive greedy fails (gets stuck on walls), random fails (too slow), and the world-model planner reaches the goal at the cost of higher planning latency.

The horizon sweep takes the same world-model planner and runs it at several lookahead depths, producing the success-rate / latency curve that operationalises the "Planning Horizon" metric. See [docs/02_metric_taxonomy.md](docs/02_metric_taxonomy.md) for the worked example.

JSON reports are saved next to each script.

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
├── docs/
│   ├── 00_thesis.md
│   ├── 01_product_wedge.md
│   ├── 02_metric_taxonomy.md
│   ├── 03_benchmark_cards.md
│   ├── 04_industrial_use_cases.md
│   └── 05_30_day_prototype_plan.md
├── src/wmel/
│   ├── metrics.py
│   ├── benchmark_runner.py
│   ├── experiments.py
│   ├── report.py
│   └── adapters/
│       ├── base.py
│       ├── random_policy.py
│       ├── greedy_policy.py
│       ├── lewm_adapter_stub.py
│       └── tabular_world_model.py
├── examples/
│   ├── two_room_toy/
│   │   ├── environment.py
│   │   ├── run_baseline.py
│   │   └── README.md
│   └── maze_toy/
│       ├── environment.py
│       ├── run_baseline.py
│       └── README.md
├── tests/
├── pyproject.toml
├── LICENSE
├── .gitignore
├── CONTRIBUTING.md
└── .github/workflows/tests.yml
```

## Roadmap

- **v0.1**: two-room benchmark, random and greedy baselines, scorecard, JSON report.
- **v0.2**: maze benchmark, concrete `TabularWorldModelPlanner` subclass of `LeWMAdapterStub`, tightened `BenchmarkEnvironment` contract (`action_space`).
- **v0.3.1**: `horizon_sweep` experiment with Wilson and normal confidence intervals; per-call planning latency; honest perturbation accounting; regression tests for both metric invariants.
- **v0.4** (current): Markdown reporting (`to_markdown_scorecard`, `to_markdown_report`, `to_markdown_horizon_sweep`); compute-per-decision wired via `PlannerPolicy.compute_per_plan_call`.
- **v0.5**: pluggable perturbation library (displacement / blocked-cell / delayed-action), scorecard CLI.
- **v0.6**: adapter for a real research world model (via stub interface), public scoreboard format.

## Disclaimer

This is an independent research-to-product exploration. It is **not** an official artifact of AMI, Meta, the LeWorldModel project, or any of their authors. Any references to JEPA-style or LeWorldModel concepts are conceptual, not affiliational.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, code style, and the workflow for adding a metric, benchmark card, or adapter.

## License

MIT - see [LICENSE](LICENSE).
