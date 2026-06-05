# World Model Evaluation Lab

[![tests](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![live site](https://img.shields.io/badge/live%20site-%E2%86%92-0f5fbf)](https://denis-hamon.github.io/world-model-eval-lab/)

> A decision-oriented benchmark framework for evaluating action-conditioned world models beyond static AI benchmarks.

## Status (v0.18): task-level results, heterogeneous verdicts

> The [paper](paper/main.tex) is rewritten on **task-level** results: each CPG worked example samples the task's initial-state distribution (the two arms paired by start state, three seeds pooled), via the opt-in `--varied-init` harness (default off, so the original fixed-init results still reproduce; see [`experiments/RERUN_VARIED_INIT.md`](experiments/RERUN_VARIED_INIT.md)).
>
> The verdict is **heterogeneous** -- the config-sensitivity the metric is built to expose. **Acrobot** flips from `MODEL BOTTLENECK` to `PLANNER BOTTLENECK` (the oracle planner solves only ~3% of random starts, so the fixed-start gap was an artifact); **Reacher** holds `MODEL BOTTLENECK` (CPG +0.20 to +0.33); high-capacity **Cartpole** under CEM reaches `LEARNED OUTPERFORMS ORACLE` (CPG −0.27, AC and paired-bootstrap CIs both clearing zero). The headline is a **self-correction**: the metric's own interval-gated machinery overturned an earlier fixed-start result. (The Roadmap further below is historical and predates the task-level rewrite.)

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
- **v0.10**: short paper. `paper/` ships the LaTeX source, BibTeX bibliography, a stdlib `build_figures.py` that reproduces the table values from `results/dmc_acrobot/cpg.json`, and a Makefile.
- **v0.11**: the multi-seed extension promised in v0.9 and the paper's Section 5.5. `experiments/dmc_acrobot/cpg_sweep.py` pools three seeds at 50 episodes per arm per seed (n = 150 pooled) and sweeps the MLP's training-set size across `{200, 2 000, 20 000}`. The verdict hardens from `INCONCLUSIVE` to `MODEL BOTTLENECK` with an AC 95% CI of `[+0.191, +0.335]` *in every cell*; validation MSE drops by ~150x across the sweep while learned-arm success stays at zero. (Measured at a single fixed initial state -- the three seeds varied planner RNG, not the start state; a varied-initial-state re-run is in progress, see the Status note above.) Paper updated with Section 5.6 ("capacity vs. coverage") and a refreshed abstract and conclusion. The site landing and the CPG page surface the same multi-seed table.
- **v0.12**: **first published-world-model adapter shipped** (TD-MPC2). `src/wmel/adapters/tdmpc2_adapter.py` pairs TD-MPC2's encoder + latent dynamics with a small post-hoc obs decoder to expose a `(state, action) -> next_state` callable compatible with `TabularWorldModelPlanner`. *Result at n = 10, seed 0*: oracle 0.30, TD-MPC2 dynamics 0.00, MLP-on-TD-MPC2-data 0.00, CPG = +0.300 with AC CI `[-0.06, +0.56]` and verdict `INCONCLUSIVE` in all three arms - *despite* the coverage axis moving (random 0% upright → TD-MPC2 10.6%) and MLP val MSE dropping 5x. The framework correctly refused to convict "coverage alone" as the diagnosis.
- **v0.13**: **stronger planner + pooled-150** test of the v0.12 result. `src/wmel/adapters/cem_planner.py` plugs a Cross-Entropy Method MPC into the same dynamics contract. Under CEM, the oracle's success rate triples (`0.30 → 0.90`), *both* learned arms stay at `0/10`, and CPG opens to `+0.900` (CI `[+0.49, +1.01]`, `MODEL BOTTLENECK`). Pooled across three seeds at n = 150 per arm tightens this to CPG `+0.880`, CI `[+0.814, +0.923]`, half-width `0.054`. The gap is a dynamics-quality bottleneck the planner cannot close at this fixed initial state; a paired varied-initial-state re-run is pending (see the Status note above). Paper Section 5.8 "Robustness".
- **v0.14**: **in-episode perturbation robustness**. `experiments/dmc_acrobot/perturbation_cpg.py` sweeps `DropNextActions(k)` at $k \in \{0, 1, 5\}$ on the same three CEM arms. The `MODEL BOTTLENECK` verdict survives every cell. v0.14.1 adds the paper's first two figures (CPG-vs-data twin axis and uprightness coverage histogram). Paper Section 5.9.
- **v0.15** (current): **cross-environment validation**. `src/wmel/envs/dmc_cartpole.py` + a four-arm CPG matrix on DMC Cartpole-swingup at TD-MPC2 `model_size = 5` AND `model_size = 1`, three seeds pooled to n=30 per arm, 10⁶ env steps each. At `size = 5`, `MODEL BOTTLENECK` reproduces in all four cells; at `size = 1`, three of four cells stay at `MODEL BOTTLENECK` but the CEM × TD-MPC2 cell flips to **`INCONCLUSIVE`** (learned `0.533` vs oracle `0.500`, CPG `-0.033`, AC CI `[-0.28, +0.21]`) — first moderate-n `INCONCLUSIVE` verdict in the paper. First non-trivial learned-arm successes in the paper (`0.200`-`0.533` depending on planner/capacity). Planner-capacity asymmetry across envs: random-shooting outperforms CEM on Cartpole's oracle, inverting the Acrobot pattern. Paper Section 5.10 + Figures 3 and 4.

## What's next

The GPU experiment queue lives in [`experiments/GPU_ROADMAP.md`](experiments/GPU_ROADMAP.md). Task 1 (Cartpole size=1) is done and incorporated in v0.15. Three tasks remain:

1. **Horizon-of-planning ablation under CEM** — attributes the dynamics bottleneck to compounding error vs. distribution mismatch.
2. ~~**Cross-env to DMC Reacher-easy** — third env after Acrobot + Cartpole.~~ Done: landed in **v0.17.0** (`src/wmel/envs/dmc_reacher.py`).
3. **Cartpole size=5 pooled-150** — CI tightening (cosmetic).
4. **Observation-noise perturbation** — closes the §5.9 flagged limitation.

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
