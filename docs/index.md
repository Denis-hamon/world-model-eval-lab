---
layout: default
image: /assets/architecture.svg
next:
  title: "00 - Thesis"
  url: 00_thesis.html
---

<section class="hero">
  <div class="hero-copy">
    <h1>Evaluating world models <span class="accent">like they will ship</span></h1>
    <p class="hero-pitch">
      Static AI benchmarks measure how well a model <em>predicts</em>. They miss what an applied team actually needs to know: success rate, latency budget, compute cost, robustness under perturbation. This is a small, opinionated evaluation layer that closes that gap.
    </p>
    <blockquote class="hero-quote">
      The next bottleneck for world models is not only model quality. It is proof of usefulness.
    </blockquote>
    <div class="hero-cta">
      <a class="btn-primary" href="06_demo.html">Read the walkthrough</a>
      <a class="btn-ghost" href="https://github.com/Denis-hamon/world-model-eval-lab">Code on GitHub</a>
    </div>
  </div>
  <figure class="hero-figure">
    <img src="assets/maze.svg" alt="A 7x7 maze with an animated agent walking the optimal path from start to goal." />
    <figcaption>An agent walks the 7x7 maze. Optimal path = 14 actions. The world-model planner finds it in ~33 steps with replanning.</figcaption>
  </figure>
</section>

<ul class="stat-strip">
  <li><span class="stat-value">5</span><span class="stat-label">tagged releases</span></li>
  <li><span class="stat-value">81</span><span class="stat-label">passing tests</span></li>
  <li><span class="stat-value">CPU-only</span><span class="stat-label">no GPU required</span></li>
  <li><span class="stat-value">25 s</span><span class="stat-label">to reproduce the headline sweep</span></li>
  <li><span class="stat-value">0</span><span class="stat-label">ML dependencies at runtime</span></li>
</ul>

[![tests](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green)](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/LICENSE)

## Three policies on the maze, side by side
{:.reveal}

Same environment, same 30 episodes, same seed - three different planners. Numbers below are pulled verbatim from `examples/maze_toy/sample_report.json`, regenerated every time `python -m examples.maze_toy.run_baseline` is run.
{:.reveal}

<section class="policy-comparison reveal">
  <article class="policy-card policy-fail">
    <header>
      <h3>Random</h3>
      <p class="policy-tagline">Samples actions uniformly at random.</p>
    </header>
    <div class="big-number">0%</div>
    <p class="big-label">success rate over 30 episodes</p>
    <dl class="card-stats">
      <div><dt>latency / call</dt><dd>0.03 ms</dd></div>
      <div><dt>compute / decision</dt><dd>n/a</dd></div>
      <div><dt>verdict</dt><dd>Wanders near the start. Goal stays out of reach.</dd></div>
    </dl>
  </article>

  <article class="policy-card policy-fail">
    <header>
      <h3>Greedy (no waypoint)</h3>
      <p class="policy-tagline">Always step toward the goal in Manhattan distance.</p>
    </header>
    <div class="big-number">0%</div>
    <p class="big-label">success rate over 30 episodes</p>
    <dl class="card-stats">
      <div><dt>latency / call</dt><dd>0.001 ms</dd></div>
      <div><dt>compute / decision</dt><dd>n/a</dd></div>
      <div><dt>verdict</dt><dd>Walks into the wall. Plan diverges from env, stuck.</dd></div>
    </dl>
  </article>

  <article class="policy-card policy-success">
    <header>
      <h3>Tabular world model</h3>
      <p class="policy-tagline">Random-shooting MPC over a learned-style dynamics function.</p>
    </header>
    <div class="big-number">100%</div>
    <p class="big-label">success rate over 30 episodes</p>
    <dl class="card-stats">
      <div><dt>latency / call</dt><dd>3.12 ms</dd></div>
      <div><dt>compute / decision</dt><dd>~256 rollout-units</dd></div>
      <div><dt>verdict</dt><dd>Finds the corridor. Goal in ~34 steps (optimal is 14).</dd></div>
    </dl>
  </article>
</section>

<figure class="figure-wide reveal">
  <img src="assets/policy_comparison.svg" alt="Three side-by-side mini-mazes. The random agent wanders near the start; the greedy agent walks into the wall and stays stuck; the world-model agent finds the corridor and walks the optimal path to the goal." />
  <figcaption>Three agents, three outcomes, one shared evaluation contract. Each panel animates the agent of its policy in the same maze.</figcaption>
</figure>

The captured terminal output of the run that produced those numbers:
{:.reveal}

```text
$ python -m examples.maze_toy.run_baseline
Scorecard: random  (perturbation: env-default)
  episodes                       : 30
  action success rate            : 0.000
  average steps to success       : n/a
  planning latency per call (ms) : 0.026
  perturbation recovery rate     : 0.000
  average compute per decision   : n/a

Scorecard: greedy-no-waypoint  (perturbation: env-default)
  episodes                       : 30
  action success rate            : 0.000
  average steps to success       : n/a
  planning latency per call (ms) : 0.002
  perturbation recovery rate     : 0.000
  average compute per decision   : n/a

Scorecard: tabular-world-model  (perturbation: env-default)
  episodes                       : 30
  action success rate            : 1.000
  average steps to success       : 33.800
  planning latency per call (ms) : 3.120
  perturbation recovery rate     : 1.000
  average compute per decision   : 256.410

Wrote sample report to examples/maze_toy/sample_report.json
```
{:.reveal}

## The same contract holds for a learned model
{:.reveal}

The three policies above use stdlib-only Python. The thesis of this framework - that *any* action-conditioned world model can plug into the same evaluation layer - is only credible if it actually works on a learned model. So here is the smallest possible demonstration: train a tiny PyTorch MLP on the maze's transitions, plug it in as the `dynamics` callable, run the same benchmark.
{:.reveal}

<section class="policy-comparison reveal">
  <article class="policy-card policy-success">
    <header>
      <h3>Oracle dynamics (stdlib)</h3>
      <p class="policy-tagline">The reference run from the section above, kept here for side-by-side reading.</p>
    </header>
    <div class="big-number">100%</div>
    <p class="big-label">success rate over 30 episodes</p>
    <dl class="card-stats">
      <div><dt>latency / call</dt><dd>3.12 ms</dd></div>
      <div><dt>compute / decision</dt><dd>~256 rollout-units</dd></div>
      <div><dt>verdict</dt><dd>reaches goal in ~34 steps.</dd></div>
    </dl>
  </article>

  <article class="policy-card policy-success">
    <header>
      <h3>Learned MLP dynamics (PyTorch)</h3>
      <p class="policy-tagline">Same MPC planner, but `dynamics` is now a tiny MLP trained on 64 (state, action, next_state) transitions.</p>
    </header>
    <div class="big-number">100%</div>
    <p class="big-label">success rate over 30 episodes</p>
    <dl class="card-stats">
      <div><dt>latency / call</dt><dd>236.93 ms</dd></div>
      <div><dt>compute / decision</dt><dd>~256 rollout-units</dd></div>
      <div><dt>verdict</dt><dd>contract holds. Latency is 76x higher.</dd></div>
    </dl>
  </article>
</section>

Same success, same steps to success, same nominal compute - **76 times the per-call latency at horizon 20.** Without measuring latency per call, you would conclude "it works just as well!" while the actual deployment cost is two orders of magnitude higher. That is exactly the trade-off the framework is built to expose.
{:.reveal}

<figure class="figure-wide reveal">
  <img src="assets/horizon_sweep_compare.svg" alt="Two stacked panels. Top: success rate vs plan horizon, with the oracle and learned curves overlapping at 100% past horizon 15. Bottom: per-call planning latency, with the learned curve sitting 62 to 77 times above the oracle curve depending on the horizon." />
  <figcaption>Same maze, same MPC, same evaluation contract. The success-rate curves overlap; per-call latency is <strong>62 to 77 times higher</strong> for the learned MLP, depending on the horizon. Generated by <code>python -m examples.maze_toy.run_learned_sweep</code>.</figcaption>
</figure>

The captured terminal output of the run that produced the cards above:
{:.reveal}

```text
$ pip install -e ".[learned]"
$ python -m examples.maze_toy.run_learned_baseline
Training MLP dynamics on maze transitions (~64 samples, 800 epochs)...
Training done.

Scorecard: tabular-world-model (oracle dynamics)  (perturbation: env-default)
  episodes                       : 30
  action success rate            : 1.000
  average steps to success       : 33.800
  planning latency per call (ms) : 3.116
  perturbation recovery rate     : 1.000
  average compute per decision   : 256.410

Scorecard: tabular-world-model (learned MLP dynamics)  (perturbation: env-default)
  episodes                       : 30
  action success rate            : 1.000
  average steps to success       : 33.800
  planning latency per call (ms) : 236.926
  perturbation recovery rate     : 1.000
  average compute per decision   : 256.410
```
{:.reveal}

The proof of contract: a 70-line PyTorch adapter is a drop-in for `TabularWorldModelPlanner`'s `dynamics` argument. The rest of the framework - the runner, the metrics, the scorecard, the perturbation library, the horizon sweep - does not change. Source: [`src/wmel/adapters/learned_dynamics_torch.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/src/wmel/adapters/learned_dynamics_torch.py).
{:.reveal}

## The evaluation contract any world model can plug into

![architecture](assets/architecture.svg){:.figure-architecture-img}

Every adapter exposes the four hooks above (`encode`, `rollout`, `score`, `plan`). The benchmark runner does the rest: rollouts, perturbations, latency measurement, scorecard. A concrete subclass under [`src/wmel/adapters/tabular_world_model.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/src/wmel/adapters/tabular_world_model.py) implements all four with stdlib-only random-shooting MPC.

## Effective planning horizon, made visible
{:.reveal}

The framework's first headline result: sweep the planning horizon of a tabular world-model planner on the maze toy and watch where it pays off. Hover any horizon below to see all of its metrics together. Success saturates at h = 15. Per-call latency keeps climbing past the plateau without buying any extra success - and steps-to-success start to degrade as the planner over-commits before replanning.
{:.reveal}

<div class="chart-container has-tooltips reveal" aria-label="Interactive horizon-sweep chart. Hover or focus a horizon to see its success rate, per-call latency, compute per decision, and average steps to success.">
  {% include_relative assets/horizon_sweep.svg %}
</div>

The same scorecard structure applies to every benchmark card in [03_benchmark_cards.html](03_benchmark_cards.html). The applied questions change - "Can a world model push a part into spec faster than a hand-tuned controller on a 50 ms decision loop?" for Push-T, "Does a stacking model transfer to a new goal without retraining?" for OGBench Cube - but the columns stay the same.
{:.reveal}

## Reproduce in 25 seconds, on a laptop, no GPU
{:.reveal}

```bash
git clone https://github.com/Denis-hamon/world-model-eval-lab.git
cd world-model-eval-lab
pip install -e ".[dev]"
```
{:.reveal}

Then run a single benchmark, or sweep the planning horizon, directly via the installed `wmel` console script - no need to call into the `examples/` modules:
{:.reveal}

```bash
# One scorecard, one JSON report
wmel run --env maze_toy --policy tabular-world-model --episodes 30 --output run.json

# Horizon sweep, comma-separated horizons, one combined JSON
wmel sweep --env maze_toy --plan-horizons 5,10,15,20,30 --output sweep.json
```
{:.reveal}

Both commands honour the same `Perturbation` library as the Python API:
{:.reveal}

```bash
# Drop the next 2 queued actions when the perturbation fires
wmel run --env maze_toy --policy tabular-world-model --perturbation drop-next-2 \
  --perturb-prob 0.3 --output run.json

# Compose env-default and drop-next-3 at the same trigger moment
wmel sweep --env maze_toy --perturbation 'composite:env-default+drop-next-3' \
  --output sweep.json
```
{:.reveal}

Every JSON report carries a versioned envelope (`schema_version`, `wmel_version`, `generated_at`), so downstream tooling can rely on a stable shape across releases. All runs are deterministic with `seed=0`. The same scripts under `examples/maze_toy/run_*.py` keep working for users who prefer the Python API.
{:.reveal}

## Read more
{:.reveal}

- [Thesis](00_thesis.html) - why static benchmarks miss the point.
- [Evaluation gap](01_evaluation_gap.html) - what is missing between research and deployment.
- [Metric taxonomy](02_metric_taxonomy.html) - the metric set, with a worked horizon-sweep example.
- [Benchmark cards](03_benchmark_cards.html) - Push-T, Reacher, Two-Room, Maze, OGBench Cube.
- [Industrial use cases](04_industrial_use_cases.html) - robotics, industrial automation, datacenter ops, logistics, safety monitoring.
- [30-day study plan](05_30_day_prototype_plan.html) - week-by-week scope and status.
- [Reading a scorecard](06_demo.html) - row-by-row walkthrough of a real sweep result.
{:.reveal}

## Releases
{:.reveal}

- [v0.7.0](https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.7.0) - `wmel` CLI, versioned JSON schema, perturbation-aware sweep, stdlib-only CI job.
- [v0.6.0](https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.6.0) - proof of contract for learned PyTorch dynamics on the maze.
- [v0.5.0](https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.5.0) - pluggable perturbation library.
- [v0.4.0](https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.4.0) - Markdown export and compute-per-decision.
- [v0.3.1](https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.3.1) - initial public release.
{:.reveal}

## Disclaimer
{:.reveal}

This is an independent study of evaluation methodology for action-conditioned world models. It is **not** an official artifact of AMI, Meta, the LeWorldModel project, or any of their authors, and **not** an artifact of any current or past employer of the author. References to JEPA-style or LeWorldModel concepts are conceptual, not affiliational.
{:.reveal}
