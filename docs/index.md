---
layout: default
image: /assets/architecture.svg
next:
  title: "00 - Thesis"
  url: 00_thesis.html
---

<div class="release-banner">
  <span class="tag">v0.17.0</span>
  <span class="release-banner-text">
    A third environment ships: the four-arm CPG matrix is replayed on DMC Reacher-easy, the first task with a two-dimensional action and an exactly-reconstructed oracle. It is the cleanest case in the paper &mdash; the oracle solves the reach perfectly (<code>1.000</code>) and both learned arms are clearly non-zero (the published TD-MPC2 dynamics reaches <code>0.567</code>&ndash;<code>0.633</code>; the MLP arm <code>0.300</code>&ndash;<code>0.333</code>), so all four cells are <code>MODEL BOTTLENECK</code> on genuine, non-degenerate gaps. Evidence the metric tracks gap <em>magnitude</em>, not just presence.
  </span>
  <a href="paper.html#sec-reacher">Read the section &rarr;</a>
</div>

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
      <a class="btn-primary" href="#problem">Start the walkthrough</a>
      <a class="btn-ghost" href="07_cpg.html">Read about CPG</a>
      <a class="btn-ghost" href="https://github.com/Denis-hamon/world-model-eval-lab">GitHub</a>
    </div>
  </div>
  <figure class="hero-figure">
    <img src="assets/maze.svg" alt="A 7x7 maze with an animated agent walking the optimal path from start to goal." />
    <figcaption>An agent walks the 7x7 maze. Optimal path = 14 actions. The world-model planner finds it in ~33 steps with replanning.</figcaption>
  </figure>
</section>

<ul class="stat-strip">
  <li><span class="stat-value">v0.17.0</span><span class="stat-label">current version</span></li>
  <li><span class="stat-value">148</span><span class="stat-label">passing tests</span></li>
  <li><span class="stat-value">4 envs</span><span class="stat-label">maze toy + 3 DMC tasks</span></li>
  <li><span class="stat-value">CPU-only</span><span class="stat-label">no GPU required</span></li>
  <li><span class="stat-value">0</span><span class="stat-label">ML dependencies at runtime</span></li>
</ul>

[![tests](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Denis-hamon/world-model-eval-lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green)](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/LICENSE)

<aside class="whats-new">
  <h3>What's new in v0.17</h3>
  <ul>
    <li><strong>Third environment: DMC Reacher-easy</strong> (<code>src/wmel/envs/dmc_reacher.py</code>). The first task with a two-dimensional action (a $3\times3 = 9$ torque grid) and an oracle reconstructed exactly (reproduces <code>env.step</code> to $&lt;10^{-16}$). Four-arm CPG matrix pooled to $n = 30$ at TD-MPC2 <code>model_size = 1</code>.</li>
    <li><strong>The cleanest gap-magnitude case in the paper</strong>: the oracle solves the reach in every cell ($1.000$) and both learned arms are clearly non-zero (TD-MPC2 $0.567$ random-shooting / $0.633$ CEM, the paper's highest learned-arm successes). All four cells are <code>MODEL BOTTLENECK</code> on genuine, non-degenerate gaps ($+0.367$ to $+0.700$), and the verdict ranks the two learned dynamics by how much planning success they forfeit.</li>
    <li><strong>Three environments, one verdict</strong>: across Acrobot, Cartpole, and Reacher &mdash; underactuated swing-up to actuated reaching, 1-D to 2-D actions, oracle rates from $0.30$ to $1.00$ &mdash; the gated verdict reproduces <code>MODEL BOTTLENECK</code> wherever a real gap exists and abstains where it does not. The paper's Reacher section, abstract, intro, and conclusion are updated; Figure 3 extended to three series.</li>
    <li><strong>From v0.16: a power-analysis tool</strong>. Because the verdict gate is a function of the confidence interval, it doubles as a power calculator: the per-arm episode count a comparison needs before its interval clears zero. A plausible leaderboard near-tie ($0.94$ vs $0.92$ at $n = 100$) is shown to be statistically indistinguishable from noise (it needs $n = 209$). See the paper's power-analysis section and Figure 5.</li>
  </ul>
</aside>

<section class="chapter" id="problem" markdown="1">
  <p class="chapter-eyebrow">Step 01</p>
  <h2 class="chapter-title">The problem</h2>
  <p class="chapter-lead">Action-conditioned world models are routinely evaluated on prediction quality: reconstruction loss, FID, held-out next-frame accuracy. None of these answer the question an applied team must answer before integrating a model into a control loop.</p>

The question is **decision quality, not prediction quality.** Does the model, when used by a planner, produce decisions that **succeed** within the **latency** and **compute** budget the deployment will accept? Does it **recover** from perturbations? Does it **generalise** across related tasks?

A low validation MSE is necessary but not sufficient. The framework's headline example: a Markovian MLP world model with val_mse $= 0.026$ on DMC Acrobot. Predicting accurately. Planning a $0\%$ success rate.

See [Thesis](00_thesis.html) and [Evaluation gap](01_evaluation_gap.html) for the long form.
</section>

<section class="chapter" id="contract" markdown="1">
  <p class="chapter-eyebrow">Step 02</p>
  <h2 class="chapter-title">The evaluation contract</h2>
  <p class="chapter-lead">Every adapter exposes four hooks. The benchmark runner does the rest: rollouts, perturbations, latency measurement, scorecard.</p>

![architecture](assets/architecture.svg){:.figure-architecture-img}

The contract is intentionally minimal: `encode` (observation &rarr; latent), `rollout` (latent + action sequence &rarr; latent sequence), `score` (latent + goal &rarr; reward), `plan` (observation + goal + horizon &rarr; action sequence). Anything that implements these four methods plugs into the same runner and gets compared on the same scorecard structure.

Concrete subclasses live in [`src/wmel/adapters/`](https://github.com/Denis-hamon/world-model-eval-lab/tree/main/src/wmel/adapters): a stdlib tabular planner, a PyTorch MLP, and the DMC Acrobot oracle. None of them know about the runner, the metrics, or each other.
</section>

<section class="chapter" id="toy" markdown="1">
  <p class="chapter-eyebrow">Step 03</p>
  <h2 class="chapter-title">How it behaves on a toy</h2>
  <p class="chapter-lead">A 7x7 maze with a vertical wall and one doorway. Same env, same 30 episodes, same seed; three different planners. The framework's first non-trivial demonstration that the contract holds and the metrics discriminate.</p>

  <h3 class="chapter-sub">Three policies, side by side</h3>

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
  <figcaption>Three agents, three outcomes, one shared evaluation contract.</figcaption>
</figure>

  <h3 class="chapter-sub">The same contract holds for a learned model</h3>

The thesis is only credible if a real learned model can plug into the same evaluation layer. The smallest demonstration: a tiny PyTorch MLP trained on 64 maze transitions, passed in as the `dynamics=` callable.

<section class="policy-comparison reveal">
  <article class="policy-card policy-success">
    <header>
      <h3>Oracle dynamics (stdlib)</h3>
      <p class="policy-tagline">The reference run from the section above.</p>
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
      <p class="policy-tagline">Same MPC planner, dynamics is now a tiny MLP trained on 64 transitions.</p>
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

Same success, same steps to success, same nominal compute -- **76 times the per-call latency at horizon 20.** Without measuring latency per call, you would conclude "it works just as well!" while the actual deployment cost is two orders of magnitude higher.

  <h3 class="chapter-sub">Effective planning horizon, made visible</h3>

Sweep the planning horizon of the tabular world-model planner and watch where it pays off. Hover any horizon to see all four metrics together. Success saturates at $h = 15$; per-call latency keeps climbing past the plateau without buying any extra success.

<div class="chart-container has-tooltips reveal" aria-label="Interactive horizon-sweep chart. Hover or focus a horizon to see its success rate, per-call latency, compute per decision, and average steps to success.">
  {% include_relative assets/horizon_sweep.svg %}
</div>
</section>

<section class="chapter" id="real" markdown="1">
  <p class="chapter-eyebrow">Step 04</p>
  <h2 class="chapter-title">How it scales to real control</h2>
  <p class="chapter-lead">The framework's flagship metric. Run the same random-shooting MPC planner twice on DeepMind Control Suite Acrobot-swingup &mdash; once against oracle dynamics (real MuJoCo physics), once against a Markovian MLP world model trained on 2&nbsp;000 random transitions. The only thing that changes is the <code>dynamics=</code> callable. The success-rate difference is the <strong>Counterfactual Planning Gap</strong>.</p>

<section class="policy-comparison reveal">
  <article class="policy-card policy-success">
    <header>
      <h3>Oracle dynamics</h3>
      <p class="policy-tagline">Random-shooting MPC against real MuJoCo physics.</p>
    </header>
    <div class="big-number">30%</div>
    <p class="big-label">success rate over 10 episodes</p>
    <dl class="card-stats">
      <div><dt>latency / call</dt><dd>77.3 ms</dd></div>
      <div><dt>compute / decision</dt><dd>407.1 rollout-units</dd></div>
      <div><dt>avg steps to success</dt><dd>180.7</dd></div>
    </dl>
  </article>

  <article class="policy-card policy-fail">
    <header>
      <h3>Learned MLP dynamics</h3>
      <p class="policy-tagline">Same MPC, same scoring, MLP trained on 2&nbsp;000 random transitions.</p>
    </header>
    <div class="big-number">0%</div>
    <p class="big-label">success rate over 10 episodes</p>
    <dl class="card-stats">
      <div><dt>latency / call</dt><dd>65.3 ms</dd></div>
      <div><dt>compute / decision</dt><dd>157.3 rollout-units</dd></div>
      <div><dt>val MSE</dt><dd>0.026 (low) - yet success collapses</dd></div>
    </dl>
  </article>

  <article class="policy-card policy-cpg">
    <header>
      <h3>Counterfactual Planning Gap</h3>
      <p class="policy-tagline">Decoupling model error from planner capacity.</p>
    </header>
    <div class="big-number">+0.30</div>
    <p class="big-label">raw difference of success rates</p>
    <dl class="card-stats">
      <div><dt>AC 95% CI</dt><dd>[-0.06, +0.56]</dd></div>
      <div><dt>n / arm</dt><dd>10 episodes</dd></div>
      <div><dt>verdict</dt><dd><span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span></dd></div>
    </dl>
  </article>
</section>

A low validation MSE on prediction quality does **not** translate into closed-loop success. CPG quantifies the planning-side gap with an Agresti--Caffo $95\%$ confidence interval that **does not collapse** at the boundary proportions $p \in \{0, 1\}$ where the standard Wald approximation degenerates. The verdict is gated on the CI lower bound, not the raw point estimate -- at $n = 10$ the framework reports `INCONCLUSIVE` rather than over-claiming a model bottleneck.

  <h3 class="chapter-sub">Multi-seed extension: capacity vs.\ coverage</h3>

At $n = 10$ the framework refused to commit. We then pooled three seeds at $n = 50$ episodes per arm per seed and swept the MLP's training-set size by a factor of $100$. The verdict hardens to **MODEL BOTTLENECK** with a tight, identical confidence interval *in every cell*.

| Train size | Val MSE | Oracle | Learned | Raw CPG | AC 95% CI | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| $200$ | $0.0651$ | $40/150$ | $0/150$ | $+0.267$ | $[+0.191, +0.335]$ | <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span> |
| $2{,}000$ | $0.0233$ | $40/150$ | $0/150$ | $+0.267$ | $[+0.191, +0.335]$ | <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span> |
| $20{,}000$ | $0.0004$ | $40/150$ | $0/150$ | $+0.267$ | $[+0.191, +0.335]$ | <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span> |

Held-out validation MSE drops by **~150 times** across the three cells. Planning success stays at **exactly zero**. The gap does not close. A prediction-quality metric alone would have declared the largest-data cell solved; CPG points to a *data-coverage* bottleneck (random rollouts in Acrobot never visit the upright-balancing regime) as the most parsimonious read, with planner-side and score-function residuals as plausible second-order contributors. The recommended remediation is to change the data-collection policy, not to grow the model.

[Read the full page on CPG &rarr;](07_cpg.html) &nbsp;&middot;&nbsp; [Read the paper &rarr;](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/main.tex)

  <h3 class="chapter-sub">Across three environments, and a power-analysis tool</h3>

The same four-arm matrix (random-shooting / CEM, against a learned MLP and against published TD-MPC2 dynamics) was then replayed on two further DeepMind Control Suite tasks. On **Cartpole-swingup** the verdict reproduces in every cell at TD-MPC2 `model_size = 5`; at `model_size = 1` the CEM x TD-MPC2 cell flips to `INCONCLUSIVE` (learned $0.533$ vs oracle $0.500$, CI $[-0.28, +0.21]$) -- the first moderate-$n$ cell where the gate refuses to commit. On **Reacher-easy** the oracle solves the reach perfectly ($1.000$) and both learned arms are clearly non-zero ($0.300$ to $0.633$), so all four cells are `MODEL BOTTLENECK` on genuine, non-degenerate gaps ($+0.367$ to $+0.700$) -- the cleanest evidence the metric tracks gap *magnitude* rather than gap presence.

Because the verdict gate is a function of the confidence interval, it also answers a question a bare leaderboard cannot: **how many episodes a comparison needs before its ranking is trustworthy.** A plausible $0.94$-vs-$0.92$ near-tie at $n = 100$ is statistically indistinguishable from noise; the gate shows it needs $n = 209$ per arm before the interval clears zero.

[Cross-env (Cartpole) &rarr;](paper.html#sec-crossenv) &nbsp;&middot;&nbsp; [Third env (Reacher) &rarr;](paper.html#sec-reacher) &nbsp;&middot;&nbsp; [Power analysis &rarr;](paper.html#sec-power)
</section>

<section class="chapter" id="reproduce" markdown="1">
  <p class="chapter-eyebrow">Step 05</p>
  <h2 class="chapter-title">Reproduce in 25 seconds</h2>
  <p class="chapter-lead">No GPU, no heavy ML dependency at runtime. Core install plus an installed CLI; optional extras for PyTorch and DMC.</p>

```bash
git clone https://github.com/Denis-hamon/world-model-eval-lab.git
cd world-model-eval-lab
pip install -e ".[dev]"
```

Then run a single benchmark or sweep the planning horizon via the installed `wmel` console script:

```bash
# One scorecard, one JSON report
wmel run --env maze_toy --policy tabular-world-model --episodes 30 --output run.json

# Horizon sweep, comma-separated horizons, one combined JSON
wmel sweep --env maze_toy --plan-horizons 5,10,15,20,30 --output sweep.json
```

The DMC Acrobot CPG worked example needs the `[control]` and `[learned]` extras:

```bash
pip install -e ".[dev,control,learned]"
python -m experiments.dmc_acrobot.cpg
# -> results/dmc_acrobot/cpg.json
```

Every JSON report carries a versioned envelope (`schema_version`, `wmel_version`, `generated_at`).
</section>

## Where to read next

<section class="reading-paths">
  <article class="path-card">
    <p class="path-eyebrow">For the researcher</p>
    <h3>How the framework thinks</h3>
    <p class="path-lead">The metric taxonomy, the four-method evaluation contract, and the CPG definition with its Agresti-Caffo CI and gated verdict.</p>
    <ul class="path-links">
      <li><a href="00_thesis.html">Thesis</a></li>
      <li><a href="01_evaluation_gap.html">Evaluation gap</a></li>
      <li><a href="02_metric_taxonomy.html">Metric taxonomy</a></li>
      <li><a href="07_cpg.html">Counterfactual Planning Gap</a></li>
    </ul>
  </article>

  <article class="path-card">
    <p class="path-eyebrow">For the practitioner</p>
    <h3>Plug a model in</h3>
    <p class="path-lead">A walkthrough of one scorecard, the benchmark cards each environment maps to, and the industrial use-cases the framework is built around.</p>
    <ul class="path-links">
      <li><a href="06_demo.html">Reading a scorecard</a></li>
      <li><a href="03_benchmark_cards.html">Benchmark cards</a></li>
      <li><a href="04_industrial_use_cases.html">Industrial use cases</a></li>
      <li><a href="05_30_day_prototype_plan.html">30-day study plan</a></li>
    </ul>
  </article>

  <article class="path-card">
    <p class="path-eyebrow">For the reader</p>
    <h3>The paper and its sources</h3>
    <p class="path-lead">The short paper accompanying the framework, the LaTeX source, the reproducibility script, and the citation entry.</p>
    <ul class="path-links">
      <li><a href="https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/main.tex">Paper (main.tex)</a></li>
      <li><a href="https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/references.bib">BibTeX (references.bib)</a></li>
      <li><a href="https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/build_figures.py"><code>build_figures.py</code></a></li>
      <li><a href="https://github.com/Denis-hamon/world-model-eval-lab/blob/main/CITATION.cff"><code>CITATION.cff</code></a></li>
    </ul>
  </article>
</section>

## Milestones

<p class="timeline-note">Tagged GitHub releases run through <a href="https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.11.0">v0.11.0</a> (the framework and the first paper draft). The research milestones since then are tracked in the paper and the version number; a consolidated <code>v1</code> release will be tagged when the paper is submitted. Version labels below link to their release tag where one exists, otherwise to the paper section that documents the milestone.</p>

<section class="release-timeline">
  <article class="release-card release-current">
    <div class="release-head">
      <a class="release-version" href="paper.html#sec-reacher">v0.17.0</a>
      <span class="release-meta">2026-05-31 &middot; current</span>
    </div>
    <p class="release-title">Third environment: DMC Reacher-easy</p>
    <p class="release-body">The four-arm CPG matrix replayed on a third env: the first task with a two-dimensional action and an exactly-reconstructed oracle. Oracle solves the reach in every cell ($1.000$); both learned arms are clearly non-zero (TD-MPC2 $0.567$&ndash;$0.633$, the paper's highest), so all four cells are <code>MODEL BOTTLENECK</code> on genuine, non-degenerate gaps ($+0.367$ to $+0.700$) &mdash; the cleanest evidence the metric tracks gap magnitude, not just presence. See the paper's Reacher section; Figure 3 extended to three series.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="paper.html#sec-power">v0.16</a>
      <span class="release-meta">2026-05-29</span>
    </div>
    <p class="release-title">Power analysis: how many episodes a ranking needs</p>
    <p class="release-body">The verdict gate, read as a power calculator: <code>ac_ci_half_width</code>, <code>required_n_for_half_width</code>, <code>detectable_gap_at_n</code>. A plausible $0.94$-vs-$0.92$ leaderboard near-tie at $n = 100$ is shown statistically indistinguishable from noise (needs $n = 209$). Paper power-analysis section + Figure 5. Also: CPG positioned neutrally against the concurrent swm platform paper.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="paper.html#sec-crossenv">v0.15.0</a>
      <span class="release-meta">2026-05-23</span>
    </div>
    <p class="release-title">Cross-environment: DMC Cartpole-swingup, two capacities</p>
    <p class="release-body">Four-arm CPG matrix on a second env at TD-MPC2 <code>model_size = 5</code> AND <code>model_size = 1</code>, $n = 30$ pooled each. All four cells at <code>size = 5</code> reproduce <code>MODEL BOTTLENECK</code>; the CEM&times;TD-MPC2 cell at <code>size = 1</code> flips to <code>INCONCLUSIVE</code> (learned $0.533$ vs oracle $0.500$, CPG $-0.033$, CI $[-0.28, +0.21]$) &mdash; first moderate-$n$ <code>INCONCLUSIVE</code> in the paper. See the paper's cross-environment section + Figures 3 and 4.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="paper.html#sec-empirical">v0.14.1</a>
      <span class="release-meta">2026-05-23</span>
    </div>
    <p class="release-title">First two paper figures (CPG vs data, coverage histogram)</p>
    <p class="release-body">PGF/TikZ Figure 1 (val MSE plummets $\sim 150\times$ while CPG stays flat at $+0.267$, asymmetric Agresti-Caffo CI) and Figure 2 (uprightness coverage: $0/2000$ random states reach upright vs $20.2\%$ for oracle). Two adversarial-review fixes addressed: sig-fig parity and asymmetric error bars.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="paper.html#sec-robustness">v0.14.0</a>
      <span class="release-meta">2026-05-23</span>
    </div>
    <p class="release-title">Robustness sweep: published model, stronger planner, perturbation</p>
    <p class="release-body">Three new axes test the v0.11 <code>MODEL BOTTLENECK</code> verdict: TD-MPC2 (2M env steps) as <code>dynamics</code>, CEM as planner, <code>DropNextActions(k)</code> as in-episode perturbation. Verdict survives all three; pooled-150 under CEM tightens CI half-width to <code>0.054</code>. See the paper's robustness sections.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.11.0">v0.11.0</a>
      <span class="release-meta">2026-05-16</span>
    </div>
    <p class="release-title">Multi-seed CPG sweep: capacity vs.\ coverage</p>
    <p class="release-body">Verdict hardens from <code>INCONCLUSIVE</code> (n = 10) to <code>MODEL BOTTLENECK</code> (n = 150 pooled across three seeds); training-set sweep across <code>{200, 2 000, 20 000}</code> transitions leaves verdict and CI <em>identical</em> while validation MSE drops 150&times;. Paper Section 5.5 + 5.6 updated.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.10.0">v0.10.0</a>
      <span class="release-meta">2026-05-16</span>
    </div>
    <p class="release-title">Short paper: Counterfactual Planning Gap</p>
    <p class="release-body">~7-page LaTeX paper under <code>paper/</code>, 23 BibTeX entries, reproducibility script, three adversarial-review findings addressed before tag.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.9.0">v0.9.0</a>
      <span class="release-meta">2026-05</span>
    </div>
    <p class="release-title">CPG metric with Agresti-Caffo CI and gated verdict</p>
    <p class="release-body">Five-branch verdict gated on the CI lower bound; honest <code>INCONCLUSIVE</code> at n=10 instead of over-claiming a Wald-CI-driven significance.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.8.0">v0.8.0</a>
      <span class="release-meta">2026-05</span>
    </div>
    <p class="release-title">DMC Acrobot-swingup wired in</p>
    <p class="release-body">First non-toy environment via <code>wmel.envs.dmc_acrobot</code>, with a Markovian MLP learned dynamics and an oracle dynamics factory. <code>dm-control</code> is an optional extra.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.7.0">v0.7.0</a>
      <span class="release-meta">2026-04</span>
    </div>
    <p class="release-title">CLI, versioned JSON schema, perturbation-aware sweep</p>
    <p class="release-body"><code>wmel run</code> / <code>wmel sweep</code> console scripts; JSON envelope with <code>schema_version</code>, <code>wmel_version</code>, <code>generated_at</code>. Second CI job locks the no-torch runtime promise.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.6.0">v0.6.0</a>
      <span class="release-meta">2026-04</span>
    </div>
    <p class="release-title">Proof of contract for learned PyTorch dynamics</p>
    <p class="release-body">PyTorch MLP fits the maze's transition table and plugs in as a drop-in <code>dynamics=</code> callable. Identical success, 76x higher per-call latency -- the trade-off the framework is built to expose.</p>
  </article>

  <article class="release-card">
    <div class="release-head">
      <a class="release-version" href="https://github.com/Denis-hamon/world-model-eval-lab/releases/tag/v0.5.0">v0.5.0</a>
      <span class="release-meta">2026-04</span>
    </div>
    <p class="release-title">Pluggable perturbation library</p>
    <p class="release-body"><code>Perturbation</code>, <code>EnvPerturbation</code>, <code>DropNextActions</code>, <code>CompositePerturbation</code>. Runner inner loop switched to <code>deque</code> for O(1) action-queue pops.</p>
  </article>
</section>

## Disclaimer

This is an independent study of evaluation methodology for action-conditioned world models. It is **not** an official artifact of AMI, Meta, the LeWorldModel project, or any of their authors, and **not** an artifact of any current or past employer of the author. References to JEPA-style or LeWorldModel concepts are conceptual, not affiliational.
