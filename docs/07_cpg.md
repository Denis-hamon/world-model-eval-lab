---
layout: default
title: "Counterfactual Planning Gap"
description: "A decision-grade metric for decoupling model error from planner capacity in world model evaluation."
prev:
  title: "06 - Reading a scorecard"
  url: 06_demo.html
next:
  title: "Back to home"
  url: index.html
---

# Counterfactual Planning Gap

A single scalar that answers the question every applied team eventually asks: *if I swap in the learned world model in place of an oracle, how much of the planning success do I lose?*

Run the same planner on the same benchmark **twice**. Once with oracle dynamics (the true environment's transition function), once with a learned model. The only thing that changes between the two runs is the `dynamics=` callable. The success-rate difference, with its calibrated confidence interval and a gated verdict, is the **Counterfactual Planning Gap (CPG)**.

<section class="chapter" id="definition" markdown="1">
  <p class="chapter-eyebrow">Step 01</p>
  <h2 class="chapter-title">Definition</h2>
  <p class="chapter-lead">The metric is a fraction in $[-1, +1]$.</p>

$$
\mathrm{CPG} \;=\; \mathrm{success\_rate}(D^{\star}) \;-\; \mathrm{success\_rate}(D_\theta)
$$

with $D^{\star}$ the oracle dynamics and $D_\theta$ the learned model. All other quantities (env, planner, scoring function, $N$ episodes, horizon, seed, and the initial-state distribution) are held fixed between the two runs.

This identification &mdash; same planner, same scoring, same env, same seed, only `dynamics=` swapped &mdash; attributes the success-rate difference to that callable. CPG is therefore a property of the **(model, planner, distribution) triple**, *not* a planner- or distribution-free property of the model alone: as the worked example below shows, the same model can flip the verdict when only the evaluation distribution changes. Read CPG relative to a stated planner and a stated initial-state distribution.
</section>

<section class="chapter" id="why-ac" markdown="1">
  <p class="chapter-eyebrow">Step 02</p>
  <h2 class="chapter-title">Why Agresti--Caffo, not Wald</h2>
  <p class="chapter-lead">The standard Wald confidence interval collapses to a point at the boundary proportions $p \in \{0, 1\}$ &mdash; exactly the regime this framework lands in at small $n$. The Agresti--Caffo plus-4 adjustment fixes that.</p>

The standard Wald $95\%$ CI on a difference of two binomial proportions uses

$$
\mathrm{SE}_{\mathrm{Wald}} \;=\; \sqrt{\frac{p_o(1-p_o)}{n_o} + \frac{p_\ell(1-p_\ell)}{n_\ell}}.
$$

When either arm sits at $p \in \{0, 1\}$, the Wald variance collapses to zero, the interval shrinks to a point, and a meaningless "significant" result drops out. A learned planner that fails on every episode is precisely this regime.

The **Agresti--Caffo plus-4** adjustment fixes this by adding one pseudo-success and one pseudo-failure to each arm before computing the standard-normal CI:

$$
\tilde p = \frac{s + 1}{n + 2}, \qquad
\mathrm{SE}_{\mathrm{AC}} \;=\; \sqrt{\frac{\tilde p_o (1 - \tilde p_o)}{n_o + 2} + \frac{\tilde p_\ell (1 - \tilde p_\ell)}{n_\ell + 2}}
$$

$$
\mathrm{CI}_{95\%}(\mathrm{CPG}) \;=\; \bigl[\, \tilde\Delta - 1.96\,\mathrm{SE}_{\mathrm{AC}}, \;\; \tilde\Delta + 1.96\,\mathrm{SE}_{\mathrm{AC}} \,\bigr]
$$

The variance never collapses, coverage is honest down to single-digit $n$, and the interval converges to Wald for moderate-to-large samples. The framework reports **both** the raw $\hat\Delta$ (what a reader expects) and the AC CI (what is defensible). They coincide for large $n$.
</section>

<section class="chapter" id="verdict" markdown="1">
  <p class="chapter-eyebrow">Step 03</p>
  <h2 class="chapter-title">The verdict, gated on the CI</h2>
  <p class="chapter-lead">A CPG reported without a significance gate over-claims. A <code>gap = +0.1</code> from $n = 10$ is indistinguishable from noise but would otherwise read as a model bottleneck. The framework exposes a five-branch verdict that consults the AC interval, <strong>not</strong> the raw point estimate.</p>

<div class="verdict-legend" markdown="0">
  <p><span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span> &mdash; $\mathrm{CI}_{\mathrm{lo}} &gt; 0$. The oracle is reliably better; closing the gap is a model problem.</p>
  <p><span class="verdict-pill verdict-learned-outperforms">LEARNED OUTPERFORMS</span> &mdash; $\mathrm{CI}_{\mathrm{hi}} &lt; 0$. Rare; investigate regularisation or planner-search interactions.</p>
  <p><span class="verdict-pill verdict-planner-bottleneck">PLANNER BOTTLENECK</span> &mdash; CI crosses $0$ <em>and</em> both success rates are within $\tau$ of $0$. Neither planner solves the task; the framework needs a stronger search, not a stronger model.</p>
  <p><span class="verdict-pill verdict-as-good-as">MODEL AS GOOD AS ORACLE</span> &mdash; CI crosses $0$ <em>and</em> both success rates are within $\tau$ of $1$. The learned model matches the oracle for planning purposes on this task.</p>
  <p><span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span> &mdash; CI crosses $0$ in a middle-of-the-road regime. The sample size is insufficient to discriminate; report this verdict and run more episodes.</p>
</div>

The default tolerance is $\tau = 0.05$. Crucially, `MODEL BOTTLENECK` is **not** the default when $\hat\Delta > 0$ &mdash; it requires the AC lower bound to be strictly positive.
</section>

<section class="chapter" id="example" markdown="1">
  <p class="chapter-eyebrow">Step 04</p>
  <h2 class="chapter-title">Worked example: a fixed-init snapshot, then the metric correcting itself</h2>
  <p class="chapter-lead">The framework's reference run. Random-shooting MPC over a five-level torque discretisation, $50$ candidates of $15$-step horizon, $10$ episodes per arm, seed $0$, on DMC Acrobot-swingup.</p>

|  | Oracle dynamics | Learned MLP |
| --- | ---: | ---: |
| Success rate | $0.30$ ($3/10$) | $0.00$ ($0/10$) |
| Avg. steps to success | $180.7$ | n/a |
| Per-call latency (ms) | $77.3$ | $65.3$ |
| Compute / decision | $407.1$ | $157.3$ |

| Counterfactual Planning Gap | |
| --- | ---: |
| Raw $\hat\Delta$ | $+0.300$ |
| Agresti--Caffo $95\%$ CI | $[-0.059, +0.559]$ |
| Verdict | <span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span> |

The data is *suggestive* of a model bottleneck &mdash; the raw point estimate is positive and large &mdash; but with $n_\ell = 10$ and the learned arm reporting $0/10$, the AC CI cannot rule out zero. The honest call is to **run more episodes**. As the next sub-section shows, doing so first appears to confirm a model bottleneck, and then &mdash; once the *initial-state distribution* is sampled rather than fixed &mdash; overturns it.

Numbers above are pulled verbatim from [`results/dmc_acrobot/cpg.json`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/results/dmc_acrobot/cpg.json). Regenerate with:

```bash
pip install -e ".[dev,control,learned]"
python -m experiments.dmc_acrobot.cpg
```

  <h3 class="chapter-sub">The metric corrects itself</h3>

The honest next step at $n = 10$ is more episodes. Pooling three seeds at $50$ episodes per arm and switching to a stronger CEM planner makes the fixed-start gap look decisive: the oracle solves $88\%$ of episodes, the learned arm stays at $0$, and the verdict hardens to `MODEL BOTTLENECK` ($\mathrm{CPG} = +0.88$, AC CI $[+0.81, +0.92]$). A point-estimate leaderboard would publish that headline.

It is an artifact of the **evaluation distribution**. Every episode above started from the *same* fixed initial state -- a deterministic env reset, with only the planner's internal randomness varying -- and on Acrobot that start happens to be an unusually easy swing-up. (A coverage receipt confirms why the learned arm fails at that start: on the uprightness axis $u = \cos\theta_1 + \cos\theta_2$, $0/2000$ random-rollout training states reach $u > 1.0$, while the oracle planner visits the upright regime on ~20% of its steps -- the MLP is extrapolating off its training manifold during planning.) But the fixed start is not the task. Sampling the task's actual initial-state distribution (a fresh start per episode, the two arms **paired** by start state) collapses the oracle's success rate from $0.88$ to $\sim\!3\%$. With the oracle planner itself solving only $\sim\!3\%$ of random starts, the gap closes and the verdict flips to `PLANNER BOTTLENECK`: even a perfect model would not help, because the search cannot solve the task from a typical start.

| Initial state | Planner | Dynamics | Oracle | Learned | CPG (AC 95% CI), verdict |
|---|---|---|---:|---:|---|
| fixed, pooled 150 | CEM | TD-MPC2 | $0.88$ | $0.00$ | $+0.88$ $[+0.81, +0.92]$, <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span> |
| **task**, pooled 150 | CEM | MLP | $0.033$ | $0.020$ | $+0.013$ $[-0.027, +0.053]$, <span class="verdict-pill verdict-planner-bottleneck">PLANNER BOTTLENECK</span> |
| **task**, pooled 150 | CEM | TD-MPC2 | $0.033$ | $0.027$ | $+0.007$ $[-0.035, +0.049]$, <span class="verdict-pill verdict-planner-bottleneck">PLANNER BOTTLENECK</span> |

This is the single strongest piece of evidence for what the metric is *for*: a calibrated, interval-gated statistic, run honestly over the task distribution, overturned a headline that a point estimate at one configuration would have published. Numbers from [`results/dmc_acrobot/`](https://github.com/Denis-hamon/world-model-eval-lab/tree/main/results/dmc_acrobot) (`varied_init: true`); regenerate the task-level cells via [`RERUN_VARIED_INIT.md`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/experiments/RERUN_VARIED_INIT.md).

  <h3 class="chapter-sub">Heterogeneity across three tasks</h3>

Run over the task distribution on three DMC tasks, the verdict is *heterogeneous*: the gate fires four of its five branches on real data. This is the headline result -- a fixed metric, or a point-estimate leaderboard, reports none of this structure.

| Task | Representative cell | CPG (AC 95% CI) | Verdict |
|---|---|---:|---|
| Acrobot-swingup | CEM, TD-MPC2, task-level pooled 150 | $+0.007$ $[-0.035, +0.049]$ | <span class="verdict-pill verdict-planner-bottleneck">PLANNER BOTTLENECK</span> |
| Reacher-easy | all four arms, pooled 30 | $+0.20$ to $+0.33$ | <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span> |
| Cartpole-swingup | CEM, TD-MPC2, `model_size = 5`, pooled 30 | $-0.27$ $[-0.48, -0.02]$ | <span class="verdict-pill verdict-learned-outperforms">LEARNED OUTPERFORMS</span> |
| Cartpole-swingup | RS, TD-MPC2, `model_size = 5`, pooled 30 | $+0.167$ $[-0.03, +0.34]$ | <span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span> |

- **Acrobot** is `PLANNER BOTTLENECK` (above): the oracle planner solves only ~3% of random starts, so neither arm wins.
- **Reacher** is uniformly `MODEL BOTTLENECK`: the oracle solves the reach perfectly ($1.000$), *both* learned arms are clearly non-zero ($0.667$ to $0.800$), and yet every AC lower bound on the gap stays strictly positive. The verdict here tracks gap *magnitude*, not a learned arm pinned at zero.
- **Cartpole** at the larger TD-MPC2 capacity under CEM is `LEARNED OUTPERFORMS ORACLE`: the learned model lets CEM solve $0.733$ of episodes against the oracle planner's $0.467$. The AC CI $[-0.48, -0.02]$ and a paired-bootstrap CI $[-0.50, -0.03]$ both clear zero (the varied-init arms are paired by initial state). The remaining Cartpole cells are `MODEL BOTTLENECK` or `INCONCLUSIVE` -- one environment, three branches.

Sources: [`results/dmc_acrobot/`](https://github.com/Denis-hamon/world-model-eval-lab/tree/main/results/dmc_acrobot), [`results/dmc_cartpole/`](https://github.com/Denis-hamon/world-model-eval-lab/tree/main/results/dmc_cartpole), [`results/dmc_reacher/`](https://github.com/Denis-hamon/world-model-eval-lab/tree/main/results/dmc_reacher). See the paper's [self-correction](paper.html#sec-selfcorrection), [Cartpole](paper.html#sec-crossenv), and [Reacher](paper.html#sec-reacher) sections.
</section>

<section class="chapter" id="use-cases" markdown="1">
  <p class="chapter-eyebrow">Step 05</p>
  <h2 class="chapter-title">When to use CPG, when not</h2>
  <p class="chapter-lead">The metric is decision-grade by construction; it inherits the limits of the comparison it packages.</p>

  <h3 class="chapter-sub">Use CPG when</h3>

- **Simulated environments** where an oracle dynamics is cheap to instantiate (most physics-based control tasks; gridworlds; OGBench-style tasks).
- **Comparing learned models** of different capacity, training-data budget, or architecture against the same oracle.
- **Decoupling diagnostics** when a model with a low validation MSE produces a planner that does not succeed &mdash; CPG attributes the failure to the model or rules it out.

  <h3 class="chapter-sub">Avoid (or treat as a surrogate) when</h3>

- **Hardware-in-the-loop / real-world environments** where no oracle dynamics is available. Surrogate variants (a higher-fidelity model standing in for the oracle) are future work.
- **Stochastic envs where the success criterion is poorly defined** &mdash; the metric inherits whatever success rule the benchmark provides.
- **Single-run reporting at $n &lt; 10$** &mdash; the AC CI will refuse to commit. That is the correctly-calibrated behaviour, not a defect.
</section>

## Methodology we draw on

As physical-AI world models proliferate and get ranked on success-rate leaderboards, two ideas from that evaluation literature sharpen what CPG already does. We borrow the *methods*, not the metrics: pixel-fidelity scores (PSNR/SSIM/LPIPS, generation FID) are exactly the prediction-quality measures this page argues are silent on decisions, so they stay out.

- **Pairwise, matched-condition ranking.** RoboArena ([Atreya et al., 2025](https://arxiv.org/abs/2506.18123)) ranks generalist robot policies from *double-blind pairwise comparisons*, and flags that the Bradley-Terry assumption of identical conditions per comparison breaks when evaluators pick disparate tasks -- so they add a task-aware ranking. CPG's design already gives the matched condition for free: under varied-init sampling every model is scored on the *same* per-episode initial state, so each pairwise comparison is condition-matched by construction. `wmel.metrics.paired_bradley_terry_ranking` builds on this -- it ranks N models from their paired per-episode runs with a Bradley-Terry fit and a paired bootstrap CI on the *ranking itself*, with an optional `groups=` argument that stratifies the bootstrap by task so the rank interval reflects the task mix (a calibration correction on the interval only -- it does not change the point ranking). When two models' rank intervals overlap, the leaderboard cannot separate them -- the ranking analogue of an `INCONCLUSIVE` verdict.

  ```python
  from wmel.metrics import paired_bradley_terry_ranking
  ranking = paired_bradley_terry_ranking(
      {"oracle": oracle_results, "tdmpc2": tdmpc2_results, "mlp": mlp_results}
  )
  ranking.ranks       # {"oracle": 1, "tdmpc2": 2, "mlp": 3}
  ranking.rank_ci     # {"tdmpc2": (2, 3), ...}  -- overlap => not separable
  ```

- **Calibration as a stated design goal.** NVIDIA's Cosmos human-evaluation framework decomposes each output into atomic yes/no checks, with the stated rationale that "as SOTA models saturate existing automated leaderboards, score differences between releases are often too narrow for meaningful comparison." That is the same diagnosis as this framework's power analysis (a $0.94$-vs-$0.92$ tie at $n=100$ is noise), arrived at independently -- external corroboration that calibrated, interval-aware reading of a leaderboard is the right design target. A binary decomposition of the success criterion into per-condition checks, each with its own AC interval, is a natural future extension that keeps the binomial machinery intact.

## Source

- [`src/wmel/metrics.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/src/wmel/metrics.py) -- `CPGResult`, `counterfactual_planning_gap`, `cpg_verdict`, `paired_bradley_terry_ranking`.
- [`experiments/dmc_acrobot/cpg.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/experiments/dmc_acrobot/cpg.py) -- end-to-end run on DMC Acrobot-swingup.
- [`paper/main.tex`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/main.tex) -- short paper that formalises the metric and reports the worked example.
