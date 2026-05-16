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

with $D^{\star}$ the oracle dynamics and $D_\theta$ the learned model. All other quantities (env, planner, scoring function, $N$ episodes, horizon, seed) are held fixed between the two runs.

This identification &mdash; same planner, same scoring, same env, same seed, only `dynamics=` swapped &mdash; is what licenses interpreting CPG as a property of the *model*, not of the planner or the env.
</section>

<section class="chapter" id="why-ac" markdown="1">
  <p class="chapter-eyebrow">Step 02</p>
  <h2 class="chapter-title">Why Agresti--Caffo, not Wald</h2>
  <p class="chapter-lead">The standard Wald confidence interval collapses to a point at the boundary proportions $p \in \\{0, 1\\}$ &mdash; exactly the regime this framework lands in at small $n$. The Agresti--Caffo plus-4 adjustment fixes that.</p>

The standard Wald $95\%$ CI on a difference of two binomial proportions uses

$$
\mathrm{SE}_{\mathrm{Wald}} \;=\; \sqrt{\frac{p_o(1-p_o)}{n_o} + \frac{p_\ell(1-p_\ell)}{n_\ell}}.
$$

When either arm sits at $p \in \\{0, 1\\}$, the Wald variance collapses to zero, the interval shrinks to a point, and a meaningless "significant" result drops out. A learned planner that fails on every episode is precisely this regime.

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
  <h2 class="chapter-title">Worked example: DMC Acrobot-swingup at $n = 10$</h2>
  <p class="chapter-lead">The framework's reference run. Random-shooting MPC over a five-level torque discretisation, $50$ candidates of $15$-step horizon, $10$ episodes per arm, seed $0$.</p>

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

The data is *suggestive* of a model bottleneck &mdash; the raw point estimate is positive and large &mdash; but with $n_\ell = 10$ and the learned arm reporting $0/10$, the AC CI cannot rule out zero. The honest call is to **run more episodes**. A multi-seed extension that pushes $N$ to $100$ episodes per arm would tighten the AC half-width by roughly $\sqrt{10}$, moving the lower bound either firmly above zero or covering it more thoroughly.

Numbers are pulled verbatim from [`results/dmc_acrobot/cpg.json`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/results/dmc_acrobot/cpg.json). Regenerate with:

```bash
pip install -e ".[dev,control,learned]"
python -m experiments.dmc_acrobot.cpg
```
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

## Source

- [`src/wmel/metrics.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/src/wmel/metrics.py) -- `CPGResult`, `counterfactual_planning_gap`, `cpg_verdict`.
- [`experiments/dmc_acrobot/cpg.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/experiments/dmc_acrobot/cpg.py) -- end-to-end run on DMC Acrobot-swingup.
- [`paper/main.tex`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/main.tex) -- short paper that formalises the metric and reports the worked example.
