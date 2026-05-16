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

## In one paragraph

Run the same planner on the same benchmark **twice**. Once with oracle dynamics (the true environment's transition function), once with a learned model. The only thing that changes between the two runs is the `dynamics=` callable passed to the planner. The success-rate difference is the Counterfactual Planning Gap (CPG). Report the **raw** point estimate, the **Agresti--Caffo $95\%$ confidence interval** on the gap, and a **five-branch verdict gated on the CI** -- not on the raw point estimate.

## Definition

$$
\mathrm{CPG} \;=\; \mathrm{success\_rate}(D^{\star}) \;-\; \mathrm{success\_rate}(D_\theta)
$$

with $D^{\star}$ the oracle dynamics and $D_\theta$ the learned model. All other quantities (env, planner, scoring function, $N$ episodes, horizon, seed) are held fixed between the two runs. The metric is a fraction in $[-1, +1]$.

## Why Agresti--Caffo, not Wald

The standard Wald $95\%$ CI on a difference of two binomial proportions uses

$$
\mathrm{SE}_{\mathrm{Wald}} \;=\; \sqrt{\frac{p_o(1-p_o)}{n_o} + \frac{p_\ell(1-p_\ell)}{n_\ell}}.
$$

When either arm sits at $p \in \{0, 1\}$ -- which happens routinely on hard control tasks at small $n$, exactly the regime this framework lands in -- the Wald variance collapses to zero, the interval shrinks to a point, and a meaningless "significant" result drops out.

The **Agresti--Caffo plus-4** adjustment fixes this by adding one pseudo-success and one pseudo-failure to each arm before computing the standard-normal CI:

$$
\tilde p = \frac{s + 1}{n + 2}, \qquad
\mathrm{SE}_{\mathrm{AC}} \;=\; \sqrt{\frac{\tilde p_o (1 - \tilde p_o)}{n_o + 2} + \frac{\tilde p_\ell (1 - \tilde p_\ell)}{n_\ell + 2}}
$$

$$
\mathrm{CI}_{95\%}(\mathrm{CPG}) \;=\; \bigl[\, \tilde\Delta - 1.96\,\mathrm{SE}_{\mathrm{AC}}, \;\; \tilde\Delta + 1.96\,\mathrm{SE}_{\mathrm{AC}} \,\bigr]
$$

The variance never collapses, coverage is honest down to single-digit $n$, and the interval converges to Wald for moderate-to-large samples. We report **both** the raw $\hat\Delta$ (what a reader expects) and the AC CI (what is defensible). They coincide for large $n$.

## Verdict (gated on the CI)

A CPG reported without a significance gate over-claims: a `gap = +0.1` from $n = 10$ is indistinguishable from noise but would otherwise read as a model bottleneck. The framework therefore exposes a five-branch verdict that consults the AC interval, **not** the raw point estimate.

<div class="verdict-legend" markdown="0">
  <p><span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span> &mdash; $\mathrm{CI}_{\mathrm{lo}} &gt; 0$. The oracle is reliably better; closing the gap is a model problem.</p>
  <p><span class="verdict-pill verdict-learned-outperforms">LEARNED OUTPERFORMS</span> &mdash; $\mathrm{CI}_{\mathrm{hi}} &lt; 0$. Rare; investigate regularisation or planner-search interactions.</p>
  <p><span class="verdict-pill verdict-planner-bottleneck">PLANNER BOTTLENECK</span> &mdash; CI crosses $0$ <em>and</em> both success rates are within $\tau$ of $0$. Neither planner solves the task; the framework needs a stronger search, not a stronger model.</p>
  <p><span class="verdict-pill verdict-as-good-as">MODEL AS GOOD AS ORACLE</span> &mdash; CI crosses $0$ <em>and</em> both success rates are within $\tau$ of $1$. The learned model matches the oracle for planning purposes on this task.</p>
  <p><span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span> &mdash; CI crosses $0$ in a middle-of-the-road regime. The sample size is insufficient to discriminate; report this verdict and run more episodes.</p>
</div>

The default tolerance is $\tau = 0.05$. Crucially, `MODEL BOTTLENECK` is **not** the default when $\hat\Delta > 0$ -- it requires the AC lower bound to be strictly positive.

## Worked example: DMC Acrobot-swingup at $n = 10$

The framework's reference run. Random-shooting MPC over a five-level torque discretisation, $50$ candidates of $15$-step horizon, $10$ episodes per arm, seed $0$.

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

The data is *suggestive* of a model bottleneck -- the raw point estimate is positive and large -- but with $n_\ell = 10$ and the learned arm reporting $0/10$, the AC CI cannot rule out zero. The honest call is to **run more episodes**. A multi-seed extension that pushes $N$ to $100$ episodes per arm would tighten the AC half-width by roughly $\sqrt{10}$, moving the lower bound either firmly above zero or covering it more thoroughly.

Numbers are pulled verbatim from [`results/dmc_acrobot/cpg.json`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/results/dmc_acrobot/cpg.json). Regenerate with:

```bash
pip install -e ".[dev,control,learned]"
python -m experiments.dmc_acrobot.cpg
```

## When to use CPG

- **Simulated environments** where an oracle dynamics is cheap to instantiate (most physics-based control tasks; gridworlds; OGBench-style tasks).
- **Comparing learned models** of different capacity, training-data budget, or architecture against the same oracle.
- **Decoupling diagnostics** when a model with a low validation MSE produces a planner that does not succeed -- CPG attributes the failure to the model or rules it out.

## When CPG is undefined

- **Hardware-in-the-loop / real-world environments** where no oracle dynamics is available. Surrogate variants (a higher-fidelity model standing in for the oracle) are future work.
- **Stochastic envs where the success criterion is poorly defined** -- the metric inherits whatever success rule the benchmark provides.
- **Single-run reporting at $n &lt; 10$** -- the AC CI will refuse to commit. That is the correctly-calibrated behaviour, not a defect.

## Source

- [`src/wmel/metrics.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/src/wmel/metrics.py) -- `CPGResult`, `counterfactual_planning_gap`, `cpg_verdict`.
- [`experiments/dmc_acrobot/cpg.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/experiments/dmc_acrobot/cpg.py) -- end-to-end run on DMC Acrobot-swingup.
- [`paper/main.tex`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/main.tex) -- short paper that formalises the metric and reports the worked example.
