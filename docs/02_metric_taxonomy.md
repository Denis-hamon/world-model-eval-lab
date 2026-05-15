---
prev:
  title: "01 - The evaluation gap"
  url: 01_evaluation_gap.html
next:
  title: "03 - Benchmark cards"
  url: 03_benchmark_cards.html
---
# 02 - Metric Taxonomy

This is the first pass of a decision-grade metric set for action-conditioned world models. Each metric is chosen because it answers a question an applied team would actually ask before integrating a model.

## Summary table

Hover any metric name for a one-line reading; click to jump to the full definition and formula at the [bottom of the page](#definitions-and-formulas).

| Metric | Definition | Why it matters | Example measurement |
| --- | --- | --- | --- |
| [Action Success Rate](#m-success-rate "Fraction of episodes that reached the goal."){:.metric-link} | Fraction of episodes in which the agent reaches the goal within the horizon. | The headline number. If this is near zero, nothing else matters. | Over 200 episodes of Two-Room with horizon 50, success rate = 0.87. |
| [Planning Latency](#m-planning-latency "Mean wall-clock time per plan() call, across the whole run."){:.metric-link} | Wall-clock time to produce one planned action sequence. **Reported per `plan()` call, not per episode.** | Tells you whether the model can close a control loop in real time. | mean = 2.4 ms per `plan()` call on the maze toy (CPU). |
| [Compute per Decision](#m-compute-per-decision "Model work per executed action, in policy-declared units."){:.metric-link} | Estimated FLOPs or model forward passes per planned action. | Translates research compute into product cost (energy, dollars, GPU hours). | 1.2 model rollouts per decision, average horizon 8. |
| [Planning Horizon](#m-planning-horizon "Effective lookahead depth at which success stops improving."){:.metric-link} | Effective look-ahead depth at which performance stops improving. | Tells you how far the model can usefully imagine before it becomes noise. | Success rate plateaus at horizon = 12; longer horizons add cost without value. |
| [Perturbation Recovery](#m-perturbation-recovery "Of episodes that actually got perturbed, the fraction that still succeeded."){:.metric-link} | Success rate conditional on a perturbation event during the episode. | Measures robustness in the only way a product environment delivers it - by surprise. | Baseline success = 0.87; under perturbation = 0.61; recovery rate = 0.70. |
| [Sample Efficiency](#m-sample-efficiency "Performance as a function of training samples or interactions."){:.metric-link} | Performance as a function of training samples or environment interactions. | Distinguishes models that need a research-lab dataset from models that can ship. | Reaches 80 percent of asymptotic success with 5k transitions. |
| [Surprise Detection](#m-surprise-detection "How well the model flags out-of-distribution inputs."){:.metric-link} | Ability of the model to flag observations its predictor finds unlikely. | A precondition for safe behaviour - "I do not know what is going on" is a feature. | AUROC = 0.78 on held-out anomalous frames vs in-distribution frames. |
| [Latent Interpretability](#m-latent-interpretability "Whether the latent state exposes task-relevant structure."){:.metric-link} | Degree to which the latent state exposes task-relevant structure. | Helps debugging, safety review, and integration with classical control. | Linear probe on latent predicts agent position with R^2 = 0.93. |

## Planning-horizon curve (worked example)

The "Planning Horizon" metric is operationalised by `wmel.experiments.horizon_sweep`. Running it on the maze toy environment with `TabularWorldModelPlanner` produces a textbook curve - per-call planning latency grows monotonically with horizon, success rate plateaus, and beyond the plateau steps-to-success starts to degrade because the planner over-commits before replanning:

![horizon sweep](assets/horizon_sweep.svg)

```
Horizon sweep: tabular-world-model
  plan_h |   success |          95% CI |   steps | latency_ms |       95% CI (ms)
  -------------------------------------------------------------------------------
       5 |     0.000 | [0.00, 0.11]   |     n/a |      0.882 | [0.87, 0.89]
      10 |     0.900 | [0.74, 0.97]   |    31.3 |      1.588 | [1.56, 1.62]
      15 |     1.000 | [0.89, 1.00]   |    30.5 |      2.393 | [2.34, 2.44]
      20 |     1.000 | [0.89, 1.00]   |    33.8 |      3.085 | [3.08, 3.09]
      30 |     1.000 | [0.89, 1.00]   |    41.8 |      4.579 | [4.55, 4.60]
```

Reading the curve:

- `plan_h=5` is too shallow to find a solution. Success is 0 percent.
- `plan_h=10` mostly works (90 percent success) but is brittle.
- `plan_h=15` matches the maze's optimal path length and saturates at 100 percent.
- Past the plateau, latency keeps rising while success does not move and steps-to-success degrades - a clean illustration of the "useful look-ahead depth" the metric is meant to expose.

Latency is measured **per `plan()` call** (the unit the metric is defined in), not per episode. Replanning more often does not earn a policy a free latency discount. The success-rate column uses a Wilson score interval; the latency column uses a normal interval on the sample mean. The same script writes the full report as JSON to `examples/maze_toy/horizon_sweep_report.json` for downstream tooling.

Reproduce with:

```bash
python -m examples.maze_toy.run_horizon_sweep
```

### Paste-ready Markdown

`wmel.experiments.to_markdown_horizon_sweep(sweep)` returns the same data as a Markdown table - including the compute-per-decision column - that drops directly into a PR description or a doc:

```markdown
### Horizon sweep: `tabular-world-model`

| plan_horizon | success_rate | success_95ci | avg_steps | latency_ms_per_call | latency_95ci | compute_per_decision |
| ---: | ---: | :--- | ---: | ---: | :--- | ---: |
| 5 | 0.000 | [0.00, 0.11] | n/a | 0.875 | [0.87, 0.89] | 368.250 |
| 10 | 0.900 | [0.74, 0.97] | 31.3 | 1.578 | [1.55, 1.61] | 350.575 |
| 15 | 1.000 | [0.89, 1.00] | 30.5 | 2.348 | [2.34, 2.36] | 278.689 |
| 20 | 1.000 | [0.89, 1.00] | 33.8 | 3.096 | [3.07, 3.12] | 256.410 |
| 30 | 1.000 | [0.89, 1.00] | 41.8 | 4.614 | [4.55, 4.68] | 277.512 |
```

The three metric dimensions - planning horizon, latency per call, and compute per decision - now appear together on one row, which is the trade-off surface this taxonomy advocates.

`wmel.report.to_markdown_scorecard(scorecard)` does the same for a single scorecard.

## Notes

- **Latency, compute, and horizon** form a single trade-off surface. A useful scorecard reports them together, not in isolation.
- **Perturbation Recovery** requires a perturbation library. `wmel.perturbations` ships three composable types: `EnvPerturbation` (delegates to `env.perturb()`), `DropNextActions(k)` (action-level - simulates actuator drops by removing the next `k` queued actions), and `CompositePerturbation(*parts)` (chains the first two for combined failure modes). `BenchmarkRunner` takes a `perturbation` kwarg and records the chosen strategy on the `Scorecard`, so a policy benchmarked under different perturbations produces distinguishable scorecards.
- **Surprise Detection** and **Latent Interpretability** are model-level diagnostics. They are part of the scorecard because they are precisely what a research-grade predictor is *supposed* to be good at - if it is not, that is itself a finding.
- All metrics should be reported with seeds, sample sizes, and confidence intervals when feasible.

## Versioning

This taxonomy is intentionally a starting point. Additions are welcome, but every new metric should answer an applied question, come with an example measurement, and have a corresponding test on synthetic data.

---

## Definitions and formulas

Each metric below is written as a ratio of plain-English quantities first, then in math notation. Skip this section on a first read - the [summary table](#summary-table) and the [worked example](#planning-horizon-curve-worked-example) carry the same information without symbols.

### Action success rate {#m-success-rate}

Reads as: "How often did the agent reach the goal?"

$$
\text{success\_rate} \;=\; \frac{\text{episodes that succeeded}}{\text{episodes total}}
$$

Bounded in $[0, 1]$. Headline number. If this is near zero, no other metric matters.

### Average steps to success {#m-avg-steps}

Reads as: "Among the episodes that succeeded, how many steps did the agent take on average?"

$$
\text{avg\_steps} \;=\; \frac{\text{total steps in successful episodes}}{\text{number of successful episodes}}
$$

Reported as `n/a` when zero episodes succeeded. Lower is better, with the maze's optimal path (14) as the theoretical floor.

### Planning latency, per call {#m-planning-latency}

Reads as: "How long does a single `plan()` invocation take on average?"

$$
\bar{\ell} \;=\; \frac{\text{total milliseconds spent in plan()}}{\text{total number of plan() calls}}
$$

Both sums are taken across **all episodes in the run**, then divided. A policy that replans 10 times per episode and a policy that replans once contribute equally per call, so neither hides behind a per-episode mean. This is the v0.3.1 fix: the previous reporting used per-episode totals and biased latency comparisons.

### Compute per decision {#m-compute-per-decision}

Reads as: "How much model work, in policy-declared units, does it take to produce one executed action?"

$$
\bar{c} \;=\; \frac{c_{\text{plan}} \;\times\; \text{total number of plan() calls}}{\text{total number of executed steps}}
$$

$c_{\text{plan}}$ is the policy's declared cost per `plan()` call (FLOPs, model forward passes, rollouts - whatever unit makes sense). For `TabularWorldModelPlanner`:

$$
c_{\text{plan}} \;=\; \text{num\_candidates} \;\times\; \text{plan\_horizon}
$$

so the maze run at horizon 15 with 200 candidates yields $c_{\text{plan}} = 3000$ rollout-units per call. Divided by the actual steps executed, that comes out to $\bar{c} \approx 280$ rollout-units per decision.

### Planning horizon (effective) {#m-planning-horizon}

Reads as: "What is the smallest lookahead beyond which success rate stops improving?"

In words: pick a tolerance $\epsilon$ (typically $0.01$ or smaller). $H^{\ast}$ is the smallest lookahead such that any deeper lookahead would buy less than $\epsilon$ extra success rate.

$$
H^{\ast} \;=\; \min \Bigl\{ H \;:\; \text{success\_rate}(H') - \text{success\_rate}(H) \leq \epsilon \;\; \text{for all } H' > H \Bigr\}
$$

For the maze toy with $\epsilon = 0.01$: $H^{\ast} = 15$, exactly one step past the maze's optimal-path length of 14. Below $H^{\ast}$ you lose success; above it you spend more latency and compute for nothing.

### Perturbation recovery rate {#m-perturbation-recovery}

Reads as: "Of the episodes where `env.perturb()` actually fired, what fraction still reached the goal?"

$$
r \;=\; \frac{\text{actually-perturbed episodes that still succeeded}}{\text{actually-perturbed episodes}}
$$

"Actually-perturbed" is stricter than "scheduled for perturbation": episodes that succeed before the perturbation step are not counted. This keeps the denominator honest at the cost of a smaller effective sample when policies are very fast. (This was the v0.3 bug fixed in v0.3.1.)

### Wilson 95% confidence interval (for success rate) {#m-wilson}

Reads as: "Given $\hat{p}$ observed successes out of $n$ episodes, the lower and upper bounds of success rate I can defend at 95% confidence."

The interval is **asymmetric** around $\hat{p}$, which is exactly what you want near 0% and 100% where the textbook normal approximation predicts impossible values (success rates below 0 or above 1).

$$
\hat{p}_{\text{lo}}, \hat{p}_{\text{hi}} \;=\; \frac{\hat{p} + \dfrac{z^{2}}{2n} \;\pm\; z \sqrt{\dfrac{\hat{p}(1-\hat{p})}{n} + \dfrac{z^{2}}{4n^{2}}}}{1 + \dfrac{z^{2}}{n}}
$$

with $z = 1.96$ for a two-sided 95% interval. Reading guide:

- The numerator $\hat{p} + \tfrac{z^{2}}{2n}$ is the **centre** of the interval - shifted away from $\hat{p}$ near the extremes.
- The $\pm z\sqrt{\cdot}$ piece is the **half-width**.
- The denominator $1 + \tfrac{z^{2}}{n}$ shrinks toward $1$ as $n$ grows, so for large $n$ the Wilson interval collapses to the normal one.

For $\hat{p} = 1.0$ and $n = 30$ this gives $[0.89, 1.00]$. To tighten the lower bound to $0.95$ at the same observed rate, you need $n \gtrsim 73$.

### Normal 95% confidence interval (for mean latency) {#m-normal-ci}

Reads as: "The range around the observed mean latency where 95% of intervals constructed this way would contain the true mean."

$$
\bar{\ell} \;\pm\; 1.96 \;\cdot\; \frac{\sigma_{\ell}}{\sqrt{n_{\ell}}}
$$

with $\sigma_{\ell}$ the standard deviation of the per-call latency samples (flattened across episodes) and $n_{\ell}$ the total number of plan() calls. The normal approximation is fine here because latencies are bounded away from 0 and we typically have many samples.

### Sample efficiency {#m-sample-efficiency}

Not formalised in this taxonomy yet (no off-the-shelf metric works across all training regimes). Track it as a learning curve: success rate as a function of training samples or environment interactions. Report the sample count at which the model reaches a fixed fraction (typically 0.8) of its asymptotic success.

### Surprise detection {#m-surprise-detection}

Reads as: "Can the model tell when an observation is out of its training distribution?" Operationalised as AUROC on a held-out set of anomalous-vs-in-distribution inputs:

$$
\text{AUROC} \;=\; \Pr \bigl[ \text{score}(\text{anomalous}) \;>\; \text{score}(\text{in-distribution}) \bigr]
$$

where `score` is a model-defined surprise signal (negative log-likelihood of the prediction, distance from the latent prior, etc.). 0.5 is random, 1.0 is perfect ranking.

### Latent interpretability {#m-latent-interpretability}

No single formula. Typically reported as the $R^{2}$ of a linear probe predicting a task-relevant variable (agent position, object pose) from the latent state. Higher is more structured; very high values may indicate the latent is just the input.