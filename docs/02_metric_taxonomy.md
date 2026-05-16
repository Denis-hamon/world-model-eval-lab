---
prev:
  title: "01 - The evaluation gap"
  url: 01_evaluation_gap.html
next:
  title: "03 - Benchmark cards"
  url: 03_benchmark_cards.html
---
# 02 - Metric Taxonomy

This is the first pass of a **decision-grade** metric set for action-conditioned world models. Each metric is chosen because it answers a question an applied team would actually ask before integrating a model.

## What "decision-grade" means here

A metric is **decision-grade** when both of the following hold:

1. **Its units translate directly to a deployment-time cost or capability.** Success rate is a fraction in `[0, 1]`. Planning latency is in milliseconds. Compute per decision is in policy-declared units (FLOPs, model forward passes, rollouts). Perturbation recovery is a fraction. All of these are quantities a procurement, robotics, or controls team can act on without further conversion.

2. **It is computable from a closed-loop run of the model, not from the model in isolation.** Reconstruction loss, FID, next-frame prediction error, and embedding-distance benchmarks are *model-internal* quantities - they describe how well the model fits its training distribution. A decision-grade metric requires the model to be *used* (encoded, rolled out, scored, planned) inside an environment, and reports what happened to the agent as a result.

The two criteria together exclude both pure prediction quality (which fails criterion 2) and abstract "alignment" or "interpretability" scores that do not translate into shippable units (which fail criterion 1).

The taxonomy below lists the metrics that meet both criteria in this study's first pass. Additions are welcome; the contribution procedure in [CONTRIBUTING.md](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/CONTRIBUTING.md) requires every proposed metric to pass the same two-criterion test.

## Summary table

Hover (or focus) any metric name to see its formula in a popover. The popover is the canonical definition; this page does not duplicate the formulas anywhere else.

<table class="metric-table">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Definition</th>
      <th>Why it matters</th>
      <th>Example measurement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Action Success Rate</a>
        <div class="formula-popover" role="tooltip">
          <p class="popover-reads">Reads as: how often did the agent reach the goal?</p>
          $$\text{success\_rate} \;=\; \frac{\text{episodes that succeeded}}{\text{episodes total}}$$
          <p class="popover-note">Bounded in $[0, 1]$. If this is near zero, no other metric matters.</p>
        </div>
      </td>
      <td>Fraction of episodes in which the agent reaches the goal within the horizon.</td>
      <td>The headline number. If this is near zero, nothing else matters.</td>
      <td>Over 200 episodes of Two-Room with horizon 50, success rate = 0.87.</td>
    </tr>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Planning Latency</a>
        <div class="formula-popover" role="tooltip">
          <p class="popover-reads">Reads as: how long does a single <code>plan()</code> call take, on average?</p>
          $$\bar{\ell} \;=\; \frac{\text{total ms spent in plan()}}{\text{total number of plan() calls}}$$
          <p class="popover-note">Per call, not per episode. A policy that replans more often cannot hide behind a per-episode mean.</p>
        </div>
      </td>
      <td>Wall-clock time to produce one planned action sequence. <strong>Reported per <code>plan()</code> call, not per episode.</strong></td>
      <td>Tells you whether the model can close a control loop in real time.</td>
      <td>mean = 2.4 ms per <code>plan()</code> call on the maze toy (CPU).</td>
    </tr>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Compute per Decision</a>
        <div class="formula-popover" role="tooltip">
          <p class="popover-reads">Reads as: how much model work, in policy-declared units, does one executed action take?</p>
          $$\bar{c} \;=\; \frac{c_{\text{plan}} \;\times\; \text{total plan() calls}}{\text{total executed steps}}$$
          <p class="popover-note">For <code>TabularWorldModelPlanner</code>: $c_{\text{plan}} = N_{\text{cand}} \cdot H_{\text{plan}}$ rollout-units.</p>
        </div>
      </td>
      <td>Estimated FLOPs or model forward passes per planned action.</td>
      <td>Translates research compute into product cost (energy, dollars, GPU hours).</td>
      <td>1.2 model rollouts per decision, average horizon 8.</td>
    </tr>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Planning Horizon</a>
        <div class="formula-popover" role="tooltip">
          <p class="popover-reads">Reads as: smallest lookahead beyond which a deeper search does not buy meaningfully more success.</p>
          $$H^{\ast} \;=\; \min\Bigl\{\,H \,:\; \text{success}(H') - \text{success}(H) \leq \epsilon \;\;\forall\, H' > H\,\Bigr\}$$
          <p class="popover-note">For the maze toy with $\epsilon = 0.01$: $H^{\ast} = 15$, one step past the maze's optimal-path length.</p>
        </div>
      </td>
      <td>Effective look-ahead depth at which performance stops improving.</td>
      <td>Tells you how far the model can usefully imagine before it becomes noise.</td>
      <td>Success rate plateaus at horizon = 12; longer horizons add cost without value.</td>
    </tr>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Perturbation Recovery</a>
        <div class="formula-popover" role="tooltip">
          <p class="popover-reads">Reads as: of episodes where <code>env.perturb()</code> actually fired, the fraction that still reached the goal.</p>
          $$r \;=\; \frac{\text{actually-perturbed episodes that succeeded}}{\text{actually-perturbed episodes}}$$
          <p class="popover-note">"Actually-perturbed" excludes episodes that succeed before the perturbation step (v0.3.1 fix).</p>
        </div>
      </td>
      <td>Success rate conditional on a perturbation event during the episode.</td>
      <td>Measures robustness in the only way a real environment delivers it - by surprise.</td>
      <td>Baseline success = 0.87; under perturbation = 0.61; recovery rate = 0.70.</td>
    </tr>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Sample Efficiency</a>
        <div class="formula-popover" role="tooltip">
          <p class="popover-reads">Reads as: performance as a function of training samples or environment interactions.</p>
          <p class="popover-note">No single closed-form formula. Reported as the sample count at which the model reaches a fixed fraction (typically 0.8) of its asymptotic success rate. Track the success-rate-vs-samples curve.</p>
        </div>
      </td>
      <td>Performance as a function of training samples or environment interactions.</td>
      <td>Distinguishes models that need a research-lab dataset from models that can ship.</td>
      <td>Reaches 80 percent of asymptotic success with 5k transitions.</td>
    </tr>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Surprise Detection</a>
        <div class="formula-popover" role="tooltip">
          <p class="popover-reads">Reads as: how well does the model flag out-of-distribution inputs?</p>
          $$\text{AUROC} \;=\; \Pr\!\bigl[\,\text{score}(\text{anomalous}) \,>\, \text{score}(\text{in-distribution})\,\bigr]$$
          <p class="popover-note">$0.5$ is random ranking, $1.0$ is perfect. <code>score</code> is a model-defined surprise signal (negative log-likelihood, distance from latent prior, etc.).</p>
        </div>
      </td>
      <td>Ability of the model to flag observations its predictor finds unlikely.</td>
      <td>A precondition for safe behaviour - "I do not know what is going on" is a feature.</td>
      <td>AUROC = 0.78 on held-out anomalous frames vs in-distribution frames.</td>
    </tr>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Latent Interpretability</a>
        <div class="formula-popover" role="tooltip">
          <p class="popover-reads">Reads as: does the latent state expose task-relevant structure?</p>
          $$R^{2} \;=\; 1 \;-\; \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$
          <p class="popover-note">Typically reported as the $R^{2}$ of a linear probe predicting a task-relevant variable (agent position, object pose) from the latent. Very high values may indicate the latent is just the input.</p>
        </div>
      </td>
      <td>Degree to which the latent state exposes task-relevant structure.</td>
      <td>Helps debugging, safety review, and integration with classical control.</td>
      <td>Linear probe on latent predicts agent position with $R^2 = 0.93$.</td>
    </tr>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Wilson 95% interval</a>
        <div class="formula-popover formula-popover-wide" role="tooltip">
          <p class="popover-reads">Reads as: lower and upper bounds for the success rate, defendable at 95% confidence. Asymmetric near 0% and 100% (which is where horizon sweeps spend most of their data).</p>
          $$\hat{p}_{\text{lo}}, \hat{p}_{\text{hi}} \;=\; \frac{\hat{p} + \dfrac{z^{2}}{2n} \;\pm\; z\sqrt{\dfrac{\hat{p}(1-\hat{p})}{n} + \dfrac{z^{2}}{4n^{2}}}}{1 + \dfrac{z^{2}}{n}}$$
          <p class="popover-note">$z = 1.96$ for two-sided 95%. At $\hat{p} = 1$, $n = 30$: $[0.89, 1.00]$. To push the lower bound to $0.95$ at the same $\hat{p}$: $n \geq 73$.</p>
        </div>
      </td>
      <td>Lower and upper bounds for the observed success rate, asymmetric near the extremes.</td>
      <td>Tells you what reliability you can defend to a procurement or regulatory team, not just the point estimate.</td>
      <td>$\hat{p} = 1.00$ over $n = 30$ gives $[0.89, 1.00]$ at 95% confidence.</td>
    </tr>
    <tr>
      <td class="metric-cell">
        <a href="#" class="metric-link" tabindex="0">Normal CI on mean latency</a>
        <div class="formula-popover" role="tooltip">
          <p class="popover-reads">Reads as: the range around the observed mean latency where 95% of constructed intervals would contain the true mean.</p>
          $$\bar{\ell} \;\pm\; 1.96 \cdot \frac{\sigma_{\ell}}{\sqrt{n_{\ell}}}$$
          <p class="popover-note">$\sigma_{\ell}$ is the standard deviation of the per-call latencies; $n_{\ell}$ is the total number of <code>plan()</code> calls across all episodes.</p>
        </div>
      </td>
      <td>Symmetric interval on the mean per-call latency.</td>
      <td>Normal works here because latencies are bounded away from 0 and we typically have many samples.</td>
      <td>$\bar{\ell} = 2.35 \pm 0.05$ ms per call on the maze toy at horizon 15.</td>
    </tr>
  </tbody>
</table>

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
