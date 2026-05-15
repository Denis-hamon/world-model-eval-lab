# 02 - Metric Taxonomy

This is the first pass of a product-grade metric set for action-conditioned world models. Each metric is chosen because it answers a question a product team would actually ask before integrating a model.

## Summary table

| Metric | Definition | Why it matters | Example measurement |
| --- | --- | --- | --- |
| Action Success Rate | Fraction of episodes in which the agent reaches the goal within the horizon. | The headline number. If this is near zero, nothing else matters. | Over 200 episodes of Two-Room with horizon 50, success rate = 0.87. |
| Planning Latency | Wall-clock time to produce one planned action sequence. **Reported per `plan()` call, not per episode.** | Tells you whether the model can close a control loop in real time. | mean = 2.4 ms per `plan()` call on the maze toy (CPU). |
| Compute per Decision | Estimated FLOPs or model forward passes per planned action. | Translates research compute into product cost (energy, dollars, GPU hours). | 1.2 model rollouts per decision, average horizon 8. |
| Planning Horizon | Effective look-ahead depth at which performance stops improving. | Tells you how far the model can usefully imagine before it becomes noise. | Success rate plateaus at horizon = 12; longer horizons add cost without value. |
| Perturbation Recovery | Success rate conditional on a perturbation event during the episode. | Measures robustness in the only way a product environment delivers it - by surprise. | Baseline success = 0.87; under perturbation = 0.61; recovery rate = 0.70. |
| Sample Efficiency | Performance as a function of training samples or environment interactions. | Distinguishes models that need a research-lab dataset from models that can ship. | Reaches 80 percent of asymptotic success with 5k transitions. |
| Surprise Detection | Ability of the model to flag observations its predictor finds unlikely. | A precondition for safe behaviour - "I do not know what is going on" is a feature. | AUROC = 0.78 on held-out anomalous frames vs in-distribution frames. |
| Latent Interpretability | Degree to which the latent state exposes task-relevant structure. | Helps debugging, safety review, and integration with classical control. | Linear probe on latent predicts agent position with R^2 = 0.93. |

## Planning-horizon curve (worked example)

The "Planning Horizon" metric is operationalised by `wmel.experiments.horizon_sweep`. Running it on the maze toy environment with `TabularWorldModelPlanner` produces a textbook curve - per-call planning latency grows monotonically with horizon, success rate plateaus, and beyond the plateau steps-to-success starts to degrade because the planner over-commits before replanning:

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

This taxonomy is intentionally a starting point. Additions are welcome, but every new metric should answer a product question, come with an example measurement, and have a corresponding test on synthetic data.
