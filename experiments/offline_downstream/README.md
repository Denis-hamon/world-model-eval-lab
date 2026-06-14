# Offline metric → downstream performance (the external-validity keystone)

Does a *cheap, planner-free offline metric* predict a model's *downstream
decision quality* (planning success / CPG)? If yes, you can screen world models
without the cost of closed-loop planning; if no, closed-loop evaluation (CPG) is
irreplaceable. Either answer is a result — and it is the external-validity test
that turns this lab from a tool into a finding.

The pipeline is two stages:

1. **Offline-scores sweep** (has-deps): for each evaluated cell, train/load a
   model, compute planner-free offline metrics, run the planner to get the
   downstream value, and write one row per cell to a bundle JSON.
2. **`correlate.py`** (stdlib, no GPU): read the bundle and report, per offline
   metric, its rank correlation with the downstream target + a bootstrap CI
   (`wmel.metrics.bootstrap_correlation_ci`).

## Offline metrics

- **M1 one-step error** — naive reconstruction fidelity (the *foil*).
- **M2 k-step open-loop divergence** — compounding error over a fixed action
  sequence.
- **M3 action-ranking agreement** — does the model rank actions by
  closeness-to-goal the way the truth does? The *decision-aware* metric: a model
  can have large M1 yet small M3, and only M3 is what a planner needs.

The discriminating question is whether the decision-aware M3 predicts downstream
success where the naive M1 does not.

## Stage 1 — CPU proof of concept (maze): `maze_quality_sweep.py`

```bash
pip install -e ".[learned,dev]"
python -m experiments.offline_downstream.maze_quality_sweep      # writes results/offline_downstream/maze_offline_scores.json
python -m experiments.offline_downstream.correlate              # writes results/offline_downstream/offline_vs_downstream.json
```

The learned MLP arms on DMC sit at the planning floor (success 0 regardless of
prediction quality — the dissociation the paper reports), so they give no
downstream spread to correlate against. The maze toy is the CPU-feasible setting
where under-training the learned dynamics genuinely degrades planning, so
downstream success varies and the pipeline can be exercised end to end. The
quality knob is training `epochs`.

### Result (this CPU PoC)

All three offline metrics predict downstream success and their intervals clear
zero. On the **fair common subset** (n=17 -- the cells where every metric is
defined) the rank correlations are essentially equal: M1 rho=-0.90, M2 rho=-0.91,
M3 rho=+0.90. So the naive one-step metric is **not** inferior to the
decision-aware M3 here.

**Do not read the per-metric view as "M3 strongest."** The per-metric rows use
each metric's own usable cells (M1/M2 on n=24, M3 on n=17, because M3 is
undefined for the action-blind worst models -- which are all failures). Comparing
those magnitudes across different subsets is invalid; the apparent M3 > M1 gap is
an artifact of M3's smaller, easier subset. Only the common-subset block is a
fair head-to-head, and there the metrics tie. `correlate.py` prints both and
stores a `comparability_note`.

**Scope honesty.** Maze is a small deterministic env where an accurate model is
*sufficient* for planning -- a "prediction approx decision" regime -- which is
exactly why all three metrics (including naive M1) predict equally well. The maze
PoC validates the pipeline and shows the positive regime; it does **not** settle
the M1-vs-M3 question. Three further caveats: M1 and M2 are rank-redundant
(rho ~ 0.98, effectively one metric); downstream success is near-bimodal (only
the epochs=64 row is intermediate), so the correlation mostly separates good- vs
bad-quality models rather than grading them; and M2 uses a single fixed action
sequence. The discriminating test -- whether decision-aware M3 predicts where
naive M1 fails -- needs Stage 2 (DMC/TD-MPC2), where capable models span a range
of non-floor downstream success.

## Stage 2 — the headline cells (GPU): a DMC/TD-MPC2 sweep

The discriminating test needs cells where *capable* models span a quality range
with non-floor, varying downstream success — i.e. TD-MPC2 at several capacities /
checkpoints across Acrobot, Cartpole, Reacher (the cells already in `results/`,
whose downstream CPG ranges from −0.27 to +0.93). Computing M1–M3 for those
requires loading the TD-MPC2 checkpoints (GPU; not committed), so this stage runs
on the GPU box. Its output appends rows to the same bundle schema, after which
`correlate.py` reproduces the correlation with no GPU.

## Bundle schema

`results/offline_downstream/*offline_scores.json`:

```json
{
  "cells": [
    {"<descriptor fields>": "...", "m1_...": 0.0, "m2_...": 0.0, "m3_...": 0.0,
     "success_rate": 0.0}
  ]
}
```

`correlate.py` auto-detects numeric metric columns (excluding descriptors like
`epochs`/`seed`/`model`), correlates each against `--downstream` (default
`success_rate`; use `cpg` for the DMC bundle), and drops non-finite cells per
metric.
