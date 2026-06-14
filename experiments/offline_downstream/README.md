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

## Stage 2 — the headline cells (CPU, checkpoint-gated): `tdmpc2_offline_metrics.py`

The discriminating test needs cells where *capable* models span a quality range
with non-floor, varying downstream success — i.e. the committed TD-MPC2 cells
across Acrobot, Cartpole, Reacher, whose downstream CPG spans roughly −0.10 to
+0.80. `tdmpc2_offline_metrics.py` loads each cell's trained latent dynamics and
computes M1/M2/M3 against the matched oracle on a fixed seeded state sample, then
pairs them with that cell's already committed CPG (`results/dmc_*/tdmpc2_cpg*.json`).

This is **CPU**, not GPU: TD-MPC2 inference defaults to CPU and the planner is
not run here — only the `(state, action) → next_state` callable is queried. It is
**checkpoint-gated**: the `.pt` weights are gitignored and not in the repo. Run it
where they exist (the training box, or after `scp`-ing the checkpoints into
`results/dmc_*/`), with the TD-MPC2 setup done (`scripts/setup_tdmpc2.sh` + the
deps listed in `experiments/dmc_reacher/tdmpc2_cpg.py`). On a checkout without the
weights every cell is reported as skipped, so the script still runs and tells you
exactly which files are missing.

```bash
python -m experiments.offline_downstream.tdmpc2_offline_metrics    # writes results/offline_downstream/tdmpc2_offline_scores.json
python -m experiments.offline_downstream.correlate \
    --bundle results/offline_downstream/tdmpc2_offline_scores.json --downstream gap
```

**Scale confound — read this before pooling.** M1 and M2 are L2 distances in each
environment's own state space, so their magnitudes are **not** comparable across
Cartpole and Reacher; pooling them into one cross-env correlation would unfairly
handicap them against the unitless, rank-based M3 — the same apples-to-oranges
trap the maze PoC review flagged. The fair head-to-head is **within an
environment**, so `tdmpc2_offline_metrics.py` prints a within-env Spearman for
every env with ≥3 cells; the pooled `correlate.py` output is meaningful only for
M3. Cartpole has the widest within-env gap spread (size 1 and size 5, six cells)
and is the cell group where the M1-vs-M3 question can actually be decided.

## Bundle schema

`results/offline_downstream/*offline_scores.json`:

The maze bundle (Stage 1) is flat with a `success_rate` downstream:

```json
{"cells": [{"epochs": 64, "seed": 0, "m1_mismatch": 0.0, "m2_kstep_divergence": 0.0,
            "m3_action_agreement": 0.0, "success_rate": 0.0}]}
```

The TD-MPC2 bundle (Stage 2) carries the offline metrics + `gap` flat, and parks
every non-metric number inside `context` so the auto-detector sees only m1/m2/m3:

```json
{"cells": [{"env": "dmc_cartpole", "model_size": 5, "seed": 0, "verdict": "MODEL BOTTLENECK",
            "m1_l2_onestep": 0.0, "m2_l2_kstep": 0.0, "m3_action_agreement": 0.0, "gap": 0.5,
            "context": {"oracle_success_rate": 0.0, "learned_success_rate": 0.0,
                        "training_steps": 0, "n_states": 150, "k_steps": 10}}]}
```

`correlate.py` auto-detects numeric metric columns (excluding descriptors like
`epochs`/`seed`/`model`), correlates each against `--downstream` (default
`success_rate`; use `gap` for the TD-MPC2 bundle), and drops non-finite cells per
metric. The TD-MPC2 bundle keeps non-metric numbers (success rates, training
steps, state count) inside a per-cell `context` object so the auto-detector does
not mistake them for offline metrics, and stores the within-env correlation under
`within_env_correlation`.
