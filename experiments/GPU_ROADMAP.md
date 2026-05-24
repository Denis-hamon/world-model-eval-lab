# GPU experiment queue

This document tracks GPU workloads queued for `world-model-eval-lab`. Each entry is self-contained enough to be the body of a "go do this" prompt for the Claude Code session running on the OVH RTX 5000 instance. Tasks are ranked by impact-per-GPU-hour; dispatch them in order unless a specific scientific question takes priority.

**Conventions for any GPU job started from this queue**:

- Branch naming: `phase-5N-XXX` where `N` continues the existing sequence. Last shipped on main: `phase-5m-cartpole`. The next available letter is `5n`.
- Git identity: `Denis Hamon <denis.hamon1@gmail.com>`. Use `git -c user.name="Denis Hamon" -c user.email="denis.hamon1@gmail.com" commit`.
- Zero emojis anywhere (code, commits, PR descriptions, prints). No exception.
- Pre-tag adversarial review: when the experiment converges and tests pass, spawn a general-purpose Agent to review the diff. Address findings before opening the PR.
- Open the PR as DRAFT first. Let the user flip it to ready-for-merge.
- Never commit credentials or `.env` files. If you see `terraform.tfvars` or `OVH_*` env vars, ignore them.
- Long-running jobs go in `tmux` or `nohup` so SSH disconnects do not kill them.
- The CI `paper-pdf` workflow regenerates `main.pdf` and `main.bbl` after merge; do not commit those by hand.

When you pick up a task, change its `Status:` from `queued` to `in_flight` in this file (separate commit on the same branch). When it merges, change to `done` and link the PR.

---

## Task 1 — Cartpole size=1 seed 0 + pooled n=30

**Why**: v0.15 §5.10 reports Cartpole at size=5 fully pooled (n=30) but size=1 only has seeds 1+2 landed (n=20, no pooled JSON). Completing seed 0 plus producing the size=1 pooled-30 JSONs closes the capacity symmetry the table currently lacks.

**Branch**: `phase-5n-cartpole-size1-seed0`
**Effort**: ~6-10 h GPU
**Priority**: HIGH (smallest delta, finishes a half-shipped story)
**Status**: queued

**Steps**:

1. On the VM, `git pull` so you have the v0.15 baseline.
2. Reuse `experiments/dmc_cartpole/tdmpc2_cpg.py` to train TD-MPC2 with `model_size=1`, seed `0`, `1_000_000` env steps. The script's existing `--seed` and `--model-size` flags should already cover this.
3. Run the per-seed CPG arms for seed 0 at size=1:
   - `python -m experiments.dmc_cartpole.tdmpc2_cpg --seed 0 --model-size 1 ...`
   - `python -m experiments.dmc_cartpole.cem_cpg --seed 0 --model-size 1 ...`
   - `python -m experiments.dmc_cartpole.coverage_mlp_on_tdmpc2 --seed 0 --model-size 1 ...`
4. Pool to n=30 with `experiments/dmc_cartpole/pool_cpg.py` across seeds 0+1+2.

**Expected outputs** (committed to main via PR):

- `results/dmc_cartpole/tdmpc2_cpg_seed0.json`
- `results/dmc_cartpole/cem_cpg_seed0.json`
- `results/dmc_cartpole/coverage_mlp_on_tdmpc2_cpg_seed0.json`
- `results/dmc_cartpole/tdmpc2_cpg_pooled.json` (size=1, no `size5_` suffix)
- `results/dmc_cartpole/cem_cpg_pooled.json` (size=1)
- `results/dmc_cartpole/coverage_mlp_on_tdmpc2_cpg_pooled.json` (size=1)

**Paper update**:

- Extend §5.10's table or add a paragraph noting the size=1 cell with the same four-arm structure at n=30.
- The expected qualitative result is the same `MODEL BOTTLENECK` verdict at lower model capacity, with possibly larger gaps (size=1 should fit the dynamics worse).

---

## Task 2 — Horizon-of-planning ablation under CEM on Acrobot

**Why**: the paper claims the dominant bottleneck is "dynamics quality at planning horizon 15" but does not test the H axis directly. Sweeping `H ∈ {1, 5, 10, 15, 20, 30}` under CEM with the existing checkpoints attributes the bottleneck to *compounding error* (CPG should decrease as H shrinks) vs. *off-manifold distribution mismatch* (CPG would stay flat). It closes the last methodological hole the paper leaves open.

**Branch**: `phase-5o-horizon-cem`
**Effort**: ~12-24 h GPU
**Priority**: HIGH (highest scientific payoff per GPU-hour)
**Status**: queued

**Steps**:

1. Extend `experiments/dmc_acrobot/cem_cpg.py` to accept `--horizon` (default 15). Or write a new `cem_cpg_horizon_sweep.py` that loops over horizons.
2. For each `H ∈ {1, 5, 10, 15, 20, 30}`, run the same setup as the v0.13 pooled-150 cell on the two CEM arms (`mlp_on_tdmpc2_data` and `tdmpc2`). Reuse the existing TD-MPC2 checkpoint at `model_size=5` on Acrobot, do not retrain.
3. Pool to n=150 per cell (3 seeds × 50 episodes).
4. Compute CPG per cell with the existing `wmel.metrics.counterfactual_planning_gap`.

**Expected outputs**:

- `results/dmc_acrobot/horizon_cem_h1_pooled.json`
- `results/dmc_acrobot/horizon_cem_h5_pooled.json`
- (etc., one per H value)
- Or a single aggregated `results/dmc_acrobot/horizon_cem_sweep.json` with a `cells` array indexed by H.

**Paper update**:

- New §5.11 "Horizon-of-planning ablation". One table with rows = H and columns = (oracle, MLP learned, TD-MPC2 learned, CPG, CI, verdict) per CEM arm; or a single grouped plot (CPG vs H, two lines for the two arms).
- Expected punch: if CPG decreases monotonically with smaller H, compounding error is confirmed as the driver. If CPG stays flat, the bottleneck is independent of planning horizon and the diagnosis shifts to off-manifold distribution shift.
- Likely a new figure too (paper currently has Figures 1+2+3).

---

## Task 3 — Cross-env to DMC Reacher-easy

**Why**: third env after Acrobot + Cartpole. Solidifies the "verdict generalizes across DMC tasks" claim. Reacher-easy is the obvious next pick because it shares the underactuated-control family but with a different actuator topology.

**Branch**: `phase-5p-reacher`
**Effort**: ~1-2 days GPU
**Priority**: MEDIUM
**Status**: queued

**Steps**:

1. Write `src/wmel/envs/dmc_reacher.py` mirroring `dmc_cartpole.py`'s structure. Use `dm_control.suite.load('reacher', 'easy')`.
2. Write `experiments/dmc_reacher/{tdmpc2,cem,coverage_mlp_on_tdmpc2,pool}_cpg.py` mirroring `dmc_cartpole/` exactly.
3. Train TD-MPC2 with `model_size=5`, `1_000_000` env steps, seeds 0-2.
4. Run the 4-arm CPG matrix, pool to n=30.

**Expected outputs**: same structure as `results/dmc_cartpole/`, replacing `cartpole` with `reacher`.

**Paper update**:

- Either §5.10 becomes a multi-env table with Cartpole + Reacher columns, or a new §5.12 "Second cross-env: Reacher".
- The cross-env figure (`paper/figures/cross_env_cpg.tex`) becomes a 3-env grouped bar chart.

---

## Task 4 — Cartpole size=5 pooled-150 (CI tightening)

**Why**: Cartpole's CIs are wider than Acrobot's because Cartpole is at n=30 and Acrobot is at n=150. Pooling Cartpole to n=150 makes the cross-env table apples-to-apples and tightens every CI by a factor of ~2.2.

**Branch**: `phase-5q-cartpole-pooled-150`
**Effort**: ~20-30 h GPU
**Priority**: LOW (cosmetic; the verdict already holds at n=30)
**Status**: queued

**Steps**:

1. Run additional seeds (3, 4) at 10 episodes each, or extend each existing seed to 50 episodes, to reach n=150 pooled.
2. Re-pool with `experiments/dmc_cartpole/pool_cpg.py`.

**Expected outputs**:

- `results/dmc_cartpole/{tdmpc2,cem,coverage_mlp_on_tdmpc2}_cpg_size5_pooled150.json`

**Paper update**: replace the §5.10 CI half-widths with the tighter pooled-150 numbers.

---

## Task 5 — Observation-noise perturbation on Acrobot

**Why**: §5.9 closes by flagging that "a genuine fragility test would need an observation-noise perturbation that hurts the learned arms differentially. The DMC wrapper that ships with v0.8 does not expose observation noise; that hook is the natural next axis the perturbation library can carry." This task delivers exactly that and lets §5.9 stop apologising for being structurally one-sided.

**Branch**: `phase-5r-obs-noise-perturbation`
**Effort**: ~6-12 h GPU + adapter work
**Priority**: NICE TO HAVE (closes a flagged limitation)
**Status**: queued

**Steps**:

1. Extend `src/wmel/envs/dmc_acrobot.py` with an observation-noise hook (additive Gaussian on the flat 6-d obs, configurable σ).
2. Write `experiments/dmc_acrobot/obs_noise_perturbation_cpg.py`.
3. Sweep σ ∈ {0, 0.01, 0.05, 0.1} on the CEM × {MLP-on-TDMPC2-data, TD-MPC2} arms, n=50 per cell.

**Expected outputs**: `results/dmc_acrobot/obs_noise_perturbation_cpg.json` with a `cells` array indexed by σ.

**Paper update**:

- §5.9 augmented with the observation-noise row, OR a new §5.12 "Differential perturbation". The point: if observation noise hurts the learned arms more than the oracle (because the learned dynamics is sensitive to obs distribution shift), the gap *widens* under noise; the verdict stays the same.

---

## Picking the next task

If the GPU is free right now, the order is **Task 1 → Task 2 → Task 5 → Task 3 → Task 4**.

- Task 1 finishes a half-shipped story (v0.15) and is cheap.
- Task 2 closes the largest open methodological hole in the paper.
- Task 5 closes a self-flagged limitation cheaply.
- Task 3 extends generality (needs new env adapter, real work).
- Task 4 is pure CI tightening, do last when bandwidth allows.

When you finish one, before starting the next: update its `Status:` to `done`, link the PR, and rebase / pull the latest main.
