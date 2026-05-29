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

## Task 1 — Cartpole size=1 seed 0 + pooled n=30 [DONE before v0.15]

**Status**: **done**. The size=1 pooled-30 cells landed on `main` between v0.14.1 and v0.15. Result JSONs:

- `results/dmc_cartpole/tdmpc2_cpg_pooled.json`
- `results/dmc_cartpole/cem_cpg_pooled.json`
- `results/dmc_cartpole/coverage_mlp_on_tdmpc2_cpg_pooled.json`

The pooled-30 size=1 cells are integrated into paper §5.10 (Table~\ref{tab:crossenv_size1}) as part of the v0.15 release. **Headline finding**: the CEM~$\times$~TD-MPC2 cell flips to `INCONCLUSIVE` -- learned arm reaches $0.533$ vs an oracle at $0.500$, raw CPG $-0.033$, AC CI $[-0.28, +0.21]$. First moderate-$n$ `INCONCLUSIVE` verdict in the paper; smaller-capacity TD-MPC2 closes the gap on Cartpole's CEM oracle that the larger-capacity model does not. Do not re-run.

---

## Task 2 — Horizon-of-planning ablation under CEM on Acrobot

**Why**: the paper claims the dominant bottleneck is "dynamics quality at planning horizon 15" but does not test the H axis directly. Sweeping `H ∈ {1, 5, 10, 15, 20, 30}` under CEM with the existing checkpoints attributes the bottleneck to *compounding error* (CPG should decrease as H shrinks) vs. *off-manifold distribution mismatch* (CPG would stay flat). It closes the last methodological hole the paper leaves open.

**Branch**: `phase-5o-horizon-cem`
**Effort**: ~12-24 h GPU
**Priority**: HIGH (highest scientific payoff per GPU-hour)
**Status**: done (PR pending)

**Steps**:

1. Extend `experiments/dmc_acrobot/cem_cpg.py` to accept `--horizon` (default 15). Or write a new `cem_cpg_horizon_sweep.py` that loops over horizons.
2. For each `H ∈ {1, 5, 10, 15, 20, 30}`, run the same setup as the v0.13 pooled-150 cell on the two CEM arms (`mlp_on_tdmpc2_data` and `tdmpc2`). Reuse the existing TD-MPC2 checkpoint at `model_size=1` on Acrobot (the v0.12 / v0.13 checkpoint, see `experiments/dmc_acrobot/tdmpc2_cpg.py:137`), do not retrain.
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

## Task 3 — Cross-env to DMC Reacher-easy [ADAPTER SHIPPED; TRAINING IS THE GPU WORK]

**Why**: third env after Acrobot + Cartpole. This is the highest-value generality win for the paper: it tests whether the INCONCLUSIVE-to-MODEL-BOTTLENECK transition (the power-analysis thesis) reproduces on a structurally different task -- a 2-DOF reaching arm, not an underactuated swing-up, and the first 2-D-action env in the repo.

**Branch**: `phase-5t-reacher-training`
**Effort**: ~3-4 wall-days on 2x L40S (6 TD-MPC2 trainings dominate)
**Priority**: HIGH (the multi-env claim)
**Status**: queued

**ALREADY DONE on main (do NOT redo):**

- `src/wmel/envs/dmc_reacher.py` is shipped: `DMCReacherEnv` (6-D obs, 9-action discrete grid = 3 levels x 2 joints), `make_reacher_oracle_dynamics()` (oracle VERIFIED to reproduce `env.step` to 5.6e-17 over a 50-step rollout -- exact reconstruction, no atan2 loss; target recovered from `to_target`), and `reacher_reach_score` (finger-to-target distance; lower is better, the exact quantity the DMC reward thresholds, same minimizing convention as the swing-up scores). Regression test in `tests/test_dmc_reacher.py`.
- The oracle and env adapter need NO further work. Trust the regression test.

**Remaining GPU work (this task):**

1. Train TD-MPC2 on `reacher`/`easy`, `model_size=1`, `1_000_000` env steps, seeds {0,1,2} = 3 runs. (Drop `model_size=5` unless time permits -- size=1 is the atlas-relevant regime, per the plan's cut order. Run 2 seeds in parallel across the 2 L40S.)
2. Write `experiments/dmc_reacher/{tdmpc2_cpg,cem_cpg,pool_cpg}.py` by copying the `experiments/dmc_cartpole/` scripts and changing the env import to `dmc_reacher` + the score to `reacher_reach_score`. NOTE the action is 2-D: the random-shooting / CEM planners must sample from the 9-action grid (`DMCReacherEnv().action_space`), not a 5-level 1-D set. Verify the planner handles 2-tuples (it should -- it treats actions as opaque hashables).
3. Run the standard 4-arm matrix (random-shoot + CEM) x {TD-MPC2 dynamics, MLP-on-TD-MPC2-data} x seeds {0,1,2}, 10 ep/arm/seed, pool to n=30.
4. Smoke-test first: `... --smoke` (1 seed, few episodes, few hundred train steps) must pass end-to-end before the full run. Long runs in `tmux`/`nohup`.

**Expected outputs**: `results/dmc_reacher/*` mirroring `results/dmc_cartpole/`.

**Paper update** (after results land):

- New subsection (next free, likely §5.12) "Third environment: Reacher". Report HONESTLY whether the INCONCLUSIVE ridge / MODEL BOTTLENECK plateau reproduces. Yes = the verdict generalizes across DMC tasks; No = the pattern was env-specific, still a finding.
- Extend `paper/figures/cross_env_cpg.tex` to a 3-env grouped bar chart.
- Do NOT introduce swm baselines (DINO-WM/LeWorldModel/PLDM) or pixel/FoV axes -- non-affiliation guardrail.

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
**Status**: done (PR pending)

**Steps**:

1. Extend `src/wmel/envs/dmc_acrobot.py` with an observation-noise hook (additive Gaussian on the flat 6-d obs, configurable σ).
2. Write `experiments/dmc_acrobot/obs_noise_perturbation_cpg.py`.
3. Sweep σ ∈ {0, 0.01, 0.05, 0.1} on the CEM × {MLP-on-TDMPC2-data, TD-MPC2} arms, n=50 per cell.

**Expected outputs**: `results/dmc_acrobot/obs_noise_perturbation_cpg.json` with a `cells` array indexed by σ.

**Paper update**:

- §5.9 augmented with the observation-noise row, OR a new §5.12 "Differential perturbation". The point: if observation noise hurts the learned arms more than the oracle (because the learned dynamics is sensitive to obs distribution shift), the gap *widens* under noise; the verdict stays the same.

---

## Picking the next task

If the GPU is free right now, the order is **Task 3 (Reacher training) → Task 4** (Tasks 1, 2, 5 are done; Task 3's adapter+oracle are shipped and only the training remains). Task 3 is now top priority: it is the multi-env generality claim and the power-analysis thesis's cross-env test, and the hard part (the verified oracle) is already on main.

- Task 2 closes the largest open methodological hole in the paper.
- Task 5 closes a self-flagged limitation cheaply.
- Task 3 extends generality (needs new env adapter, real work).
- Task 4 is pure CI tightening, do last when bandwidth allows.

When you finish one, before starting the next: update its `Status:` to `done`, link the PR, and rebase / pull the latest main.
