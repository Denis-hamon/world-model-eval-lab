# Re-run guide: task-distribution sampling (`--varied-init`)

## Why this re-run exists

Every CPG worked example in the repo was, until now, evaluated at a **single
fixed initial state**. The env adapters load with `task_kwargs={"random": 0}`
and the `BenchmarkRunner` builds a fresh env per episode, so every episode --
across all "seeds" -- started from the *same* state (and, for Reacher, the
*same* target). The "seeds" varied only the planner RNG, not the environment.

Consequences, verified empirically (`MUJOCO_GL=disable`, three fresh resets per
env return identical observations):

- Reported success rates estimated `P(success | one fixed start, planner noise)`,
  not the task's initial-state distribution. "N episodes per arm" overstated
  how much of the task was sampled.
- A quick CPU check on the Acrobot flagship showed the headline is
  config-sensitive: at the fixed start, oracle `0.30` / learned `0.00` /
  CPG `+0.30`; over a 10-instance task sample, oracle `0.10` / learned `0.10` /
  CPG `0.00` (`INCONCLUSIVE`). n=10 is noisy, but the direction is clear --
  **the fixed `random=0` start is easier-than-average for the oracle's
  random-shooting MPC.** The pooled re-run below is needed to get the real
  task-level CPG.

## What `--varied-init` does

The `--varied-init` flag (added to every CPG driver, **off by default** so the
committed results still reproduce) routes env construction through
`experiments/_seeding.py`:

- **Eval** uses `eval_varied_factory(EnvClass, seed)`: episode *k* draws task
  seed `seed * 100_000 + k`. Both/all arms of one comparison build their own
  factory from the **same** `seed`, so episode *k* starts from the **same**
  state in every arm -- a genuinely *paired* design that samples the task.
- **Training-data collection** uses `train_varied_factory(EnvClass, seed)`,
  which draws from a **disjoint** block of initial states (base seed
  `10_000 + seed`), so the model is never trained on the start states it is
  scored on.

Pooling distinct seeds `{0, 1, 2}` then yields disjoint sets of task instances.

## Re-run commands

Run **every** arm and env with `--varied-init` so the pooled dataset is
internally consistent (each output JSON records `"varied_init": true`). The
re-run **overwrites** the committed single-config JSONs; that is intended (the
paper moves to task-level numbers) and git preserves the old files.

### Acrobot (flagship + robustness)

CPU is enough for the oracle and MLP arms; the TD-MPC2 / CEM arms need the
existing Acrobot TD-MPC2 checkpoint (GPU for inference, no retraining).

```bash
# Flagship random-shooting, n=10 (CPU):
MUJOCO_GL=disable python -m experiments.dmc_acrobot.cpg --varied-init

# Multi-seed CEM sweep -> pooled n=150 (reuses the Acrobot TD-MPC2 ckpt; GPU):
python -m experiments.dmc_acrobot.cem_cpg_sweep --varied-init --device cuda

# Robustness arms (reuse ckpt; GPU for the TD-MPC2 arm):
python -m experiments.dmc_acrobot.tdmpc2_cpg            --varied-init --device cuda
python -m experiments.dmc_acrobot.coverage_mlp_on_tdmpc2 --varied-init --device cuda
python -m experiments.dmc_acrobot.perturbation_cpg      --varied-init --device cuda
python -m experiments.dmc_acrobot.cem_cpg               --varied-init --device cuda
```

### Cartpole (cross-env, two capacities)

For each `MODEL_SIZE in {1, 5}` and `SEED in {0, 1, 2}` (reuses the existing
`tdmpc2_agent*.pt` checkpoints -- `tdmpc2_cpg.py` resumes and **skips training**
when the checkpoint is already at the target step count, so the GPU is used for
evaluation inference only):

```bash
for M in 1 5; do for S in 0 1 2; do
  python -m experiments.dmc_cartpole.tdmpc2_cpg             --varied-init --seed $S --model-size $M --device cuda
  python -m experiments.dmc_cartpole.cem_cpg               --varied-init --seed $S --model-size $M --device cuda
  python -m experiments.dmc_cartpole.coverage_mlp_on_tdmpc2 --varied-init --seed $S --model-size $M --device cuda
done; done
# Pool each capacity:
python -m experiments.dmc_cartpole.pool_cpg --model-size 1 --seeds 0 1 2
python -m experiments.dmc_cartpole.pool_cpg --model-size 5 --seeds 0 1 2
```

#### Clean Cartpole init-only ablation (matched checkpoint)

The varied-init Cartpole numbers above (committed at the default filenames) were
produced from checkpoints that had to be **retrained from scratch** (the
originals were cleaned off disk). Comparing them to the v0.17 fixed-init numbers
therefore confounds the initial-state change with a checkpoint change. For a
clean, non-confounded init-only ablation, evaluate the **same** on-disk
checkpoints at **fixed** init and write to a distinct suffix (so the committed
varied results are not overwritten), using the `--out-suffix` flag:

```bash
for M in 1 5; do for S in 0 1 2; do
  python -m experiments.dmc_cartpole.tdmpc2_cpg             --seed $S --model-size $M --device cuda --out-suffix _fixedinit
  python -m experiments.dmc_cartpole.cem_cpg               --seed $S --model-size $M --device cuda --out-suffix _fixedinit
  python -m experiments.dmc_cartpole.coverage_mlp_on_tdmpc2 --seed $S --model-size $M --device cuda --out-suffix _fixedinit
done; done
python -m experiments.dmc_cartpole.pool_cpg --model-size 1 --seeds 0 1 2 --out-suffix _fixedinit
python -m experiments.dmc_cartpole.pool_cpg --model-size 5 --seeds 0 1 2 --out-suffix _fixedinit
```

This yields `*_fixedinit*.json` (fixed init) alongside the default-name varied
files, both from the same checkpoints -- a clean fixed-vs-varied pair for the
v0.18 Cartpole section. `--out-suffix` affects only the output result JSON, not
the `.pt` checkpoint paths.

### Reacher (third env, `model_size = 1`)

```bash
for S in 0 1 2; do
  python -m experiments.dmc_reacher.tdmpc2_cpg             --varied-init --seed $S --model-size 1 --device cuda
  python -m experiments.dmc_reacher.cem_cpg               --varied-init --seed $S --model-size 1 --device cuda
  python -m experiments.dmc_reacher.coverage_mlp_on_tdmpc2 --varied-init --seed $S --model-size 1 --device cuda
done
python -m experiments.dmc_reacher.pool_cpg --seeds 0 1 2
```

## Scope boundaries (intentionally NOT varied here)

- **TD-MPC2's own RL training env** (`tdmpc2_make_env` / `_train_tdmpc2`) is
  untouched. TD-MPC2 already resets across initial states during RL training;
  the bug was only that it was *evaluated* at a fixed start. Checkpoints are
  reused as-is.
- **The TD-MPC2 latent decoder's training rollouts** (inside `tdmpc2_cpg.py`)
  are left as-is; varying them is a second-order refinement.
- **`cem_cpg_horizon_sweep.py` and `obs_noise_perturbation_cpg.py`** are not
  patched: they are exploratory and do not back any main paper table. Patch
  them with the same one-line pattern if a future table needs them.

## After the re-run

The pooled JSONs (each tagged `"varied_init": true`) become the basis for the
v0.18 paper revision. Expect the headline to change: the fixed-start oracle
advantage will shrink, and some `MODEL BOTTLENECK` verdicts may soften toward
`INCONCLUSIVE` at the current sample sizes -- which the power analysis already
tells us how to size. Re-pool to n=150 where a verdict lands near the gate.
