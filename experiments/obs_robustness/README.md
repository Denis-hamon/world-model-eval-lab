# Observation-richness robustness of the CPG verdict (reframed T2.3)

Does the **prediction ≠ decision** dissociation — a model can have large
observation-space prediction error yet a small Counterfactual Planning Gap, and
vice versa — survive when the world model no longer sees the clean, privileged
low-dimensional state? Every CPG cell shipped so far uses the minimal DMC state
(joint angles/velocities). The threat to validity: maybe the dissociation is an
artifact of that tidy observation.

## Why not pixels

The roadmap's original T2.3 was "one DMC-from-pixels task." A pixel /
field-of-view evaluation axis is the signature of the DINO-WM /
stable-worldmodel agenda, and the lab keeps a deliberate **non-affiliation
guardrail** against introducing that axis (`experiments/GPU_ROADMAP.md:92`). So
T2.3 is reframed to deliver the same scientific payload — *does the verdict hold
beyond the clean state?* — with **nuisance-augmented state** instead of pixels.
No pixel axis, no JEPA-adjacent baselines, same question answered.

## Construction

`observation_augmentation.py` (pure, stdlib-only) wraps any
`BenchmarkEnvironment` so the observation becomes `true_state ++ nuisance`:

- the **simulator stays the oracle** — the augmented oracle steps the true
  physics on the state slice and reproduces the nuisance exactly, so it
  reproduces the augmented env step to the same precision the base oracle
  reproduces the base env;
- the **planner scores only the true-state slice**, so decisions depend on real
  physics alone, never on the nuisance;
- the **learned MLP trains on the augmented observation**, so its one-step error
  (the naive M1 foil from the keystone) is moved up or down by the nuisance
  design while the closed-loop gap should not move.

Two nuisance kinds, both deterministic **functions of the state** (hence exactly
reproducible by the oracle, which is handed only the augmented obs), differing
only in how learnable they are in a single step:

| kind | nuisance | expected effect on one-step MSE |
|------|----------|---------------------------------|
| `redundant` | smooth low-frequency features `tanh(state)` | **deflates** (easy padding dims a good model nails) |
| `high_freq` | high-frequency features `sin(K·state)`, K large | **inflates** (a finite smooth MLP cannot resolve the frequency) |

This is a genuine **one-step** difficulty — the quantity the keystone's M1 foil
measures. (A chaotic *temporal* map would not do: its one-step update is a
trivially fittable parabola, and its sensitivity only inflates *multi-step*
rollout error, which this metric does not compute. `sin(K·state)` is hard to
predict from the state in a single step, which is the point.)

`width=0` is the no-nuisance control and reproduces the baseline Cartpole CPG.

## Falsifiable claim

Across nuisance kind and width, the hypothesis is that **CPG and `mse_state`
stay ≈ flat** while **`mse_total` moves a lot** (down for `redundant`, up for
`high_freq`). That would show a popular "model quality" proxy —
observation-space one-step MSE — is movable by task-irrelevant observation
design **without touching the closed-loop verdict**, sharpening prediction ≠
decision. If instead CPG shifts materially with the nuisance, the verdict is
observation-form-dependent — itself a reportable finding. The direction of each
effect is **measured per cell, not assumed**: the `mse_total / mse_state /
mse_nuisance` split is reported for every cell so the mechanism is visible
rather than asserted.

Honest confound: the nuisance dims are also model *inputs*, so a finite MLP can
be mildly distracted by them, nudging `mse_state` (and thus CPG) a little. The
analysis reports `mse_state` separately precisely so that input-distraction
effect is visible rather than hidden inside an aggregate.

## Result (full CPU sweep, 21 cells: {redundant, high_freq} x {0,4,16,64} x 3 seeds)

The decoupling holds, and strongly. A popular "model quality" proxy --
observation-space one-step MSE -- swings by orders of magnitude purely from
task-irrelevant observation design, while the downstream gap does not track it:

- `mse_total` ranges **0.0000 to 0.283** (the `high_freq` arm climbs monotonically
  with width: 0.037 -> 0.117 -> 0.278 at w = 4/16/64; `redundant` stays ~0).
- the CPG `gap` stays at **mean +0.79** (range +0.10..+1.00) with no monotone
  relation to width or kind. **Spearman(mse_total, gap) = +0.09, 95% CI
  [-0.37, +0.51]**; Spearman(mse_state, gap) = +0.03 -- both within noise.
- `mse_state` -- the decision-relevant slice -- stays small in every cell, **~1e-5
  to ~3e-3**, trending mildly upward with `high_freq` width (the input-distraction
  confound disclosed above), i.e. orders of magnitude below `mse_total`.

Three honest caveats:

1. **The learned MLP is at MODEL BOTTLENECK (fails to plan) in nearly every
   cell** (learned planning success is 0 in 14 of 21 cells, mean 0.11; oracle is
   0.8-1.0), so `gap` ~ oracle success regardless of observation. The decoupling
   (one-step MSE moves ~5e4x, the gap does not) is the point, but the outcome
   held constant is *near-failure*, not a graded success rate -- the same DMC-MLP
   planning floor that pushed the keystone PoC to the maze toy.
2. **The low Spearman is consistent-with-decoupling, not positive proof of it.**
   Because `gap` is near-constant and oracle-pinned, it has little variance to
   correlate with, which mechanically suppresses any correlation; the CI is
   correspondingly wide ([-0.37, +0.51]). Read this as "the MSE swing buys no
   detectable change in the gap," not as an estimated zero relationship.
3. **The deflation direction did not materialise.** The MLP already fits the
   clean Cartpole state to ~0 one-step MSE, so the smooth `redundant` features
   (also fit to ~0) leave `mse_total` at the floor -- there is nothing to deflate.
   Only the inflation direction (`high_freq`) shows a large, monotone effect.

So the genuinely new content here is not prediction != decision itself (the
keystone already establishes that) but its **invariance**: a ~5e4x swing in
observation-space one-step MSE, engineered purely from task-irrelevant
observation design, buys no detectable change in the closed-loop gap, while the
decision-relevant `mse_state` stays orders of magnitude smaller throughout. The
committed JSON has every per-cell row.

## Run (CPU)

```bash
pip install -e ".[control,learned,dev]"
python -m experiments.obs_robustness.cartpole_obs_robustness --smoke   # wiring check
python -m experiments.obs_robustness.cartpole_obs_robustness           # full sweep
```

Writes `results/obs_robustness/cartpole_obs_robustness.json` (one row per
`kind × width × seed`, with `gap`, the AC CI, `verdict`, and the
`mse_total / mse_state / mse_nuisance` split; non-metric numbers live under
`context`). CPU only — no GPU, no checkpoints. The pure augmentation algebra is
unit-tested in `tests/test_observation_augmentation.py` with a synthetic env.
