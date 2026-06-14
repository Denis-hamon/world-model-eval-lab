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
