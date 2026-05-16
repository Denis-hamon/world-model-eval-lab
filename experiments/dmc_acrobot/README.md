# experiments/dmc_acrobot/

The first non-toy environment in this repository: DeepMind Control Suite **Acrobot-swingup**. A two-link underactuated pendulum that must build energy through swinging and then balance the tip upright.

## Why this env

Acrobot-swingup is the smallest serious continuous-control task in the DMC suite, well within reach on CPU. It checks the framework on properties the maze toy does not exercise:

- **Continuous dynamics** (rigid-body physics, not a discrete grid).
- **Stochastic horizon** (whether you swing up at all depends on the energy trajectory).
- **Sparse-ish success** (the upright pose is a small fraction of the state space).
- **Comparable to literature** (Hafner et al. 2023 Dreamer-V3, Hansen et al. 2024 TD-MPC2, both publish numbers on this task).

## Setup

```bash
pip install -e ".[dev,control]"
```

The `control` extra pulls `dm-control` and `mujoco` (CPU physics, no rendering). On a headless box, `dm-control` emits a benign `DISPLAY` warning; physics runs anyway.

## Run

```bash
python -m experiments.dmc_acrobot.baseline
```

Writes a `Scorecard` JSON to [`results/dmc_acrobot/baseline_random.json`](../../results/dmc_acrobot/). The current scorecard uses a uniformly random policy over the discretised 5-level torque space.

## Status

- v0.8 (current): random-policy baseline only. Success rate is essentially zero - which is exactly the floor we want to pin before any model is plugged in.
- v0.9 planned: a GRU world model trained on random rollouts, plugged into `TabularWorldModelPlanner` via the existing `dynamics` callable.
- v1.0 planned: the **Counterfactual Planning Gap** metric on this env, comparing the GRU's planning impact against the oracle dynamics.

See [docs/05_30_day_prototype_plan.md](../../docs/05_30_day_prototype_plan.md) for the broader roadmap.
