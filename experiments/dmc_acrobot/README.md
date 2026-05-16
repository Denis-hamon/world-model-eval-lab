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

- **v0.8 (current): two baselines committed.**
  - `baseline.py` -> random policy. `results/dmc_acrobot/baseline_random.json`. Success rate 0%.
  - `learned_baseline.py` -> Markovian MLP world model trained on 10 x 200 = 2 000 random transitions (200 epochs), plugged into `TabularWorldModelPlanner` via the existing `dynamics=` argument with a task-specific `acrobot_upright_score`. `results/dmc_acrobot/learned_random_rollouts.json`. Validation MSE on held-out transitions ~0.026; success rate is still 0%.
- **What the two scorecards together actually show.** Prediction quality (val_mse) is good, decision quality (success_rate) is bad. The two are decoupled by construction. Three candidate explanations remain open:
  1. **Distribution shift**: the random rollouts do not cover the high-energy region where swing-up actually happens; the model extrapolates badly there.
  2. **Planner capacity**: random-shooting MPC over 5 discretised torques x 15 horizon = 5^15 candidates is a needle-in-haystack search for the energy-pumping policy, even with a perfect model.
  3. **Score approximation**: `acrobot_upright_score` uses `-(cos(upper) + cos(lower))` rather than the exact DMC reward; the gradient direction is right but the magnitude may not be aligned.
  The right way to decompose these is the Counterfactual Planning Gap metric, scheduled for v0.9.

- **v0.9 planned**: the **Counterfactual Planning Gap** metric. Same env, same planner, two dynamics callables (oracle vs learned MLP). CPG = success_rate(oracle) - success_rate(learned), reported across seeds. Decomposes the failure cleanly: a near-zero CPG with both at 0% says "planner is the bottleneck", a positive CPG says "model is the bottleneck".

- **v1.0 planned**: extend to multiple seeds with rliable-style confidence intervals, and add a second env (Cartpole-swingup or DMC Reacher) to test cross-task generalisation of any improvements.

See [docs/05_30_day_prototype_plan.md](../../docs/05_30_day_prototype_plan.md) for the broader roadmap.
