# Maze Toy

A small maze that exercises planning horizon and demonstrates the full
`LeWMAdapterStub` contract end-to-end.

## What this is

A 7x7 grid maze with walls. The agent starts top-left, the goal is bottom-right, and the direct Manhattan path is blocked. The optimal path is 14 actions long. The maze is fully connected (all 16 non-wall cells are reachable from the start).

![maze](../../docs/assets/maze.svg)

This environment is intentionally hard enough that:

- **Random** rarely succeeds within the horizon (random walks don't usually reach the goal).
- **Greedy without a waypoint** gets stuck on walls - the planner's internal model diverges from the env after the first blocked move.
- **TabularWorldModelPlanner** succeeds: it samples candidate action sequences, simulates them with its internal dynamics, and picks the one whose terminal state is closest to the goal.

The point is not the maze. The point is that the same `BenchmarkRunner` evaluates all three policies through the same `PlannerPolicy` and `BenchmarkEnvironment` interfaces, and that the world-model contract from `LeWMAdapterStub` is fully implementable without any third-party dependency.

## Run it

From the repository root:

```bash
python -m examples.maze_toy.run_baseline
```

This will:

1. Run 30 episodes with a random policy.
2. Run 30 episodes with a naive greedy policy (no waypoint hint).
3. Run 30 episodes with a `TabularWorldModelPlanner` whose internal dynamics is the maze's transition function.
4. Print a scorecard for each.
5. Write a combined JSON report to `examples/maze_toy/sample_report.json`.

Each episode has a perturbation probability of 0.2.

## What to read in the scorecard

- **Action success rate** - random should be near 0, greedy should be 0 (gets stuck), tabular-world-model should be high.
- **Average steps to success** - tabular-world-model should land near 14-25 (optimal path is 14; replanning costs some steps).
- **Average planning latency (ms)** - random/greedy are effectively free; the world-model planner is more expensive because it simulates 200 candidate sequences per `plan()` call. This is the trade-off any real world model will exhibit at scale.
- **Perturbation recovery rate** - the world-model planner should recover from most perturbations because it replans after every executed batch of actions.

## Horizon sweep

To see how performance scales with lookahead depth:

```bash
python -m examples.maze_toy.run_horizon_sweep
```

This runs `TabularWorldModelPlanner` at horizons `(5, 10, 15, 20, 30)` with 30 episodes per point and prints a success-rate / latency curve with 95 percent confidence intervals. A JSON report is written to `examples/maze_toy/horizon_sweep_report.json`. See `docs/02_metric_taxonomy.md` for the worked example.

## Extending

`TabularWorldModelPlanner` is a subclass of `LeWMAdapterStub` that fills in `encode`, `rollout`, `score`, and `plan` with toy implementations. To wire in a real model:

1. Subclass `LeWMAdapterStub` (or `TabularWorldModelPlanner` if the search logic suits you).
2. Replace `encode` with your real encoder.
3. Replace `rollout` with your real action-conditioned predictor.
4. Replace `score` with a learned or fixed latent-space distance.
5. Run the existing `BenchmarkRunner` against any `BenchmarkEnvironment`.

Nothing else in the framework needs to change.
