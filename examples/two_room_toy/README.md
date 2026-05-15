# Two-Room Toy

A minimal, CPU-only example to demonstrate the evaluation loop end-to-end.

## What this is

A deterministic 2D grid with two rooms separated by a vertical wall and a single doorway. The agent starts in the left room; the goal is in the right room. Actions are `up`, `down`, `left`, `right`.

The point of this environment is **not** to be a hard benchmark. It is the smallest setup that:

- exercises topological planning (the doorway),
- demonstrates the `BenchmarkEnvironment` and `PlannerPolicy` contracts,
- distinguishes a random policy from a structured one,
- responds to a small perturbation in a way that produces a non-trivial recovery rate.

## Run it

From the repository root:

```bash
pip install -e ".[dev]"
python -m examples.two_room_toy.run_baseline
```

This will:

1. Run 50 episodes with a random policy.
2. Run 50 episodes with a greedy policy that knows the doorway location.
3. Print a scorecard for each.
4. Write a combined JSON report to `examples/two_room_toy/sample_report.json`.

Episodes have a perturbation probability of 0.3 by default. When triggered, the perturbation snaps the agent one step opposite to its last movement.

## What to read in the scorecard

- **Action success rate** - if this is near zero for the greedy policy, the doorway hint is broken.
- **Average steps to success** - rough proxy for path optimality; greedy should be much lower than random.
- **Average planning latency (ms)** - effectively free for both baselines, but the column is wired up for real models.
- **Perturbation recovery rate** - the fraction of perturbed episodes the policy still completed.

## Extending

To plug in your own policy, implement `wmel.adapters.base.PlannerPolicy` and pass an instance to `BenchmarkRunner`. To plug in your own environment, implement `wmel.adapters.base.BenchmarkEnvironment`.
