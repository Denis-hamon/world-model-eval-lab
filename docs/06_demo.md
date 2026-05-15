# 06 - Reading a Scorecard: an applied decision walkthrough

This page is the shortest path from "I have not read the rest of these docs" to "I can defend the thesis of this repo at a whiteboard". It takes a real scorecard, reads it like a non-researcher would, and ends with a concrete decision someone could make from the numbers.

## The applied question

You are evaluating whether to deploy a learned world model in a closed-loop control system. Maybe an industrial pick-and-place arm. Maybe a warehouse routing dispatcher. Maybe a datacenter autoscaler. A research lab hands you a new model and a paper full of reconstruction-loss curves. You need to know one thing:

> Will this model actually work in the loop, at the latency and cost the deployment requires?

Reconstruction loss does not answer that. Frame-level FID does not answer that. The scorecard below does.

## The data

`python -m examples.maze_toy.run_horizon_sweep` runs `TabularWorldModelPlanner` at five planning horizons on a 7x7 maze. About 25 seconds on a laptop. Output:

```
Horizon sweep: tabular-world-model
  plan_h |   success |          95% CI |   steps | latency_ms |       95% CI (ms) | compute/dec
  ---------------------------------------------------------------------------------------------
       5 |     0.000 | [0.00, 0.11]   |     n/a |      0.875 | [0.87, 0.89] |       368.3
      10 |     0.900 | [0.74, 0.97]   |    31.3 |      1.578 | [1.55, 1.61] |       350.6
      15 |     1.000 | [0.89, 1.00]   |    30.5 |      2.348 | [2.34, 2.36] |       278.7
      20 |     1.000 | [0.89, 1.00]   |    33.8 |      3.096 | [3.07, 3.12] |       256.4
      30 |     1.000 | [0.89, 1.00]   |    41.8 |      4.614 | [4.55, 4.68] |       277.5
```

Five rows. Five applied takeaways.

## Reading the table

### `plan_horizon = 5`: success rate is zero

The model never reaches the goal. Per-call latency is 0.88 ms - cheap - but the model is unfit for purpose at this configuration. A pixel-reconstruction benchmark would never have flagged this. Only success rate does, and only because we measured it.

**Applied takeaway**: a model that is cheap to query but cannot solve the task is not deployable. Latency without success is meaningless.

### `plan_horizon = 10`: success rate jumps to 90 percent

This is the kind of single-number win that a research blog post would lead with. The scorecard adds three pieces of context that make it actionable:

- *95 percent CI is [0.74, 0.97]*: the true rate could be as low as 74 percent. Whether you ship this depends on what one failure costs.
- *average steps to success = 31.3*: solved episodes take roughly twice the optimal path length (the maze's shortest path is 14). The model is finding solutions, not optimal solutions.
- *latency = 1.58 ms per call*: well under any reasonable control budget.

**Applied takeaway**: 90 percent is a research win and a procurement problem. Read the CI. Ask "what if it fails this one episode in ten?" before celebrating.

### `plan_horizon = 15`: success rate saturates at 100 percent

At horizon 15 (just past the maze's optimal path length of 14), the model saturates. This is the **effective planning horizon** for this task. Anything below it leaves performance on the table; anything above it spends more without buying success.

**Applied takeaway**: there is a sweet spot, and the scorecard tells you exactly where it is. Below the plateau you lose success; above it you lose money.

### `plan_horizon = 20 and 30`: paying for what does not help

Success stays at 1.0, but:

- Per-call latency grows linearly: 2.35 ms -> 3.10 ms -> 4.61 ms.
- Average steps to success degrade: 30.5 -> 33.8 -> 41.8. The planner commits to longer rollouts and replans later, taking detours along the way.
- Compute per decision stays bounded (~250-370 rollout-units per executed action) because the planner returns proportionally more actions per call. But **total compute per episode grows**.

**Applied takeaway**: a research lab optimising for "horizon 30 gets 100 percent" misses that horizon 15 gets the same success at half the latency and 27 percent fewer executed steps. The applied question is rarely "can it solve the task in unlimited time and compute" - it is "what is the cheapest configuration that solves the task reliably enough".

## The decision

If you were integrating this model into a real product on the basis of this table:

1. **Pick horizon 15.** Latency is 2.35 ms per call. If your control loop runs at 100 Hz (10 ms per cycle), you have 7-8 ms left for the rest of the system.
2. **Budget for ~280 rollout-units per executed action.** If a rollout costs X (FLOPs, dollars, watts, whatever you account in), per-decision cost is 280X. Bring this to procurement.
3. **Tighten the success CI.** 100 percent in 30 episodes gives [0.89, 1.00]. Either run more episodes or accept 89 percent reliability with retry as the lower bound of what you are shipping.
4. **Demand a perturbation profile.** The current run is in a clean environment. Before shipping, request the same scorecard under a perturbation library that captures the failure modes that matter in your domain - blocked paths, delayed actions, sensor noise. The roadmap calls this out as v0.5.

## How this generalises

The same scorecard structure applies to every benchmark card in [03_benchmark_cards.md](03_benchmark_cards.md). The applied questions change - "Can a world model push a part into spec faster than a hand-tuned controller on a 50 ms decision loop?" for Push-T, "Does a stacking model transfer to a new goal without retraining?" for OGBench Cube - but the columns stay the same: success rate, steps, latency, compute, recovery. The thesis is unchanged from [00_thesis.md](00_thesis.md):

> The next bottleneck for world models is not only model quality. It is proof of usefulness.

The scorecard is what proof looks like.

## Reproduce

```bash
git clone https://github.com/Denis-hamon/world-model-eval-lab.git
cd world-model-eval-lab
pip install -e ".[dev]"
python -m examples.maze_toy.run_horizon_sweep
```

About 25 seconds on a laptop. No GPU required.
