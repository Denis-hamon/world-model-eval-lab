# 03 - Benchmark Cards

Each card maps a known research task to the **product question** it actually represents. The intent is to make it possible to skim a scorecard and understand what the numbers mean for an industrial application.

This repository ships the Two-Room and Maze environments in code. Push-T, Reacher, and OGBench Cube cards are included as targets for v0.3+.

---

## Push-T

- **Task type**: 2D rigid-body manipulation; push a T-shaped block to a target pose with a circular pusher.
- **Product interpretation**: closed-loop, contact-rich, low-DoF manipulation under partial observability.
- **Relevant industries**: light assembly, packaging lines, kitting robots, lab automation.
- **World model value hypothesis**: a learned dynamics model can short-cut the cost of physical simulation and enable plan-then-act under tight latency budgets.
- **Candidate metrics**: Action Success Rate, Planning Latency, Compute per Decision, Perturbation Recovery (block nudged mid-rollout).
- **Product question**: *Can a learned world model push a part into spec faster and more reliably than a hand-tuned controller on a 50 ms decision loop?*

---

## Reacher

- **Task type**: 2-link arm reaching a target position in 2D.
- **Product interpretation**: low-DoF kinematic control with a goal in workspace coordinates.
- **Relevant industries**: cobots, lab manipulation, simple pick-and-place, prosthetics research.
- **World model value hypothesis**: a latent dynamics model should generalize across target positions without retraining a controller per goal.
- **Candidate metrics**: Action Success Rate, Average Steps to Success, Planning Horizon, Sample Efficiency.
- **Product question**: *How many demonstrations are needed before a world-model-based planner matches an analytical inverse-kinematics controller on success rate?*

---

## Two-Room

- **Task type**: discrete 2D grid navigation; two rooms separated by a wall with a single doorway.
- **Product interpretation**: minimal example of partially observable planning with a topological bottleneck.
- **Relevant industries**: warehouse routing, indoor robot navigation, building automation, evacuation planning.
- **World model value hypothesis**: a model that learns the doorway as a latent structure should plan through it without explicit graph search.
- **Candidate metrics**: Action Success Rate, Average Steps to Success, Perturbation Recovery, Planning Latency.
- **Product question**: *Can a learned model discover and exploit topological structure (the doorway) without being told it exists?*

This is the environment shipped in `examples/two_room_toy/`.

---

## Maze

- **Task type**: discrete 2D grid navigation through a small maze with walls and dead-ends.
- **Product interpretation**: minimal example where a non-trivial planner is required - naive greedy fails and only a model that simulates candidate futures succeeds.
- **Relevant industries**: warehouse routing under partial maps, indoor robotics, building automation, last-mile delivery.
- **World model value hypothesis**: an action-conditioned predictor combined with random-shooting MPC can solve tasks that defeat reactive heuristics, at the cost of higher per-decision latency.
- **Candidate metrics**: Action Success Rate, Average Steps to Success, Planning Latency, Compute per Decision, Perturbation Recovery.
- **Product question**: *At what planning latency does a world-model-based planner stop being competitive with a reactive heuristic on routing tasks with topological bottlenecks?*

This is the environment shipped in `examples/maze_toy/`. It is the smallest setup where the full `LeWMAdapterStub` contract is exercised end-to-end via the `TabularWorldModelPlanner` subclass.

---

## OGBench Cube

- **Task type**: multi-stage block-stacking from the OGBench suite; pick, transport, place cubes to form a target configuration.
- **Product interpretation**: long-horizon manipulation with composable subgoals.
- **Relevant industries**: assembly automation, logistics palletizing, kitting, construction robotics.
- **World model value hypothesis**: hierarchical world models with subgoal latents should outperform flat planners on tasks that need multi-step reasoning.
- **Candidate metrics**: Action Success Rate, Planning Horizon, Sample Efficiency, Latent Interpretability (do subgoal latents emerge?).
- **Product question**: *Does a world model trained on diverse manipulation transfer to a new stacking goal without retraining, and at what success rate?*

---

## How to add a card

Open a pull request that:

1. States the task type in one sentence.
2. Names the product interpretation in plain English.
3. Lists at least two industries where the question matters.
4. Proposes a hypothesis the benchmark is testing.
5. Lists candidate metrics from `02_metric_taxonomy.md`.
6. Ends with a single, falsifiable product question.
