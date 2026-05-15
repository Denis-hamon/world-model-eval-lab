# 01 - Product Wedge

## Research breakthrough

Action-conditioned world models - JEPA-style latent predictors, video world models, latent dynamics models - are crossing a threshold. They can now imagine plausible futures conditioned on actions in non-trivial domains: manipulation, driving, navigation, and increasingly open-ended scenes. The research narrative is moving from "can we predict pixels" to "can we plan with the predictions".

## Product gap

Despite this, almost no published world-model result answers the questions a product organization needs answered:

- What is the **success rate** of decisions made with this model on my task?
- What is the **latency budget** required to use it in a closed loop?
- What is the **compute cost** per decision, and how does it scale with planning horizon?
- How does the model behave under **perturbation** - the kind of small surprise a real environment delivers every few seconds?
- How many demonstrations or interactions does the model need before it becomes useful on a new task?

The result is a gap. Researchers publish predictors. Product teams cannot tell which predictor, if any, is ready to ship.

## Missing layer

What is missing is not another model. It is the **evaluation layer** that sits between the model and the product:

- A standard contract any model can implement (encode, rollout, score, plan).
- A standard set of product-grade metrics that answer the questions above.
- A standard set of benchmark cards that translate academic tasks into the product questions they actually represent.
- A standard reporting format - a scorecard - that a non-researcher can read.

This is similar to what happened in classical ML around model cards and datasheets, except oriented toward decision systems instead of static models.

## First wedge

The first wedge this repository proposes is small on purpose:

- A tiny CPU-only environment (two-room grid).
- A minimal adapter interface (`PlannerPolicy`, `BenchmarkEnvironment`).
- Two trivial baselines (random and greedy) that establish the floor and a sane reference.
- A scorecard with five initial metrics.
- A JSON report that any downstream tool can consume.

That is enough to show the shape of the layer without pretending to be a complete benchmark suite.

> Before world models get a killer app, they need a killer evaluation layer.

## Why this matters

If even one research lab adopts a product-grade evaluation layer alongside their pre-print, the conversation about world models changes. Instead of arguing about reconstruction loss, the field can argue about **success rate at a 50 ms decision budget under 10 percent perturbation** - which is a conversation a product person can join.

That is the wedge. The repository is a small, opinionated bet that the wedge is worth driving.
