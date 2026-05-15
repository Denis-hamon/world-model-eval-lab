---
prev:
  title: "03 - Benchmark cards"
  url: 03_benchmark_cards.html
next:
  title: "05 - 30-day study plan"
  url: 05_30_day_prototype_plan.html
---
# 04 - Industrial Use Cases

This page connects the metric taxonomy and the benchmark cards to concrete industrial contexts where action-conditioned world models could plausibly add value. The aim is to give a non-researcher a place to point when they ask "why should I care?".

For each domain we list:

- the **problem** in operational terms,
- **why a world model could matter**,
- **what should be measured**,
- and **why LLM/VLM-style evaluation is insufficient**.

---

## Robotics and manipulation

- **Problem**: closed-loop, contact-rich tasks (pushing, inserting, stacking, threading) are hard to script and brittle to perturbation. Per-task controllers do not amortize across SKUs.
- **Why a world model could matter**: a learned latent dynamics model can simulate candidate action sequences orders of magnitude faster than a rigid-body simulator, enabling plan-then-act under tight latency budgets.
- **What should be measured**: Action Success Rate per task, Planning Latency at the control rate, Compute per Decision, Perturbation Recovery when the object is nudged mid-execution.
- **Why current LLM/VLM evaluation is insufficient**: a VLM can describe a scene; it cannot tell you whether a 30 ms plan will succeed at sub-millimetre tolerance. Caption quality and visual question answering accuracy do not predict closed-loop control performance.

---

## Industrial automation

- **Problem**: high-mix, low-volume manufacturing requires rapid reconfiguration of process control logic. Hand-coded PLC programs do not generalize across product variants.
- **Why a world model could matter**: an action-conditioned predictor over process telemetry can support model-predictive control without re-engineering the controller for each variant.
- **What should be measured**: Planning Horizon at which control performance plateaus, Sample Efficiency on new product variants, Surprise Detection on process anomalies.
- **Why current LLM/VLM evaluation is insufficient**: industrial process data is mostly low-dimensional, high-frequency, and physically constrained. Text-style benchmarks measure none of these properties.

---

## Cloud and datacenter operations

- **Problem**: scheduling, autoscaling, power management, and cooling control are decision problems with strong dynamics, expensive mistakes, and partial observability.
- **Why a world model could matter**: a learned model of workload and infrastructure response lets a planner explore counterfactual policies without touching production - turning autoscaling into a planning problem rather than a reactive heuristic.
- **What should be measured**: Action Success Rate against SLOs, Compute per Decision (the controller cannot dominate the workload it controls), Perturbation Recovery under traffic spikes, Latent Interpretability for safety review.
- **Why current LLM/VLM evaluation is insufficient**: an LLM can summarize an incident; it cannot tell you what would have happened under a different autoscaling policy. The questions that matter are counterfactual, and require an action-conditioned simulator.

---

## Logistics and warehouse routing

- **Problem**: routing pickers, AMRs, or AGVs through a warehouse under changing demand and partial occupancy maps. Classical solvers struggle when the map and demand are stochastic.
- **Why a world model could matter**: a latent model of warehouse dynamics enables short-horizon planning that adapts to congestion without a full graph re-solve every tick.
- **What should be measured**: Average Steps to Success, Perturbation Recovery (blocked aisles, dropped tasks), Planning Latency at the dispatch frequency, Sample Efficiency on new floor plans.
- **Why current LLM/VLM evaluation is insufficient**: a VLM can read a floor plan; it cannot predict the second-order effect of routing 30 vehicles through one aisle. Spatial reasoning benchmarks rarely capture dynamic congestion.

---

## Safety monitoring and anomaly detection

- **Problem**: detecting when a physical or software system is leaving its operating envelope, ideally before a failure cascades.
- **Why a world model could matter**: a predictor that has internalized "normal" dynamics can flag when reality diverges from prediction - a structural form of anomaly detection that does not require labelled failure data.
- **What should be measured**: Surprise Detection AUROC on held-out anomalies, false-alarm rate at a fixed operating point, Latent Interpretability for incident review.
- **Why current LLM/VLM evaluation is insufficient**: anomaly detection is a tails problem. Static QA accuracy says nothing about how a model behaves on inputs from outside the training distribution, which is the only regime where this use case matters.

---

## Common thread

Across all five domains, the failure mode of "evaluate the model on static benchmarks and hope for the best" is the same: the metric does not measure the property that determines whether the product ships. The point of this repository is to push that question to the front of the conversation.