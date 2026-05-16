---
prev:
  title: "03 - Benchmark cards"
  url: 03_benchmark_cards.html
next:
  title: "05 - 30-day study plan"
  url: 05_30_day_prototype_plan.html
---
# 04 - Industrial Use Cases

> **Status: speculative.** None of the nine cases below are implemented as benchmarks in this repository. The page is a market landscape - a forward-looking note on where a decision-grade evaluation layer would matter most - not a results report. The actual benchmarks shipped today are limited to the toy maze and the two-room env. See [the demo walkthrough](06_demo.html) for what is currently measurable, and [the 30-day study plan](05_30_day_prototype_plan.html) for what is in scope next.

This page connects the metric taxonomy and the benchmark cards to concrete industrial contexts where action-conditioned world models could plausibly add real value. The aim is to give a non-researcher a place to point when they ask "why should I care?" - and to make explicit which of these are speculative versus where billions of dollars of capex are already chasing the same thesis.

For each domain we list:

- the **market angle** (who is investing, why now),
- the **problem** in operational terms,
- **why a world model could matter**,
- **what should be measured**,
- and **why current LLM/VLM-style evaluation is insufficient**.

---

## 1. Autonomous driving

**Market angle**: Waymo, Tesla, Mobileye, Aurora, Wayve, Zoox, Nvidia (DRIVE), Hesai, and every major OEM run multi-billion-dollar programs. The 2023-2026 shift toward end-to-end and world-model-style stacks (Wayve GAIA, Tesla v12, Nvidia GR00T-driving variants, Genie-style learned simulators) has consolidated the thesis that **planning has to share representations with prediction** - which is what a world model is.

**Problem**: produce a steering and acceleration trajectory every ~50 ms, under partial sensor observability, with safety properties that have to hold across hundreds of millions of operational hours. The action space is small (continuous control), the consequence space is enormous (every other agent in the scene reacts).

**Why a world model could matter**: counterfactual rollout - "what happens if I brake, yield, accelerate, or change lane?" - cannot be answered by a perception model or an LLM. A learned world model predicts the joint trajectory of ego, agents, and scene under each candidate action, fast enough to compare alternatives inside a control cycle.

**What should be measured**: Action Success Rate of the planned trajectory against lane and time constraints; Planning Latency at p99 (must fit a 50 ms control loop on automotive-grade silicon); Perturbation Recovery under sensor noise, occlusion, and surprising agent behaviour; Surprise Detection so the model can hand off to a human or a fallback policy when out-of-distribution; Latent Interpretability for the regulator who wants to inspect "what the model thought was happening".

**Why current LLM/VLM evaluation is insufficient**: a VLM can describe a scene from a dashcam image. It cannot tell you whether the trajectory it proposes keeps the car inside lane bounds under the next two seconds of joint physics. The standard offline benchmarks (nuScenes, Waymo Open Dataset Q&A) measure perception accuracy, not closed-loop control safety.

---

## 2. Generalist embodied agents (humanoid and dexterous robotics)

**Market angle**: Figure AI ($2.6B valuation, 2024), 1X (OpenAI-backed), Apptronik (partnerships with Mercedes and Apple), Sanctuary AI, Tesla Optimus, Boston Dynamics. The thesis of the 2024-2026 cycle is a **single embodied foundation model** that can be re-tasked across factories, warehouses, last-mile delivery, and kitchens. Robot Transformers and vision-language-action models (OpenVLA, RT-X, Pi-0) are the language-model moment for robotics; world models are the dynamics half of that story.

**Problem**: a generalist robot has to attempt tasks it has never been specifically trained for, in environments with shifting layouts and human bystanders. Hard-coded behaviour trees and per-task policies do not scale beyond demonstrators. The model must imagine "if I grab the cup here, will it tip when I lift?" *before* the gripper closes.

**Why a world model could matter**: action-conditioned latent prediction lets a planner simulate candidate manipulations at kilohertz rates, orders of magnitude faster than any rigid-body simulator. Plan-then-act with a learned dynamics model is the only realistic path to closed-loop MPC on contact-rich, deformable, or partially-observed tasks running on on-board edge compute.

**What should be measured**: Action Success Rate per task family (pick, place, pour, insert, deform); Sample Efficiency on entirely new tasks (how many demonstrations until 80% success?); Perturbation Recovery under object slippage and unexpected contact; Compute per Decision (battery and edge-accelerator-bound).

**Why current LLM/VLM evaluation is insufficient**: standard "video question answering" or "manipulation success on a fixed task set" benchmarks tell you nothing about behaviour on tasks the model has not seen. Generalisation under real physics is the entire commercial premise of humanoid robotics, and exactly what static evaluation misses.

---

## 3. Industrial process control (advanced manufacturing, semiconductors, materials)

**Market angle**: semiconductor industry $600B annual revenue with shrinking nodes hitting physics limits; battery industry on track to exceed $400B by 2030; specialty chemicals and advanced materials are R&D-intensive sectors with multi-year iteration cycles. TSMC, ASML, Applied Materials, Samsung, every battery cell maker and every fab spend nine-figure capex on yield optimisation. The bottleneck is the wet-lab and fab feedback loop, not data.

**Problem**: design a fab recipe (process step parameters, time, temperature, dopant concentration, gas chemistry) or a material formulation (composition, crystallisation profile, additive blend) that hits a yield or performance target. Closed-loop optimisation on a fab is bottlenecked by the cost of each parameter sweep (millions of dollars per misconfigured wafer lot).

**Why a world model could matter**: a learned model of process dynamics, over PVD/CVD/etch step trajectories or over battery cycling response, gives a planner the ability to simulate dozens of candidate recipes per real-world iteration. The design cycle moves from days-per-experiment to hours-per-simulated-experiment.

**What should be measured**: Sample Efficiency (how few wet-lab iterations to converge on a target spec?); Action Success Rate against held-out target specifications; Planning Horizon at which predictions remain useful (multi-step processes); Latent Interpretability so process engineers and reliability teams can trust the model's intermediate states for root-cause analysis.

**Why current LLM/VLM evaluation is insufficient**: text-based benchmarks (PubChem-style QA, MatSci-NLP) measure factual recall. They do not measure whether the model can plan a recipe that yields a working device, nor whether its trajectory predictions hold up under physically-constrained closed-loop control.

---

## 4. Drug discovery and molecular design

**Market angle**: global pharma R&D spend exceeds $250B/year. The cost of bringing one new drug to market is roughly $2.5B over ~10 years, and the success rate from phase I to approval is below 10%. Isomorphic Labs (DeepMind spin-out), Insilico Medicine, Recursion Pharmaceuticals, Atomwise, BenevolentAI, Schrodinger, Genesis Therapeutics, plus every big pharma's internal AI group are racing to compress that cycle. A model that turns a wet-lab experiment into an in-silico simulation step is the most direct cost-cutting lever in the industry.

**Problem**: design a molecule that binds a target, is synthesisable, is non-toxic, and crosses the relevant biological barrier. The action space is "edit a molecular graph" or "propose a synthesis step"; the consequence space includes biological activity unfolding over days of cellular response.

**Why a world model could matter**: a learned dynamics model over molecular property trajectories, combined with a search procedure, can explore the chemical space at rates wet-lab synthesis cannot match. The model has to predict not just "what does this molecule look like" but "what happens downstream if I administer it" - which is exactly the planning-versus-prediction distinction this evaluation framework is built around.

**What should be measured**: Action Success Rate on retrospective drug discovery benchmarks (recovering known leads from random scaffolds); Planning Horizon (how far ahead in metabolic trajectories the model is still useful); Perturbation Recovery on noisy ADMET signals; Surprise Detection on out-of-distribution scaffolds where the model should *not* predict confidently.

**Why current LLM/VLM evaluation is insufficient**: medical and clinical LLM benchmarks (medMCQA, USMLE, MedQA) measure knowledge, not the ability to plan a multi-step synthesis or predict assay outcomes. Static molecular property benchmarks miss the trajectory dimension entirely - and trajectories are where drug development decisions actually live.

---

## 5. Energy grid balancing and renewable integration

**Market angle**: the global electricity market is around $1.4T/year. The shift to renewables (variable supply) plus rapid electrification (variable demand: EVs, heat pumps, data centres) has made grid balancing an order of magnitude harder in a decade. RTE (France), National Grid (UK), ENEL, Hitachi Energy, plus every ISO/RTO and major utility. DeepMind and others have published grid-scale RL results since 2016; the world-model angle is the natural next step.

**Problem**: balance supply and demand across a continental grid with sub-second tolerances, in the face of stochastic renewables, distributed storage, and shifting loads. A single mismatch event costs millions of euros; cascading failures can take down entire regions for hours or days.

**Why a world model could matter**: a model of grid response to dispatch actions, frequency-control actions, and market clearings enables MPC over both physical assets and contract structure. The state is high-dimensional (every node, every line, every storage asset) but the dynamics are governed by physics and contract structure - exactly the kind of structured-but-noisy domain where world models outperform reactive heuristics.

**What should be measured**: Action Success Rate against frequency stability constraints; Perturbation Recovery under loss-of-generation events and demand spikes; Compute per Decision (the controller must dominate the inertial timescale of the grid - milliseconds for frequency, minutes for market clearing); Planning Horizon, which decides which markets (real-time, intraday, day-ahead) the controller can usefully play in.

**Why current LLM/VLM evaluation is insufficient**: an LLM can summarise a grid incident report. It cannot predict the consequence of activating a peaker plant in the next 30 seconds, nor reason about what would have happened under a different dispatch policy.

---

## 6. Datacenter and AI workload orchestration

**Market angle**: hyperscale datacenter capex passed $200B/year in 2024 between Microsoft, Google, Meta, and Amazon, plus the AI-specific GPU buildout. Power, cooling, scheduling, and accelerator memory placement are the four binding constraints. Every minute of GPU underutilisation has a five-figure dollar cost on a modern cluster, and the marginal MW of power has begun to dictate where capacity goes.

**Problem**: schedule training and inference workloads across heterogeneous accelerators; predict and avoid thermal hotspots; manage power draw against utility constraints; defer non-critical work; place models for the shortest network path. The action space is "place workload X on accelerator Y at time T"; the consequence space includes thermal, network, queueing, and SLA effects that propagate over hours.

**Why a world model could matter**: predicting cluster behaviour under each candidate placement (including queueing, contention, thermal coupling) requires action-conditioned simulation. Reactive heuristics - bin-packing, classical schedulers, even modern co-scheduling - cannot reason counterfactually about consequences they have not directly observed.

**What should be measured**: Action Success Rate against SLO compliance; Compute per Decision (the controller cannot eat the workload it controls); Surprise Detection on out-of-distribution traffic patterns and failure signatures; Latent Interpretability for incident review and capacity planning.

**Why current LLM/VLM evaluation is insufficient**: an LLM can summarise a postmortem after the fact. The question that matters - "what would have happened under a different scheduling policy at 03:42:14?" - is counterfactual and requires an action-conditioned simulator running in the loop with the scheduler.

---

## 7. Logistics, supply chain, and disruption planning

**Market angle**: global supply chain software is a $20B/year market on top of trillions in physical goods movement. Maersk, DHL, Amazon Logistics, Flexport, Project44, plus every retailer with a national footprint. The post-2020 era of compounding disruptions (Suez, Red Sea, port strikes, tariff regime changes, climate events) has made *robustness under perturbation* the dominant procurement criterion, eclipsing point-optimal routing.

**Problem**: route pickers, AMRs, AGVs through warehouses; schedule fleets across multi-modal networks; reposition inventory ahead of demand; plan the response to a disruption event before it cascades. Classical solvers (MILP, OR-Tools, OptaPlanner) struggle when the map and demand are stochastic and when the right answer depends on counterfactual reasoning about the disruption's spread.

**Why a world model could matter**: a latent model of warehouse and network dynamics enables short-horizon planning that adapts to congestion and disruption without a full graph re-solve every tick. For supply-chain-scale problems, the model becomes the simulator the planner uses to reason about "what if the Suez closes for six weeks?" or "what if the Long Beach yard backs up?".

**What should be measured**: Average Steps to Success (proxy for routing efficiency); Perturbation Recovery on blocked aisles, port closures, dropped tasks; Planning Latency at the dispatch frequency; Sample Efficiency on new floor plans, new lanes, new modes.

**Why current LLM/VLM evaluation is insufficient**: a VLM can read a floor plan or a logistics dashboard. It cannot predict the second-order effects of routing 30 vehicles through one aisle, or what happens to your inventory turnover when the upstream supplier fails for two weeks. Spatial and temporal reasoning under counterfactual stress is what the use case demands.

---

## 8. Healthcare: treatment trajectory simulation

**Market angle**: US healthcare spending exceeds $4.5T/year; ICU and oncology care decisions sit at the high-cost, high-stakes end of that. Sepsis alone costs the US system ~$60B/year. The bottleneck is not data (EHR + labs + vitals is rich) but counterfactual planning - "what happens if I escalate to broad-spectrum antibiotics now versus in two hours?". Verily, Tempus, Hippocratic AI, Open Evidence, and every major academic medical centre with an AI/ML group are racing here.

**Problem**: predict patient state trajectories under candidate interventions (drug, dose, ventilator setting, fluid bolus) over a 24-72 hour window. The state is observed irregularly, the dynamics are noisy, and the cost of a wrong decision is measured in lives or in $250k+ ICU readmissions.

**Why a world model could matter**: latent dynamics over physiological state lets a clinician simulate the next six hours under three candidate treatments before committing to one. No retrospective study can substitute for prospective counterfactual reasoning on a specific patient-by-patient basis.

**What should be measured**: Action Success Rate (does the predicted trajectory match the observed one on held-out patients?); Perturbation Recovery (model robustness under sensor noise, missing values, late lab results); Latent Interpretability (clinicians and regulators must be able to ask "what is the model assuming about this patient?"); Surprise Detection (flag cases where the model is out of its training distribution and should defer).

**Why current LLM/VLM evaluation is insufficient**: a clinical LLM can pass USMLE-style multiple choice. It cannot tell you whether the trajectory it predicts under your proposed intervention is consistent with this specific patient's last 36 hours of labs and vitals. Counterfactual, patient-specific trajectory accuracy is the metric that matters, and it requires a world model evaluated against held-out longitudinal data.

---

## 9. Cyber defense and autonomous response

**Market angle**: enterprise security spend approaches $200B/year. The attacker-defender asymmetry - attackers compose multi-step kill chains over weeks, defenders detect single events in seconds - is the canonical example of an adversarial environment with delayed feedback. Microsoft Security Copilot, CrowdStrike Charlotte AI, Palo Alto Networks Cortex, SentinelOne Purple AI, plus every major SOC tool vendor are racing to close the loop with autonomous response.

**Problem**: predict the next-most-likely attacker action conditional on observed telemetry; pre-emptively contain. Wrong containment actions break production (a quarantined production host is an outage); missed actions cost millions in dwell time and lateral movement.

**Why a world model could matter**: a learned model of attacker-progression dynamics over the MITRE ATT&CK lattice gives a planner the ability to reason about "if I do nothing, what does the attacker do next? if I quarantine this host, what is the next move?". Pure rule-based detection is reactive by construction. Counterfactual planning is the only way to turn detection latency into decision time.

**What should be measured**: Action Success Rate against red-team scenarios; Perturbation Recovery under novel attacker techniques (zero-days and living-off-the-land variants); Surprise Detection AUROC for distinguishing benign anomaly from active intrusion; Planning Latency at the dispatch frequency of the SOC.

**Why current LLM/VLM evaluation is insufficient**: an LLM can summarise an attack timeline from logs after the fact. The decision that matters is what to do *next*, with the cost of being wrong asymmetric and the action space narrowing as the kill chain progresses. Static QA benchmarks measure factual knowledge, not adversarial planning under uncertainty.

---

## Common thread

Across all nine domains, the failure mode of "evaluate the model on static benchmarks and hope for the best" is the same: the metric does not measure the property that determines whether the deployment ships safely, cheaply, and reliably. The point of this repository is to push the planning question - **"useful for what, at what latency, at what compute cost, under what perturbations?"** - to the front of the conversation.

If even one of these markets matures around a shared evaluation layer, the cost of integrating a new world model into a production system drops by an order of magnitude. That is the bet this framework is built on.
