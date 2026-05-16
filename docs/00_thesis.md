---
prev:
  title: "Home"
  url: /
next:
  title: "01 - The evaluation gap"
  url: 01_evaluation_gap.html
---
# 00 - Thesis

## Static AI benchmarks are not enough for world models

Static AI benchmarks reward a model for producing the right output on a fixed input. That works well for classification, captioning, or single-turn question answering. It does not work well for a class of models whose value is **what they let an agent do next**.

A world model is, by construction, a model that predicts the consequences of actions. Evaluating it by next-frame reconstruction or static prediction loss is like evaluating a navigation system by how pretty its rendered map looks - it measures something, but not the thing that matters.

The thing that matters is whether the model lets an agent:

- choose a useful action,
- recover when something unexpected happens,
- generalize across tasks it was not explicitly trained for,
- and do all of the above fast enough and cheap enough that a product can ship.

## Action-conditioned planning needs decision-grade evaluation

Once a world model is plugged into a planner, evaluation stops being a single number. It becomes a profile:

- **Decision quality** - does the chosen action sequence succeed?
- **Latency** - how long did it take to produce that sequence?
- **Compute cost** - what did it cost in FLOPs, energy, or dollars per decision?
- **Robustness** - what happens when the environment is perturbed mid-rollout?
- **Generality** - how does the same model perform on a related but unseen task?

Each of these is an applied constraint as much as a research metric. A model that wins on next-frame prediction loss but takes 800 ms per decision is not deployable in robotics. A model that wins on success rate in a fixed seed but collapses under a 5 percent perturbation is not deployable in control. A model that requires fine-tuning per task is not yet a generalist agent.

These are not exotic concerns. They are the first questions any applied team would ask before integrating a world model into a real system - and they are largely missing from how the research community currently reports results.

## Core thesis

> The next bottleneck for world models is not only model quality. It is proof of usefulness.

Better predictors will keep being published. The community needs a shared, lightweight, opinionated way to ask **"useful for what, and at what cost?"** - and to answer that question with numbers a non-researcher can read.

This repository is one attempt at that shared layer.

## Related work

The world-model literature this study sits next to (not on top of). All citations are intentional, not affiliational.

**Predictive / latent world models**

- Ha & Schmidhuber, *World Models*, 2018.
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet), ICML 2019.
- Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer), ICLR 2020.
- Hafner et al., *Mastering Diverse Domains through World Models* (Dreamer-V3), 2023.
- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero), Nature 2020.
- Micheli, Alonso & Fleuret, *Transformers are Sample-Efficient World Models* (IRIS), ICLR 2023.
- Robine et al., *TransDreamer: Reinforcement Learning with Transformer World Models*, 2022.
- Bruce et al., *Genie: Generative Interactive Environments*, ICML 2024.
- Bruce et al., *Genie 2: A large-scale foundation world model*, 2024.

**Joint-embedding predictive architectures (the AMI line of work)**

- LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022.
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA), CVPR 2023.
- Bardes et al., *V-JEPA: Latent Video Prediction for Visual Representation Learning*, 2024.
- Bardes et al., *V-JEPA 2*, 2025.

**Action-conditioned planning evaluation**

- Tassa et al., *DeepMind Control Suite*, 2018.
- Park et al., *OGBench: Benchmarking Offline Goal-Conditioned RL*, 2024.
- Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, NeurIPS 2023.
- Hansen, Su & Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control*, ICLR 2024.

**Evaluation methodology (statistics, reproducibility)**

- Wilson, *Probable Inference, the Law of Succession, and Statistical Inference*, JASA 1927.
- Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018.
- Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice* (rliable), NeurIPS 2021.

This list is selective and is expected to grow as the study moves from toy environments to the real benchmarks listed in the [30-day plan](05_30_day_prototype_plan.html).