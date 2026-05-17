---
layout: default
title: "Counterfactual Planning Gap (paper)"
description: "Counterfactual Planning Gap: A Decision-Grade Metric for Decoupling Model Error from Planner Capacity in World Model Evaluation."
prev:
  title: "07 - Counterfactual Planning Gap"
  url: 07_cpg.html
next:
  title: "Back to home"
  url: index.html
---

<div class="paper-header">
  <p class="paper-eyebrow">Short paper &middot; v0.11.0</p>
  <h1 class="paper-title">Counterfactual Planning Gap</h1>
  <p class="paper-subtitle">A Decision-Grade Metric for Decoupling Model Error from Planner Capacity in World Model Evaluation</p>
  <p class="paper-author">Denis Hamon &nbsp;&middot;&nbsp; Independent &nbsp;&middot;&nbsp; <a href="mailto:denis.hamon1@gmail.com">denis.hamon1@gmail.com</a></p>
  <div class="paper-actions">
    <a class="btn-primary" href="https://github.com/Denis-hamon/world-model-eval-lab/raw/main/paper/main.pdf" download>Download PDF</a>
    <a class="btn-ghost" href="https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/main.tex">LaTeX source</a>
    <a class="btn-ghost" href="https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/references.bib">BibTeX</a>
  </div>
  <p class="paper-pdf-note">PDF is rebuilt by CI on every push that touches <code>paper/**</code> and committed back to <code>paper/main.pdf</code>. If the link 404s on the first visit after a tag, the CI build is in progress; refresh in a couple of minutes.</p>
</div>

## Abstract

Action-conditioned world models are routinely evaluated by prediction quality (reconstruction loss, frame-level FID, held-out one-step accuracy). Such metrics describe how well a model fits its training distribution. They are silent on the question that an applied team must answer before integrating a model into a control loop: *does the model, when used by a planner, produce decisions that succeed at the cost the deployment will accept?* We propose the **Counterfactual Planning Gap (CPG)**: the success-rate difference between a fixed planner using oracle dynamics and the same planner using the learned model on the same benchmark. The point estimate is the raw difference of success rates; the $95\%$ interval uses the Agresti--Caffo plus-4 adjustment, which keeps the variance positive at the boundary proportions $p \in \\{0, 1\\}$ where the standard Wald approximation collapses. We further define a five-branch verdict (`MODEL BOTTLENECK`, `LEARNED OUTPERFORMS ORACLE`, `PLANNER BOTTLENECK`, `MODEL AS GOOD AS ORACLE`, `INCONCLUSIVE`) gated on the lower bound of the CI rather than the raw point estimate, so under-powered runs cannot over-claim a diagnosis. We package CPG as a ~160-line addition to a reusable framework (`wmel`) and report a worked example on DeepMind Control Suite Acrobot-swingup. On $10$ episodes per arm we observe raw $\mathrm{CPG} = +0.300$, AC $95\%$ CI $[-0.06, +0.56]$, verdict `INCONCLUSIVE`. A multi-seed extension to $n = 150$ pooled per arm hardens the result to $\mathrm{CPG} = +0.267$, CI $[+0.191, +0.335]$, `MODEL BOTTLENECK`. Sweeping the MLP's training-set size by a factor of $100$ drops held-out validation MSE by $\sim\!150\times$ but leaves the verdict and CI *unchanged*: prediction quality improves dramatically, planning success stays at zero, the gap stays open. The metric does its job: it separates a model-capacity bottleneck from a data-coverage bottleneck that a prediction-quality metric alone would mask.

## 1. Introduction

The world-model literature has converged on three properties that motivate its existence: *prediction* of future observations conditioned on actions, *planning* by rolling those predictions out, and *transfer* across tasks that share latent structure [Ha & Schmidhuber, 2018; Hafner et al., 2023; Bardes et al., 2024; Bruce et al., 2024]. Most published evaluations focus on the first: reconstruction loss, frame-level FID, or next-frame prediction loss on held-out trajectories. These metrics are easy to compute and easy to compare across releases, but they are silent on a question any applied team must answer before integrating a learned world model into a control loop: *does using the model to plan produce decisions that succeed at the latency and compute the deployment will tolerate?*

The gap between prediction quality and decision quality is well documented. Hafner et al. (2019) report success rates on DeepMind Control Suite tasks alongside reconstruction loss because the two metrics disagree. Yu et al. (2020) and Kidambi et al. (2020) explicitly study *model exploitation* — the gap between a planner using a learned model and the same planner using the true environment dynamics — in offline model-based RL. Agarwal et al. (2021) document how rapidly point estimates of return mislead at small sample sizes, and prescribe interval reporting. What is missing is a packaged, reusable, decision-grade scalar that quantifies the gap with an honest confidence interval and a decision rule.

This paper makes three modest contributions:

1. We define the **Counterfactual Planning Gap (CPG)** as the success-rate difference between a planner using oracle dynamics and the same planner using a learned model, computed on identical benchmark runs that differ only in their `dynamics` callable (§3). The point estimate is the raw observed difference; the $95\%$ interval uses Agresti--Caffo plus-4. The verdict is gated on the AC lower bound.
2. We package CPG behind a minimal *evaluation contract* — a four-method adapter interface (`encode`, `rollout`, `score`, `plan`) that decouples the model from the runner and the metric (§2). Computing CPG is two calls to the same `BenchmarkRunner` plus one call to a metric function.
3. We report a worked example on DMC Acrobot-swingup with a Markovian MLP world model. The verdict at $n = 10$ is `INCONCLUSIVE`. A multi-seed extension at $n = 150$ pooled hardens to `MODEL BOTTLENECK`. A training-set-size sweep across two orders of magnitude leaves the verdict unchanged while the prediction loss drops $\sim\!150\times$ — the metric separates capacity from coverage (§4).

The framework is open source ([github.com/Denis-hamon/world-model-eval-lab](https://github.com/Denis-hamon/world-model-eval-lab)), runs CPU-only without a GPU, and has no heavy ML dependency at runtime; PyTorch and `dm-control` are pulled in only for the worked-example adapters. CI verifies the contract on Python 3.11/3.12/3.13.

## 2. The Evaluation Contract and a Decision-Grade Taxonomy {#sec-framework}

### 2.1 The four-method contract

Every adapter in the framework implements

$$
\begin{aligned}
\texttt{encode} \;&:\; \mathcal{O} \to \mathcal{Z}, \\
\texttt{rollout} \;&:\; \mathcal{Z} \times \mathcal{A}^H \to \mathcal{Z}^{H+1}, \\
\texttt{score} \;&:\; \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}, \\
\texttt{plan} \;&:\; \mathcal{O} \times \mathcal{O} \times \mathbb{N} \to \mathcal{A}^H.
\end{aligned}
$$

Here $\mathcal{O}$ is the observation space, $\mathcal{Z}$ the latent space (which may coincide with $\mathcal{O}$ for fully observed envs), $\mathcal{A}$ the action space, and $H$ the planning horizon. The runner queries `plan` on every replanning step and executes one or more of the returned actions in the real environment; the planner is free to use `encode`, `rollout`, and `score` internally however it sees fit. The contract makes no statement about how the model is trained.

For the present paper we use a random-shooting MPC: at each `plan` call, sample $N_\mathrm{cand}$ action sequences of length $H_\mathrm{plan}$ uniformly from $\mathcal{A}^{H_\mathrm{plan}}$, roll each one out through the dynamics, score the resulting trajectories, and return the best-scoring sequence prefix.

### 2.2 Decision-grade metrics

A metric is *decision-grade* iff (i) its units translate directly to a deployment-time cost or capability and (ii) it is computable only from closed-loop runs of the model, not from the model in isolation. Reconstruction loss and FID fail (ii); model-internal alignment scores fail (i). The metrics that survive both criteria are: action success rate, average steps to success, per-call planning latency, compute per decision, perturbation recovery rate, the effective planning horizon, and — the contribution of this paper — the Counterfactual Planning Gap.

## 3. Counterfactual Planning Gap {#sec-cpg}

### 3.1 Definition

Fix an environment $\mathcal{E}$, a planner $\pi$, a scoring function $\sigma$, a number of episodes $N$, a horizon $T$ and a seed. Let $\mathrm{success\\_rate}(D)$ denote the empirical success rate of $\pi$ on $N$ episodes of $\mathcal{E}$ when the planner queries dynamics $D$ during its internal rollouts. Let $D^{\star}$ be the *oracle* dynamics (the true env's transition function) and $D_\theta$ be a learned model. The Counterfactual Planning Gap is

$$
\mathrm{CPG} \;=\; \mathrm{success\\_rate}(D^{\star}) \;-\; \mathrm{success\\_rate}(D_\theta).
$$

All free quantities on the right-hand side — env, planner, score, $N$, $T$, seed — are held fixed between the two runs. The only thing that changes is the `dynamics` callable. This identification is what licenses interpreting CPG as a property of the *model* rather than the planner or the env.

### 3.2 Statistical reporting

Let $s_o, s_\ell$ be the success counts in the oracle and learned arms and $n_o, n_\ell$ the corresponding episode counts. The raw point estimate of CPG is the difference of proportions:

$$
\hat{\Delta} \;=\; \frac{s_o}{n_o} - \frac{s_\ell}{n_\ell}.
$$

The standard Wald $95\%$ CI on $\hat\Delta$ has variance $p_o(1-p_o)/n_o + p_\ell(1-p_\ell)/n_\ell$, which collapses to zero whenever *either* arm sits at $p \in \\{0, 1\\}$. With $n_\ell = 10$ episodes and a learned planner that fails on every episode, this is precisely the regime the framework lands in. A degenerate-variance CI in this regime falsely produces a tight interval and over-claims significance.

We instead use the Agresti--Caffo plus-4 adjustment [Agresti & Caffo, 2000], adding one pseudo-success and one pseudo-failure to each arm:

$$
\begin{aligned}
\tilde p_o &= \frac{s_o + 1}{n_o + 2}, \quad \tilde p_\ell = \frac{s_\ell + 1}{n_\ell + 2}, \\
\tilde \Delta &= \tilde p_o - \tilde p_\ell, \\
\mathrm{SE} &= \sqrt{\frac{\tilde p_o (1 - \tilde p_o)}{n_o + 2} + \frac{\tilde p_\ell (1 - \tilde p_\ell)}{n_\ell + 2}}, \\
\mathrm{CI}_{95\%}(\mathrm{CPG}) &= \bigl[\, \tilde\Delta - 1.96\,\mathrm{SE}, \;\; \tilde\Delta + 1.96\,\mathrm{SE} \,\bigr].
\end{aligned}
$$

The framework reports both the raw $\hat\Delta$ (what a reader expects to see) and the AC interval (what is statistically defensible). The two coincide for large $n$.

### 3.3 Gated verdict

A point estimate without a significance gate over-claims. The framework therefore exposes a five-branch decision rule that consults the AC interval, **not** the raw $\hat\Delta$:

<div class="verdict-legend" markdown="0">
  <p><span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span> &mdash; $\mathrm{CI}_{\mathrm{lo}} &gt; 0$. The oracle is reliably better; closing the gap is a model problem.</p>
  <p><span class="verdict-pill verdict-learned-outperforms">LEARNED OUTPERFORMS</span> &mdash; $\mathrm{CI}_{\mathrm{hi}} &lt; 0$. Rare; investigate regularisation or planner-search interactions.</p>
  <p><span class="verdict-pill verdict-planner-bottleneck">PLANNER BOTTLENECK</span> &mdash; CI crosses $0$ <em>and</em> both success rates are within $\tau$ of $0$. Neither planner solves the task.</p>
  <p><span class="verdict-pill verdict-as-good-as">MODEL AS GOOD AS ORACLE</span> &mdash; CI crosses $0$ <em>and</em> both success rates are within $\tau$ of $1$.</p>
  <p><span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span> &mdash; CI crosses $0$ in a middle-of-the-road regime. Run more episodes.</p>
</div>

Default tolerance $\tau = 0.05$. Crucially, `MODEL BOTTLENECK` is *not* the default when $\hat\Delta > 0$; it requires the AC lower bound to be strictly positive.

### 3.4 Properties

CPG is bounded in $[-1, 1]$, antisymmetric in the role of the two dynamics, and additive: on a benchmark suite split into disjoint sub-tasks, CPG on the union is the episode-weighted mean of the per-sub-task CPGs. The metric is silent about *why* the model degrades planning (latent shift, prediction-error accumulation, score-function mismatch); follow-up diagnostics are needed to attribute. It is, however, sufficient to distinguish model-side failures from planner-side failures at the verdict granularity.

## 4. Empirical Study: DMC Acrobot-Swingup {#sec-empirical}

### 4.1 Setup

We use DMC Acrobot-swingup [Tassa et al., 2018; Tunyasuvunakool et al., 2020]: a two-link underactuated pendulum with a continuous torque action in $[-1, +1]$ that must build energy and balance the tip upright. We discretise the action to a five-level torque set $\\{-1, -0.5, 0, +0.5, +1\\}$ to fit the framework's hashable-action contract. The observation is a six-dimensional vector $(\sin\theta_1, \sin\theta_2, \cos\theta_1, \cos\theta_2, \dot\theta_1, \dot\theta_2)$ in the layout returned by `dm_control.suite.acrobot.Physics.orientations`. Success at step $t$ is $r_t \geq 0.6$ where $r_t$ is the DMC dense reward.

The scoring function is $\sigma(\mathbf{o}, \cdot) = -(\cos\theta_1 + \cos\theta_2)$, a unit-length approximation of the negative tip height. The planner is random-shooting MPC with $N_\mathrm{cand} = 50$ candidate sequences of length $H_\mathrm{plan} = 15$, executed at every replanning step. Episodes run for at most $T = 500$ env steps. Seed $0$ throughout.

### 4.2 Oracle dynamics

We construct the oracle by instantiating a private `dm_control` environment inside a $(\mathrm{state}, \mathrm{action}) \to \mathrm{state}$ callable. Each call reconstructs $(q_\mathrm{pos}, q_\mathrm{vel})$ from the flat observation via $\mathrm{atan2}$, writes them into the private env's MuJoCo physics, calls `physics.forward`, steps once with the candidate torque, and returns the new flat observation. We verify the oracle reproduces `env.step` to numerical precision ($|\Delta| < 10^{-5}$) over a fifty-step random-policy rollout; this regression test is part of CI.

### 4.3 Learned dynamics

We train a Markovian MLP (two hidden layers of width $64$, ReLU activations) on $2\,000$ random-policy transitions collected from ten episodes of $200$ steps each. The input concatenates the six-dimensional observation with a five-dimensional one-hot action; the output predicts the next observation. Training: $200$ epochs with Adam (lr $10^{-3}$), batch size $256$, $10\%$ held-out validation split at the transition level. Final validation MSE $0.026$.

### 4.4 Results at $n = 10$

The oracle planner reaches the upright pose in $30\%$ of the episodes; the learned planner in $0\%$. The raw CPG is $+0.30$; the Agresti--Caffo $95\%$ interval is $[-0.06, +0.56]$, which crosses zero. The verdict is `INCONCLUSIVE`.

<table class="paper-table">
  <thead>
    <tr><th></th><th>Oracle dynamics</th><th>Learned MLP dynamics</th></tr>
  </thead>
  <tbody>
    <tr><td>Success rate</td><td>0.30 (3/10)</td><td>0.00 (0/10)</td></tr>
    <tr><td>Avg. steps to success</td><td>180.7</td><td>n/a</td></tr>
    <tr><td>Per-call planning latency (ms)</td><td>77.3</td><td>65.3</td></tr>
    <tr><td>Compute per decision (rollout-units)</td><td>407.1</td><td>157.3</td></tr>
  </tbody>
</table>

<table class="paper-table paper-table-narrow">
  <thead>
    <tr><th colspan="2">Counterfactual Planning Gap</th></tr>
  </thead>
  <tbody>
    <tr><td>Raw $\hat\Delta$</td><td>+0.300</td></tr>
    <tr><td>Agresti--Caffo 95% CI</td><td>[-0.059, +0.559]</td></tr>
    <tr><td>Verdict</td><td><span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span></td></tr>
  </tbody>
</table>

Three readings are consistent with the data, and the framework declines to choose between them at $n = 10$: *model bottleneck*, *sample-size artifact*, *score-function mismatch*. A multi-seed extension that pushes $N$ to $\sim 100$ episodes per arm would tighten the AC half-width by roughly $\sqrt{10}$. **The metric's job at $n = 10$ is to refuse to choose; it does, correctly.** The next subsection resolves between the three readings by delivering that extension.

### 4.5 Multi-seed extension across training-set sizes {#sec-sweep}

We extend along two axes. *Episodes per arm* grows from $10$ to $50$ per seed (pooled across three seeds, $n = 150$ per arm), pushing the AC half-width below $0.10$. *Training-set size* sweeps the MLP's data budget across nearly three orders of magnitude: $\\{200, 2\,000, 20\,000\\}$ random-policy transitions. Every other quantity is held fixed.

The result is striking. The MLP's held-out validation MSE drops by a factor of $\sim\!150$; the learned planner's success rate stays at *exactly zero* in all $450$ benchmark episodes; the oracle planner's success rate is identical across cells; CPG returns the same point estimate, the same AC CI, and the same verdict `MODEL BOTTLENECK` in every cell.

<table class="paper-table">
  <thead>
    <tr><th>Train size</th><th>Val MSE</th><th>Oracle success</th><th>Learned success</th><th>CPG verdict</th></tr>
  </thead>
  <tbody>
    <tr><td>200</td><td>0.0651</td><td>40/150 = 0.267</td><td>0/150 = 0.000</td><td>+0.267, CI [+0.19, +0.33], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>2 000</td><td>0.0233</td><td>40/150 = 0.267</td><td>0/150 = 0.000</td><td>+0.267, CI [+0.19, +0.33], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>20 000</td><td>0.0004</td><td>40/150 = 0.267</td><td>0/150 = 0.000</td><td>+0.267, CI [+0.19, +0.33], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
  </tbody>
</table>

The $n = 10$ reading admitted three explanations. The multi-seed result rules out the sample-size artifact (the CI no longer crosses zero) and is consistent with the score-function-mismatch hypothesis only as a contributor, not as the primary driver. The dominant story is *model bottleneck* — with a precise attribution that prediction quality alone obscures.

### 4.6 What CPG separates: capacity vs. coverage

A naive reading is that the MLP simply needs more capacity. The validation MSE refutes this directly: at $20\,000$ transitions the model fits the training distribution to within $4\times 10^{-4}$, indistinguishable from numerical noise. The model has *ample* capacity for what it has been asked to predict.

What it has *not* been asked to predict is the upright-balancing regime. Random-policy rollouts in Acrobot rarely reach high-energy configurations; the upright pose is essentially absent from the training distribution. The model is therefore extrapolating during planning, and its predictions, accurate on the training manifold, are unreliable off it. The planner is misled, and success collapses.

This is most parsimoniously read as a *coverage* bottleneck rather than a *capacity* bottleneck. CPG cannot tell the two apart on its own, but in conjunction with held-out validation it can: a flat CPG curve across data-size sweeps, paired with a monotonically decreasing prediction loss, points to a data-distribution problem rather than to model size or architecture. Two second-order contributors are not ruled out by this experiment: a planner that is too weak to exploit a perfect model in the high-energy regime (the oracle planner reaches upright in only $27\%$ of episodes, so a stronger search procedure — CEM, gradient-based MPC — might lift both arms), and a score function $\sigma$ that approximates rather than matches the DMC reward. We treat coverage as the dominant explanation given the $\sim\!150\times$ drop in validation loss with no movement in success.

**Empirical receipt for the coverage claim.** We measure the visited-state distribution directly. On the natural "uprightness" axis $u(\mathbf{o}) = \cos\theta_1 + \cos\theta_2 \in [-2, +2]$ (upright pose at $+2$):

<table class="paper-table">
  <thead>
    <tr><th>Dataset</th><th>$n$ states</th><th>Mean $u$</th><th>Max $u$</th><th>Frac $u > 1.0$</th><th>Frac $u > 1.5$</th></tr>
  </thead>
  <tbody>
    <tr><td>Random rollouts</td><td>2 000</td><td>$-0.503$</td><td>$+0.865$</td><td><strong>0.00%</strong></td><td><strong>0.00%</strong></td></tr>
    <tr><td>Oracle planner</td><td>846</td><td>$+0.161$</td><td>$+1.866$</td><td>20.2%</td><td>12.2%</td></tr>
  </tbody>
</table>

The upright regime that swing-up requires is **strictly absent** from the training distribution: $0/2000$ random-rollout states have $u > 1.0$. The oracle planner visits that regime in roughly one-fifth of its trajectory. The MLP has never been shown a state from which the planner needs to predict. Numbers from [`results/dmc_acrobot/coverage.json`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/results/dmc_acrobot/coverage.json); the script is [`experiments/dmc_acrobot/coverage_analysis.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/experiments/dmc_acrobot/coverage_analysis.py).

A second-axis sweep that varies the exploration policy under fixed data size would directly confirm coverage as the dominant driver. The natural remediation we recommend is to change the data-collection policy (energy-aware exploration, or relabelled trajectories that visit the swing-up regime), not to enlarge the network.

## 5. Discussion and Limitations

The empirical demonstration is intentionally narrow: one environment, one learned model, one planner, one seed family ($\{0, 1, 2\}$). The methodological contribution — a packaged CPG with an Agresti--Caffo CI and a gated verdict — is decoupled from any of those choices and applies wherever an oracle dynamics is available (so: simulated environments). On hardware-in-the-loop or physical environments the metric is undefined; surrogate CPG variants (a higher-fidelity model standing in for the oracle) are future work.

The framework's adapter interface is what makes CPG cheap to compute: any `dynamics` callable can be swapped into the same planner without touching anything else. We have so far demonstrated this on two stdlib-only callables (a tabular maze, an Acrobot oracle) and two learned callables (an MLP on the maze that memorises transitions, an MLP on Acrobot that generalises). A natural next step is to plug in a published research-grade world model (Dreamer-V3, TD-MPC2, or a JEPA-based predictor) and report CPG against the same oracle.

Other limitations worth flagging. The discrete-torque action space is a five-level approximation of the continuous control problem; continuous CPG variants would benefit from a cross-entropy-method (CEM) planner instead of random shooting. The Acrobot success criterion ($r_t \geq 0.6$) is a binary projection of a dense reward; a continuous variant of CPG that reports the difference of mean returns is straightforward and may be more sample-efficient. The planner score is task-specific and tightly coupled to the environment's reward geometry; a learned score function (predicting the DMC reward directly from the latent state) would remove this coupling, at the cost of a second model component.

## 6. Conclusion

We have proposed the Counterfactual Planning Gap as a decision-grade metric for world-model evaluation, packaged it behind a minimal evaluation contract, and reported a worked example on DMC Acrobot-swingup. At $n = 10$ the framework reports `INCONCLUSIVE`, which is the correctly-calibrated verdict at that sample size. The multi-seed extension to $n = 150$ pooled per arm tightens the CI off zero and returns `MODEL BOTTLENECK`; sweeping the training-set size across nearly three orders of magnitude leaves the verdict unchanged while the held-out prediction loss drops by $\sim\!150\times$. The takeaway is methodological: a metric that separates closed-loop success from prediction quality reveals that the bottleneck here is not the size of the model but the coverage of its training distribution — a diagnosis a prediction-quality metric alone would mask. We argue this kind of calibrated honesty — a metric that refuses to claim more than the data supports, and that supports a precise attribution once the data is sufficient — is the right design target for a methodology paper that wants to sit usefully next to a fast-moving model literature.

## Acknowledgements

This work is independent. It is not affiliated with the AMI (Advanced Machine Intelligence) program at Meta, the LeWorldModel project, the authors of any of the cited papers, or any current or past employer of the author. The repository was developed end-to-end with an LLM coding agent in the loop (Claude Code); a description of the development recipe and the pre-tag adversarial-review pattern is included in the repository.

## References

Full BibTeX in [`paper/references.bib`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/references.bib).

- Agarwal, Schwarzer, Castro, Courville, Bellemare (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS.
- Agresti, Caffo (2000). *Simple and Effective Confidence Intervals for Proportions and Differences of Proportions Result from Adding Two Successes and Two Failures.* The American Statistician.
- Assran et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* CVPR.
- Bardes et al. (2024). *V-JEPA: Latent Video Prediction for Visual Representation Learning.*
- Bardes et al. (2025). *V-JEPA 2.*
- Bruce et al. (2024). *Genie: Generative Interactive Environments.* ICML.
- Bruce et al. (2024). *Genie 2.*
- Ha, Schmidhuber (2018). *World Models.* arXiv:1803.10122.
- Hafner et al. (2019). *PlaNet — Learning Latent Dynamics for Planning from Pixels.* ICML.
- Hafner et al. (2020). *Dreamer — Dream to Control: Learning Behaviors by Latent Imagination.* ICLR.
- Hafner et al. (2023). *Dreamer-V3 — Mastering Diverse Domains through World Models.* arXiv:2301.04104.
- Hansen et al. (2024). *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR.
- Henderson et al. (2018). *Deep Reinforcement Learning that Matters.* AAAI.
- Kidambi, Rajeswaran, Netrapalli, Joachims (2020). *MOReL: Model-Based Offline Reinforcement Learning.* NeurIPS.
- LeCun (2022). *A Path Towards Autonomous Machine Intelligence.* Open Review.
- Liu et al. (2023). *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.*
- Micheli, Alonso, Fleuret (2023). *Transformers are Sample-Efficient World Models (IRIS).* ICLR.
- Newcombe (1998). *Interval Estimation for the Difference Between Independent Proportions: Comparison of Eleven Methods.* Statistics in Medicine.
- Park et al. (2024). *OGBench: Benchmarking Offline Goal-Conditioned RL.*
- Schäfer, Udluft, Zimmermann (2007). *The Recurrent Control Neural Network.* Engineering Applications of Artificial Intelligence.
- Schrittwieser et al. (2020). *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero).* Nature.
- Tassa et al. (2018). *DeepMind Control Suite.*
- Tunyasuvunakool et al. (2020). *dm_control: Software and Tasks for Continuous Control.* Software Impacts.
- Wilson (1927). *Probable Inference, the Law of Succession, and Statistical Inference.* JASA.
- Yu, Thomas, Yu, Ermon, Zou, Levine, Finn, Ma (2020). *MOPO: Model-Based Offline Policy Optimization.* NeurIPS.
