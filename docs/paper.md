---
layout: default
title: "Counterfactual Planning Gap (paper)"
description: "Counterfactual Planning Gap: An Interval-Gated Statistic for Diagnosing World-Model Bottlenecks under a Fixed Planner."
prev:
  title: "07 - Counterfactual Planning Gap"
  url: 07_cpg.html
next:
  title: "Back to home"
  url: index.html
---

<div class="paper-header">
  <p class="paper-eyebrow">Short paper &middot; v0.18.0</p>
  <h1 class="paper-title">Counterfactual Planning Gap</h1>
  <p class="paper-subtitle">An Interval-Gated Statistic for Diagnosing World-Model Bottlenecks under a Fixed Planner</p>
  <p class="paper-author">Denis Hamon &nbsp;&middot;&nbsp; Independent &nbsp;&middot;&nbsp; <a href="mailto:denis.hamon1@gmail.com">denis.hamon1@gmail.com</a></p>
  <div class="paper-actions">
    <a class="btn-primary" href="https://github.com/Denis-hamon/world-model-eval-lab/raw/main/paper/main.pdf" download>Download PDF</a>
    <a class="btn-ghost" href="https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/main.tex">LaTeX source</a>
    <a class="btn-ghost" href="https://github.com/Denis-hamon/world-model-eval-lab/blob/main/paper/references.bib">BibTeX</a>
  </div>
  <p class="paper-pdf-note">PDF is rebuilt by CI on every push that touches <code>paper/**</code> and committed back to <code>paper/main.pdf</code>. If the link 404s on the first visit after a tag, the CI build is in progress; refresh in a couple of minutes.</p>
</div>

## Abstract

Action-conditioned world models are usually evaluated by prediction quality (reconstruction loss, frame-level FID, held-out accuracy), which is silent on the question an applied team must answer before integrating a model into a control loop: *does the model, when used by a planner, produce decisions that succeed?* We propose the **Counterfactual Planning Gap (CPG)**: the success-rate difference between a fixed planner using oracle dynamics and the same planner using the learned model, on identical runs that differ only in the `dynamics` callable. We report it with an Agresti--Caffo plus-4 interval (which keeps the variance positive at the boundary proportions $p \in \{0,1\}$ where the Wald approximation collapses), a paired bootstrap where the design warrants it, and a five-branch verdict (`MODEL BOTTLENECK`, `LEARNED OUTPERFORMS ORACLE`, `PLANNER BOTTLENECK`, `MODEL AS GOOD AS ORACLE`, `INCONCLUSIVE`) gated on the lower bound of the CI rather than the point estimate, so under-powered runs cannot over-claim a diagnosis. We package CPG behind a minimal evaluation contract (`wmel`) and exercise it on three DeepMind Control Suite tasks. The central finding is that the verdict is *heterogeneous and condition-sensitive* — precisely what a calibrated metric should surface, and what a point-estimate leaderboard cannot. On Acrobot-swingup the gap looks large only because the oracle's fixed initial state is an unusually easy swing-up; sampling the task's initial-state distribution collapses the oracle to $\sim\!3\%$ success and flips the verdict from `MODEL BOTTLENECK` to `PLANNER BOTTLENECK`. On Reacher-easy the verdict is `MODEL BOTTLENECK` across all arms ($\mathrm{CPG}$ from $+0.20$ to $+0.33$). On Cartpole-swingup at higher model capacity the larger TD-MPC2 [Hansen et al., 2024] under a Cross-Entropy-Method planner *beats* the oracle planner: `LEARNED OUTPERFORMS ORACLE`, $\mathrm{CPG} = -0.27$, AC CI $[-0.48, -0.02]$ and paired-bootstrap CI $[-0.50, -0.03]$, both clearing zero. We also present this as *self-correction*: an earlier single-fixed-initial-state evaluation of this same framework reported `MODEL BOTTLENECK` on Acrobot; the metric's own interval-gated machinery, re-run over the task distribution, overturned that headline. Finally, because the gate is a function of the confidence interval, it doubles as a power-analysis tool: we give the per-arm episode count a comparison needs before its interval clears zero, and show that a plausible leaderboard near-tie ($0.94$ vs $0.92$ at $n = 100$) is statistically indistinguishable from noise.

## 1. Introduction

The world-model literature has converged on three properties that motivate its existence: *prediction* of future observations conditioned on actions, *planning* by rolling those predictions out, and *transfer* across tasks that share latent structure [Ha & Schmidhuber, 2018; Hafner et al., 2023; Bardes et al., 2024; Bruce et al., 2024]. Most published evaluations focus on the first: reconstruction loss, frame-level FID, or next-frame prediction loss on held-out trajectories. These metrics are easy to compute and easy to compare across releases, but they are silent on a question any applied team must answer before integrating a learned world model into a control loop: *does using the model to plan produce decisions that succeed at the latency and compute the deployment will tolerate?*

The gap between prediction quality and decision quality is well documented. Hafner et al. (2019) report success rates on DeepMind Control Suite tasks alongside reconstruction loss because the two metrics disagree. Yu et al. (2020) and Kidambi et al. (2020) explicitly study *model exploitation* — the gap between a planner using a learned model and the same planner using the true environment dynamics — in offline model-based RL. Agarwal et al. (2021) document how rapidly point estimates of return mislead at small sample sizes, and prescribe interval reporting. These are all healthy signals; what is missing is a packaged, reusable, decision-grade scalar that quantifies the gap with an honest confidence interval and a decision rule.

This paper makes three modest contributions:

1. **An interval-gated decision rule for the planning gap.** We define the **Counterfactual Planning Gap (CPG)** — the success-rate difference between a fixed planner using oracle dynamics and the same planner using a learned model, on identical runs differing only in their `dynamics` callable (§3) — and report it with an Agresti--Caffo plus-4 interval, a paired bootstrap where the design is paired, and a five-branch verdict gated on the CI lower bound. The diagnosis is thus only as strong as the evidence: under-powered runs return `INCONCLUSIVE` rather than over-claim. The underlying quantity is the model-exploitation gap of offline model-based RL [Yu et al., 2020; Kidambi et al., 2020]; the contribution is the packaged, gated, interval-reported *statistic and decision procedure*, not the subtraction.
2. **Heterogeneous verdicts across three tasks.** Exercised on three DeepMind Control Suite tasks (Acrobot-swingup, Cartpole-swingup, Reacher-easy), the gate fires four of its five branches on real data (§4.4–§4.6): `PLANNER BOTTLENECK` on Acrobot (the oracle planner itself solves only $\sim\!3\%$ of random initial states), `MODEL BOTTLENECK` on Reacher, `LEARNED OUTPERFORMS ORACLE` on high-capacity Cartpole under CEM, and `INCONCLUSIVE` on several moderate-$n$ near-ties. A fixed metric, or a point-estimate leaderboard, reports none of this structure.
3. **The metric as self-correction.** An earlier version of this very framework, evaluated at a single fixed initial state, reported a large `MODEL BOTTLENECK` gap on Acrobot. Re-running over the task's initial-state distribution — the design the metric's own honesty discipline demands — collapses the oracle and overturns the verdict to `PLANNER BOTTLENECK` (§4.4). The episode is the strongest evidence for the metric's purpose: a calibrated, interval-gated statistic caught a config-sensitive artifact that a point estimate would have published.

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

Fix an environment $\mathcal{E}$, a planner $\pi$, a scoring function $\sigma$, a number of episodes $N$, a horizon $T$ and a seed. Let $\mathrm{success\_rate}(D)$ denote the empirical success rate of $\pi$ on $N$ episodes of $\mathcal{E}$ when the planner queries dynamics $D$ during its internal rollouts. Let $D^{\star}$ be the *oracle* dynamics (the true env's transition function) and $D_\theta$ be a learned model. The Counterfactual Planning Gap is

$$
\mathrm{CPG} \;=\; \mathrm{success\_rate}(D^{\star}) \;-\; \mathrm{success\_rate}(D_\theta).
$$

All free quantities on the right-hand side — env, planner, score, $N$, $T$, and the initial-state distribution — are held fixed between the two runs. The only thing that changes is the `dynamics` callable, so CPG attributes the success-rate difference to that callable, holding everything else fixed. It is therefore a property of the (model, planner, distribution) triple, *not* a planner- or distribution-free property of the model alone: as §4.4 shows, the same model can flip the verdict when only the evaluation distribution changes. Read CPG relative to a stated planner and a stated distribution.

### 3.2 Statistical reporting

Let $s_o, s_\ell$ be the success counts in the oracle and learned arms and $n_o, n_\ell$ the corresponding episode counts. The raw point estimate of CPG is the difference of proportions:

$$
\hat{\Delta} \;=\; \frac{s_o}{n_o} - \frac{s_\ell}{n_\ell}.
$$

The standard Wald $95\%$ CI on $\hat\Delta$ has variance $p_o(1-p_o)/n_o + p_\ell(1-p_\ell)/n_\ell$, which collapses to zero whenever *either* arm sits at $p \in \{0, 1\}$. With $n_\ell = 10$ episodes and a learned planner that fails on every episode, this is precisely the regime the framework lands in. A degenerate-variance CI in this regime falsely produces a tight interval and over-claims significance.

We instead use the Agresti--Caffo plus-4 adjustment [Agresti & Caffo, 2000], adding one pseudo-success and one pseudo-failure to each arm:

$$
\begin{aligned}
\tilde p_o &= \frac{s_o + 1}{n_o + 2}, \quad \tilde p_\ell = \frac{s_\ell + 1}{n_\ell + 2}, \\
\tilde \Delta &= \tilde p_o - \tilde p_\ell, \\
\mathrm{SE} &= \sqrt{\frac{\tilde p_o (1 - \tilde p_o)}{n_o + 2} + \frac{\tilde p_\ell (1 - \tilde p_\ell)}{n_\ell + 2}}, \\
\mathrm{CI}_{95\%}(\mathrm{CPG}) &= \bigl[\, \tilde\Delta - 1.96\,\mathrm{SE}, \;\; \tilde\Delta + 1.96\,\mathrm{SE} \,\bigr].
\end{aligned}
$$

The framework reports both the raw $\hat\Delta$ (what a reader expects to see) and the AC interval (what is statistically defensible). The two coincide for large $n$. Where the design is paired — the two arms share each episode's initial state under varied-init sampling — we additionally report a paired bootstrap CI on the non-degenerate cells, which AC cannot exploit.

We adopt the closed-form Agresti--Caffo interval rather than a bootstrap [cf. Agarwal et al., 2021] for two reasons specific to this setting: at the boundary proportions $p \in \{0, 1\}$ that recur here (a learned arm at $0/n$), a nonparametric bootstrap of the success rate degenerates to a zero-width interval — the same pathology that motivates avoiding the Wald interval — whereas the plus-4 adjustment keeps the variance positive; and only a closed-form half-width is a function of $(n, p_o, p_\ell)$ that can be inverted *before* any rollouts, which is what makes the power analysis of §4.7 possible.

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

### 3.5 Robustness to the paired design {#sec-paired}

Under varied-init sampling the two arms share each episode's initial state, so the design is *paired* and the AC variance above is written for *independent* proportions. Whether that assumption distorts the verdicts is an empirical question about the within-pair correlation $\phi$, which we measure directly (table below): it is $\approx 0$ in the four Reacher cells and only slightly negative on Cartpole ($-0.15$ and $-0.04$), so the paired intervals here are essentially interchangeable with AC — Newcombe's half-width is within a few points of AC's, never tighter — and the independence assumption is neither inflating nor deflating the verdict. We keep AC as the primary report (it is the only estimator whose half-width is a closed form in $(n, p_o, p_\ell)$ invertible *before* any rollouts, and the right tool for the boundary $0/n$ cells where a paired bootstrap degenerates), but we nonetheless corroborate every non-degenerate cell with two paired-design estimators — Newcombe's paired-difference interval [Newcombe, 1998, paired-data method] and a paired percentile bootstrap — and a stricter exact McNemar test [McNemar, 1947] with a Holm family-wise correction across cells.

The three intervals agree on whether the interval clears zero in *every* one of the six non-degenerate cells, so the CI-gated verdicts are **not** artifacts of the independence assumption. The exact McNemar test is more demanding: after Holm correction the larger-gap MLP cells stay significant, while the smaller-gap TD-MPC2 cells at $n = 30$ — including the `LEARNED OUTPERFORMS ORACLE` cell — fall short of family-wise significance, exactly the regime the power analysis flags for the pooled-$150$ follow-up.

<table class="paper-table">
  <thead>
    <tr><th>Cell ($n=30$)</th><th>$\hat\Delta$</th><th>$\phi$</th><th>AC CI</th><th>Newcombe CI</th><th>paired-boot CI</th><th>McNemar $p$ (Holm)</th></tr>
  </thead>
  <tbody>
    <tr><td>Reacher RS &times; MLP</td><td>+0.300</td><td>0.00</td><td>[+0.11, +0.45]</td><td>[+0.12, +0.48]</td><td>[+0.13, +0.47]</td><td>0.004 (0.019)</td></tr>
    <tr><td>Reacher RS &times; TD-MPC2</td><td>+0.200</td><td>0.00</td><td>[+0.03, +0.34]</td><td>[+0.05, +0.37]</td><td>[+0.07, +0.33]</td><td>0.031 (0.094)</td></tr>
    <tr><td>Reacher CEM &times; MLP</td><td>+0.333</td><td>0.00</td><td>[+0.14, +0.49]</td><td>[+0.15, +0.51]</td><td>[+0.17, +0.50]</td><td>0.002 (0.012)</td></tr>
    <tr><td>Reacher CEM &times; TD-MPC2</td><td>+0.233</td><td>0.00</td><td>[+0.06, +0.38]</td><td>[+0.07, +0.41]</td><td>[+0.10, +0.40]</td><td>0.016 (0.062)</td></tr>
    <tr><td>Cartpole RS &times; TD-MPC2 (size 5)</td><td>+0.167</td><td>&minus;0.15</td><td>[-0.02, +0.34]</td><td>[-0.03, +0.36]</td><td>[-0.03, +0.37]</td><td>0.180 (0.180)</td></tr>
    <tr><td>Cartpole CEM &times; TD-MPC2 (size 5)</td><td>&minus;0.267</td><td>&minus;0.04</td><td>[-0.48, -0.02]</td><td>[-0.48, -0.01]</td><td>[-0.50, -0.03]</td><td>0.077 (0.154)</td></tr>
  </tbody>
</table>

The measured within-pair correlation $\phi$, three interval estimators (independent-proportions AC plus two paired-design estimators), and the exact McNemar test (Holm-adjusted $p$ in parentheses), $n = 30$ pooled. With $\phi \approx 0$ to slightly negative the paired intervals are not tighter than AC; they nonetheless agree on clearing zero in every cell. Regenerated from committed per-episode data by [`experiments/paired_intervals_audit.py`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/experiments/paired_intervals_audit.py).

### 3.6 Interpretation: a causal effect and a value-weighted error {#sec-interpretation}

Two readings make precise what CPG measures and why it tracks decision quality rather than reconstruction quality.

**CPG is a causal effect, identified by design.** Treat each episode's initial state $s_0 \sim \rho_0$ as a unit and the dynamics handed to the planner as a binary treatment $T \in \{D^\star, D_\theta\}$. Writing $Y(\cdot) \in \{0, 1\}$ for the success outcome under each treatment, CPG is the average treatment effect $\mathbb{E}_{s_0 \sim \rho_0}[\, Y(D^\star) - Y(D_\theta) \,]$, taken relative to a fixed planner and to $\rho_0$. Identification is trivial rather than assumed: the framework runs **both** arms from the same $s_0$ and the same planner seed, so the counterfactual is never missing and there is nothing to confound. The substantive conditions are instead consistency / SUTVA — the treatment changes *only* the `dynamics` callable, with no shared RNG stream or global state leaking between the two arms — and that success is a deterministic function of $(s_0, \mathrm{seed}, T)$, so the per-unit contrast $Y(D^\star) - Y(D_\theta)$ is observed exactly rather than only in expectation. The varied-init design makes this a matched-pairs estimate, for which the exact McNemar test reported in §3.5 is the matched-pairs significance test. The price of this clean identification is that the estimand is defined *relative to* the fixed planner and $\rho_0$ — not a planner-free property of the model — and that it requires an oracle dynamics, confining it to simulation.

**CPG is bounded by a value-weighted, not a reconstruction, error.** A fixed planner selects actions by ranking candidate action-sequences through a *score-to-go* computed by rolling out the dynamics. Per decision and to first order — setting aside the planner's own sampling noise and the way an error at step $t$ reshapes the state distribution visited at $t+1$ (which the expectation below is meant to absorb) — only model errors that change that ranking change the chosen action, and only changed actions change success. Heuristically, then,

$$
\lvert \mathrm{CPG} \rvert \;\lesssim\; C \cdot \mathbb{E}_{(s, a) \sim \rho^{\pi}}\!\left[\, \bigl\lvert Q_{\star}(s, a) - \hat{Q}(s, a) \bigr\rvert \,\right],
$$

where $Q_{\star}(s, a)$ and $\hat{Q}(s, a)$ are the planner's score-to-go for candidate $a$ at decision state $s$, computed by rolling out $D^\star$ and $D_\theta$ respectively; $\rho^{\pi}$ is the state–candidate distribution the fixed planner evaluates (itself dynamics-dependent — we elide that mismatch, the same effect as the self-correction of §4.4); and $C$ absorbs the horizon, the success-threshold margin, and the action-gap between competing candidates. This is an *informal* relationship in the spirit of value-error-propagation bounds, not a theorem proved here. Its content is qualitative but load-bearing: the right-hand side is the model's error **as seen through the planner's score functional** — a decision-aware, value-weighted error in the sense of the value-equivalence principle [Grimm et al., 2020; Grimm et al., 2021] and value-aware model learning [Farahmand et al., 2017], **not** a reconstruction error. This is exactly why the $\sim\!150\times$ drop in held-out prediction loss leaves CPG unmoved in §4.4: that error is orthogonal to $Q$. It also predicts the converse — a small *average* reconstruction error can still open a large gap when it concentrates where it flips the planner's ranking — which is the regime CPG is built to catch and a bare prediction metric cannot.

## 4. Empirical Study {#sec-empirical}

### 4.1 Setup

We instantiate the contract on three DeepMind Control Suite tasks [Tassa et al., 2018; Tunyasuvunakool et al., 2020]: Acrobot-swingup (a two-link underactuated pendulum that must build energy and balance upright), Cartpole-swingup (one actuated DoF), and Reacher-easy (a $2$-DOF arm reaching a per-episode target — the first two-dimensional action, discretised to a $3 \times 3$ torque grid). Acrobot's continuous torque is discretised to a five-level set $\{-1, -0.5, 0, +0.5, +1\}$. Success at step $t$ is the DMC dense reward clearing a task threshold. Two planners are used: random-shooting MPC ($N_\mathrm{cand} = 50$, horizon $H = 15$, $750$ dynamics evaluations per plan call) and a Cross-Entropy Method planner of comparable compute ($3 \times 24 \times 15 = 1{,}080$ evaluations). Episodes run for at most $T = 500$ steps.

**Sampling the task distribution.** Each episode draws a fresh initial state (and, for Reacher, target), and the two CPG arms share the per-episode env seed by index so the comparison is *paired*: episode $k$ starts from the same state in both arms. We pool three seeds. This is the design the metric's own honesty discipline demands — a single fixed initial state estimates only $P(\text{success} \mid \text{that state})$, not the task. §4.4 shows why it matters.

### 4.2 Oracle dynamics

We construct the oracle by instantiating a private `dm_control` environment inside a $(\mathrm{state}, \mathrm{action}) \to \mathrm{state}$ callable. Each call reconstructs $(q_\mathrm{pos}, q_\mathrm{vel})$ from the flat observation via $\mathrm{atan2}$, writes them into the private env's MuJoCo physics, calls `physics.forward`, steps once with the candidate torque, and returns the new flat observation. We verify the oracle reproduces `env.step` to numerical precision ($|\Delta| < 10^{-5}$) over a fifty-step random-policy rollout; this regression test is part of CI.

### 4.3 Learned dynamics

We train a Markovian MLP (two hidden layers of width $64$, ReLU activations) on $2\,000$ random-policy transitions collected from ten episodes of $200$ steps each. The input concatenates the six-dimensional observation with a five-dimensional one-hot action; the output predicts the next observation. Training: $200$ epochs with Adam (lr $10^{-3}$), batch size $256$, $10\%$ held-out validation split at the transition level. Final validation MSE $0.026$.

### 4.4 The metric as self-correction: a fixed-initial-state artifact {#sec-selfcorrection}

A first version of this study evaluated every arm at a *single fixed initial state* — the consequence of a deterministic env reset (`task_kwargs={"random": 0}`) and a fresh env per episode, so all episodes, across all seeds, began from the same configuration and only the planner's internal randomness varied. On Acrobot that fixed start happens to be an unusually easy swing-up. The oracle planner solved $30\%$ of episodes under random-shooting MPC and, under CEM, $88\%$ pooled across three seeds, while the learned arm stayed at $0$. The gap looked like a textbook `MODEL BOTTLENECK`: $\mathrm{CPG} = +0.30$ (random-shooting) rising to $+0.88$ (CEM, AC CI $[+0.81, +0.92]$).

One observation already hinted that the model was not the real story. Sweeping the MLP's training-set size across nearly three orders of magnitude ($200 \to 20{,}000$ transitions) dropped its held-out validation MSE by $\sim\!150\times$ while leaving the gap, its CI, and the verdict *exactly unchanged*: prediction and decision quality were fully dissociated — the value-(in)equivalence phenomenon that Grimm et al. (2020) and Farahmand et al. (2017) formalise. Better prediction bought no planning success, which is consistent with a model bottleneck but *equally* consistent with a planner that cannot exploit any model from this state.

Sampling the task's initial-state distribution settles it. Drawing a fresh initial state per episode (paired across arms) and pooling three seeds, the oracle's success rate *collapses*: from $0.30$ to $0.10$ at $n = 10$ under random-shooting, and from $0.88$ to $0.033$ pooled at $n = 150$ under CEM. With the oracle planner itself solving only $\sim\!3\%$ of random starts, the gap closes ($\mathrm{CPG} = +0.013$ and $+0.007$ for the two learned-dynamics families, both AC intervals straddling zero) and the verdict turns to `PLANNER BOTTLENECK`: even a perfect model would not help, because the search cannot solve the task from a typical start. The fixed-start `MODEL BOTTLENECK` was an artifact of an easy initial condition.

<table class="paper-table">
  <thead>
    <tr><th>Initial state</th><th>Planner</th><th>Dynamics</th><th>Oracle</th><th>Learned</th><th>CPG (AC 95% CI), verdict</th></tr>
  </thead>
  <tbody>
    <tr><td>fixed, $n=10$</td><td>random-shoot</td><td>MLP</td><td>0.30</td><td>0.00</td><td>+0.30 [-0.06, +0.56], <span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span></td></tr>
    <tr><td>fixed, pooled 150</td><td>CEM</td><td>TD-MPC2</td><td>0.88</td><td>0.00</td><td>+0.88 [+0.81, +0.92], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td><strong>task</strong>, $n=10$</td><td>random-shoot</td><td>MLP</td><td>0.10</td><td>0.10</td><td>+0.00 [-0.30, +0.30], <span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span></td></tr>
    <tr><td><strong>task</strong>, pooled 150</td><td>CEM</td><td>MLP</td><td>0.033</td><td>0.020</td><td>+0.013 [-0.027, +0.053], <span class="verdict-pill verdict-planner-bottleneck">PLANNER BOTTLENECK</span></td></tr>
    <tr><td><strong>task</strong>, pooled 150</td><td>CEM</td><td>TD-MPC2</td><td>0.033</td><td>0.027</td><td>+0.007 [-0.035, +0.049], <span class="verdict-pill verdict-planner-bottleneck">PLANNER BOTTLENECK</span></td></tr>
  </tbody>
</table>

Acrobot-swingup: the same arms at a fixed initial state vs. sampled over the task's initial-state distribution. The fixed start is an easy swing-up; sampling the task collapses the oracle planner to $\sim\!3\%$ and flips the verdict from `MODEL BOTTLENECK` to `PLANNER BOTTLENECK`. Fixed-init numbers are the v0.17 release; task-level from [`results/dmc_acrobot/{cpg,cem_cpg_sweep}.json`](https://github.com/Denis-hamon/world-model-eval-lab/tree/main/results/dmc_acrobot) (`varied_init: true`).

This is the strongest single piece of evidence for the metric's purpose: a calibrated, interval-gated statistic, run honestly over the task distribution, overturned a headline that a point estimate at one configuration would have published. The Acrobot `PLANNER BOTTLENECK` is robust to the obvious upgrades — swapping the bespoke MLP for a published TD-MPC2 world model and swapping random-shooting for CEM does not rescue either arm, since with the oracle planner itself solving only $\sim\!3\%$ of random starts every cell is `INCONCLUSIVE` at $n = 10$ and `PLANNER BOTTLENECK` when pooled. The remaining environments are reported at the task level throughout.

### 4.5 Cartpole-swingup: model bottlenecks, an inconclusive cell, and a learned model that beats the oracle {#sec-crossenv}

We replay the four-arm setup on DMC Cartpole-swingup (one actuated DoF), sampling the task distribution, at two TD-MPC2 capacities (`model_size` $\in \{1, 5\}$, $1 \times 10^{6}$ env steps each). Three seeds pooled, $n = 30$ per arm. The TD-MPC2 checkpoints were trained for this re-run; we therefore read the size-$5$ and size-$1$ blocks as two independent task-level snapshots, not as a clean capacity ablation (a fixed-vs-varied or size-vs-size contrast would confound the initial-state change with the separate trainings). Even so, the per-block verdicts are striking: the gate fires *three different branches* on this one environment.

<table class="paper-table">
  <thead>
    <tr><th>Planner</th><th>Dynamics</th><th>Oracle</th><th>Learned</th><th>Raw CPG</th><th>AC 95% CI, verdict</th></tr>
  </thead>
  <tbody>
    <tr><td>Random-shooting</td><td>MLP on TD-MPC2 data</td><td>0.933</td><td>0.000</td><td>+0.933</td><td>[+0.76, +0.99], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>Random-shooting</td><td>TD-MPC2</td><td>0.933</td><td>0.767</td><td>+0.167</td><td>[-0.03, +0.34], <span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span></td></tr>
    <tr><td>CEM</td><td>MLP on TD-MPC2 data</td><td>0.467</td><td>0.000</td><td>+0.467</td><td>[+0.25, +0.62], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>CEM</td><td>TD-MPC2</td><td>0.467</td><td>0.733</td><td>−0.267</td><td>[-0.48, -0.02], <span class="verdict-pill verdict-learned-outperforms">LEARNED OUTPERFORMS ORACLE</span></td></tr>
  </tbody>
</table>

DMC Cartpole-swingup, TD-MPC2 `model_size = 5`, task-level (three seeds, $n = 30$ pooled). Three verdict branches fire: `MODEL BOTTLENECK` (MLP arms), `INCONCLUSIVE` (RS × TD-MPC2), and `LEARNED OUTPERFORMS ORACLE` (CEM × TD-MPC2). For the last cell the paired bootstrap CI is $[-0.50, -0.03]$ (the $2\times2$ table is both $=10$, oracle-only $=4$, learned-only $=12$, neither $=4$), so both intervals clear zero, though the upper bound is near zero at $n = 30$. Numbers from [`results/dmc_cartpole/*_size5_pooled.json`](https://github.com/Denis-hamon/world-model-eval-lab/tree/main/results/dmc_cartpole).

<table class="paper-table">
  <thead>
    <tr><th>Planner</th><th>Dynamics</th><th>Oracle</th><th>Learned</th><th>Raw CPG</th><th>AC 95% CI, verdict</th></tr>
  </thead>
  <tbody>
    <tr><td>Random-shooting</td><td>MLP on TD-MPC2 data</td><td>0.933</td><td>0.067</td><td>+0.867</td><td>[+0.67, +0.96], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>Random-shooting</td><td>TD-MPC2</td><td>0.933</td><td>0.300</td><td>+0.633</td><td>[+0.40, +0.78], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>CEM</td><td>MLP on TD-MPC2 data</td><td>0.467</td><td>0.067</td><td>+0.400</td><td>[+0.18, +0.58], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>CEM</td><td>TD-MPC2</td><td>0.467</td><td>0.367</td><td>+0.100</td><td>[-0.15, +0.34], <span class="verdict-pill verdict-inconclusive">INCONCLUSIVE</span></td></tr>
  </tbody>
</table>

DMC Cartpole-swingup at `model_size = 1`, task-level (same protocol). Three cells are `MODEL BOTTLENECK`; the CEM × TD-MPC2 cell is `INCONCLUSIVE` ($+0.100$, CI crosses zero). Numbers from [`results/dmc_cartpole/*_pooled.json`](https://github.com/Denis-hamon/world-model-eval-lab/tree/main/results/dmc_cartpole).

The headline cell is CEM × TD-MPC2 at `model_size = 5`: the learned model lets the CEM planner solve $0.733$ of episodes against the oracle planner's $0.467$, so $\mathrm{CPG} = -0.267$ and the verdict is `LEARNED OUTPERFORMS ORACLE`. The AC interval $[-0.48, -0.02]$ and a paired bootstrap $[-0.50, -0.03]$ (the varied-init arms are paired by initial state) both clear zero. We flag two limitations rather than over-sell it: $n = 30$ leaves the upper bound close to zero, and the result rests on a single trained checkpoint; this is the cell most warranting a pooled follow-up. The mechanism is plausible: CEM, a weaker planner here than random-shooting on Cartpole's oracle ($0.467$ vs $0.933$), exploits the smoother learned latent dynamics to find solutions it cannot find against the true dynamics in the same budget. The remaining Cartpole cells are `MODEL BOTTLENECK` (the MLP-on-TD-MPC2 arms, where the learned dynamics is clearly worse) or `INCONCLUSIVE` (RS × TD-MPC2 at size $5$, $+0.167$; CEM × TD-MPC2 at size $1$, $+0.100$ — both CIs straddle zero), so a single environment exercises three of the gate's branches.

### 4.6 Third environment: DMC Reacher-easy {#sec-reacher}

To anchor the `MODEL BOTTLENECK` pole of the heterogeneity, we replay the four-arm grid on DMC Reacher-easy: a $2$-DOF arm reaching a per-episode randomized target (the varied-init protocol genuinely re-randomizes the target each episode). It is the first environment with a **two-dimensional action** (discretised to a $3 \times 3 = 9$ torque grid), and the oracle reconstruction is *exact* rather than lossy (the observation exposes `qpos` and `qvel` directly; the target is recovered from the finger-to-target vector), verified to reproduce `env.step` to $< 10^{-16}$. TD-MPC2 at `model_size = 1`, $1 \times 10^{6}$ env steps, three seeds pooled to $n = 30$ per arm.

<table class="paper-table">
  <thead>
    <tr><th>Planner</th><th>Dynamics</th><th>Oracle</th><th>Learned</th><th>Raw CPG</th><th>AC 95% CI, verdict</th></tr>
  </thead>
  <tbody>
    <tr><td>Random-shooting</td><td>MLP on TD-MPC2 data</td><td>1.000</td><td>0.700</td><td>+0.300</td><td>[+0.11, +0.45], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>Random-shooting</td><td>TD-MPC2</td><td>1.000</td><td>0.800</td><td>+0.200</td><td>[+0.03, +0.34], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>CEM</td><td>MLP on TD-MPC2 data</td><td>1.000</td><td>0.667</td><td>+0.333</td><td>[+0.14, +0.49], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
    <tr><td>CEM</td><td>TD-MPC2</td><td>1.000</td><td>0.767</td><td>+0.233</td><td>[+0.06, +0.38], <span class="verdict-pill verdict-model-bottleneck">MODEL BOTTLENECK</span></td></tr>
  </tbody>
</table>

DMC Reacher-easy, task-level (three seeds, $n = 30$ pooled, TD-MPC2 `model_size = 1`). The oracle solves the reach in every cell ($1.000$); both learned arms are clearly non-zero ($0.667$–$0.800$); all four cells are `MODEL BOTTLENECK` on non-degenerate gaps. Numbers from [`results/dmc_reacher/*_pooled.json`](https://github.com/Denis-hamon/world-model-eval-lab/tree/main/results/dmc_reacher).

Reacher is the clean `MODEL BOTTLENECK` case: the oracle solves the reach perfectly ($1.000$), *both* learned arms are clearly non-zero ($0.667$–$0.800$), and yet every cell is `MODEL BOTTLENECK` — not because a learned arm is pinned at zero, but because the AC lower bound on a genuine, non-degenerate gap ($+0.20$ to $+0.33$) is strictly positive. The verdict here tracks gap *magnitude*, not just presence: it ranks the learned dynamics by how much planning success they forfeit relative to a perfect model, while the metric's gate stays well clear of zero. Together the three environments place the gate at three different verdicts — `PLANNER BOTTLENECK` (Acrobot), a mix dominated by `MODEL BOTTLENECK` with a `LEARNED OUTPERFORMS ORACLE` cell (Cartpole), and uniform `MODEL BOTTLENECK` (Reacher).

### 4.7 Power analysis: how many episodes before a ranking is trustworthy {#sec-power}

The verdict gate turns from a post-hoc label into a planning tool once it is read forward: given hypothesised success rates and a target precision, how many episodes per arm does the AC interval need? Because the half-width depends only on the rates and $n$ (§3.2), this is pure arithmetic, computable before any rollout. The table reports the per-arm $n$ needed to reach a half-width of $0.05$ (an interval of $\pm 5$ percentage points) for a range of oracle rates with the learned arm at $0$ — the degenerate regime several cells (e.g. the MLP-dynamics arms) occupy.

<table class="paper-table">
  <thead>
    <tr><th>Oracle rate</th><th>$n$ for hw &le; 0.10</th><th>$n$ for hw &le; 0.05</th><th>$n$ for hw &le; 0.02</th></tr>
  </thead>
  <tbody>
    <tr><td>0.30</td><td>84</td><td>327</td><td>2021</td></tr>
    <tr><td>0.50</td><td>98</td><td>387</td><td>2403</td></tr>
    <tr><td>0.70</td><td>84</td><td>327</td><td>2021</td></tr>
    <tr><td>0.90</td><td>46</td><td>153</td><td>881</td></tr>
  </tbody>
</table>

Per-arm episode count needed to reach a target Agresti--Caffo half-width (learned arm at $0$, $z = 1.96$). Reproduced by `python -m experiments.power_analysis`; numbers from [`results/power_analysis.json`](https://github.com/Denis-hamon/world-model-eval-lab/blob/main/results/power_analysis.json).

The same arithmetic audits a point-estimate leaderboard. Consider two systems reported at success rates $0.94$ and $0.92$ over $n = 100$ episodes per arm with no interval — a plausible top-of-table near-tie. The AC interval on that difference has half-width $0.074$ and *straddles zero*: the reported ranking gap is statistically indistinguishable from noise at the sample size used, and separating the two to a half-width of $0.05$ would require $n = 209$ per arm. A wider gap ($0.94$ vs $0.78$) is decidable at the same $n = 100$ ($\mathrm{hw} = 0.095$, interval clears zero); a mid-table near-tie ($0.78$ vs $0.75$) is not. This is the knowledge a leaderboard cannot carry: not which number is larger, but whether the ordering survives its own sample size. A reader holding only point estimates cannot tell a real ranking from a coin flip; the power table makes the distinction mechanical.

A traced decision closes the loop. The Cartpole `model_size = 1`, CEM, TD-MPC2 cell returned `INCONCLUSIVE` at $n = 30$ ($\mathrm{CPG} = +0.100$, half-width $0.24$), as did the size-$5$ `LEARNED OUTPERFORMS ORACLE` cell whose interval barely clears zero. The table prescribes the per-arm count that would shrink each interval to a target precision and let the gate commit (or recant); reporting that number, rather than silently presenting an under-powered verdict as if it were decisive, is what *decision-grade* is meant to denote.

## 5. Discussion and Limitations

The empirical study spans three DMC control tasks (Acrobot-swingup, Cartpole-swingup, Reacher-easy), two planners (random-shooting and CEM), and two learned-dynamics families (a bespoke MLP and TD-MPC2), each evaluated over the task's initial-state distribution with the two arms paired by initial state and three seeds pooled. The headline is not a single number but a *pattern*: across the three tasks the gate commits to four of its five branches — `PLANNER BOTTLENECK` (Acrobot), `MODEL BOTTLENECK` (Reacher, most Cartpole cells), `LEARNED OUTPERFORMS ORACLE` (high-capacity Cartpole under CEM), and `INCONCLUSIVE` (several moderate-$n$ near-ties). The study is deliberately scoped to simulated, fully-observed control where an oracle dynamics is available; on hardware-in-the-loop or physical environments the metric is undefined, and surrogate-oracle variants (e.g. a higher-fidelity model or a domain-randomised reference in the spirit of Tobin et al., 2017) are future work. The adapter interface (§2) is what makes this breadth cheap: each new cell is one more `dynamics` callable or planner constructor passed into the same `BenchmarkRunner`.

**Properties and threats to validity.** First, **CPG's magnitude is relative to both the oracle planner and the evaluation distribution, not an absolute property of the model.** The self-correction of §4.4 is the sharpest case: holding the model fixed, the Acrobot gap moves from $+0.88$ to $\approx 0$ purely by replacing one fixed initial state with a sample of the task distribution. Read CPG relative to a *stated planner and a stated initial-state distribution*; the verdict gate is more robust than the point estimate because it only asks whether the interval clears zero. Second, **the load-bearing evidence is where the verdict changes — on conditions or on data**, not the degenerate cells where a learned arm sits at $0/n$ and any estimator agrees the oracle wins. The two genuinely informative behaviours are the self-correction (a verdict overturned by sampling the task) and the Cartpole `LEARNED OUTPERFORMS ORACLE` cell; a gate that flips on evidence and conditions is what a point-estimate leaderboard cannot reproduce. Third, **the design is paired and we report it as such.** Because the two arms share each episode's initial state under varied-init sampling, the non-degenerate cells (notably the `LEARNED OUTPERFORMS ORACLE` cell) are reported with a paired bootstrap CI alongside the Agresti--Caffo interval; AC remains the right tool for boundary ($0/n$) cells. Fourth, **the metric requires an oracle dynamics callable and is confined to simulation**; our Acrobot oracle reconstructs $(q_\mathrm{pos}, q_\mathrm{vel})$ lossily modulo $2\pi$ and re-steps MuJoCo, while Reacher's is exact.

**Sample size and a confound.** The Cartpole and Reacher cells are pooled to $n = 30$; this is thin for the `LEARNED OUTPERFORMS ORACLE` cell, whose upper bound is near zero, and that cell most warrants a pooled-$150$ follow-up (the gate doubles as the tool that sizes it, §4.7). The Cartpole `model_size = 5` and `= 1` blocks use *separately trained* checkpoints, so we read them as two task-level snapshots rather than a clean capacity ablation — a size-vs-size or fixed-vs-varied contrast there would confound the comparison with the distinct trainings. The clean fixed-vs-varied ablation is carried by Acrobot (§4.4), where the checkpoint is held fixed.

**Relation to evaluation platforms.** Concurrent and complementary platforms such as stable-worldmodel ([Maes et al., 2026](https://arxiv.org/abs/2605.21800)) standardise data, baselines, planners, and broad environment suites, reporting point-estimate success rate as the primary control metric. CPG and the `wmel` framework were developed independently and concurrently (the public repository's version history predates and does not derive from that platform). The contributions are orthogonal: CPG is a single decision-grade statistic with a calibrated interval and a decision rule, not a platform, and it composes with any benchmark runner that can swap the dynamics callable — a platform of that kind could compute CPG per model per environment to separate the model's contribution from the planner's.

Other limitations worth flagging explicitly. The discrete-torque action space is a five-level approximation of the continuous control problem; a continuous variant of CEM over a Gaussian per-timestep mean is a straightforward extension. The Acrobot success criterion ($r_t \geq 0.6$) is a binary projection of a dense reward; a continuous variant of CPG that reports the difference of mean returns is straightforward and may be more sample-efficient. The planner score is task-specific and tightly coupled to the environment's reward geometry; a learned score function (predicting the DMC reward directly from the latent state) would remove this coupling, at the cost of a second model component.

## 6. Conclusion

We have proposed the Counterfactual Planning Gap — the success-rate difference between a fixed planner using oracle versus learned dynamics, reported with an Agresti--Caffo interval, a paired bootstrap where the design is paired, and a five-branch verdict gated on the CI lower bound — and packaged it behind a minimal evaluation contract. Exercised over the task's initial-state distribution on three DMC tasks, the verdict is *heterogeneous*: Acrobot is `PLANNER BOTTLENECK` (the oracle planner itself solves only $\sim\!3\%$ of random starts), Reacher is uniformly `MODEL BOTTLENECK` on non-degenerate gaps ($+0.20$ to $+0.33$), and high-capacity Cartpole under CEM reaches `LEARNED OUTPERFORMS ORACLE` ($\mathrm{CPG} = -0.27$; the AC and paired-bootstrap intervals both clear zero), with several cells `INCONCLUSIVE`. The sharpest evidence for the metric's purpose is a self-correction: an earlier fixed-initial-state evaluation of this same framework reported a large `MODEL BOTTLENECK` gap on Acrobot, and re-running over the task distribution — the design the metric's own honesty discipline demands — collapsed the oracle and overturned the verdict to `PLANNER BOTTLENECK`. A calibrated, interval-gated statistic caught a configuration-sensitive artifact that a point estimate would have published. And because the gate is a function of the confidence interval, it doubles as a power tool that sizes a comparison before the rollouts and audits a leaderboard near-tie after them. We argue this kind of calibrated honesty — a metric that refuses to claim more than the data and the evaluation conditions support, and that says so out loud — is the right design target for a methodology that wants to sit usefully next to a fast-moving model literature.

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
- D'Oro, Metelli, Tirinzoni, Papini, Restelli (2020). *Gradient-Aware Model-Based Policy Search.* AAAI.
- Farahmand, Barreto, Nikovski (2017). *Value-Aware Loss Function for Model-Based Reinforcement Learning.* AISTATS.
- Farahmand (2018). *Iterative Value-Aware Model Learning.* NeurIPS.
- Grimm, Barreto, Singh, Silver (2020). *The Value Equivalence Principle for Model-Based Reinforcement Learning.* NeurIPS.
- Grimm, Barreto, Farquhar, Silver, Singh (2021). *Proper Value Equivalence.* NeurIPS.
- Ha, Schmidhuber (2018). *World Models.* arXiv:1803.10122.
- Hafner et al. (2019). *PlaNet — Learning Latent Dynamics for Planning from Pixels.* ICML.
- Hafner et al. (2020). *Dreamer — Dream to Control: Learning Behaviors by Latent Imagination.* ICLR.
- Hafner et al. (2023). *Dreamer-V3 — Mastering Diverse Domains through World Models.* arXiv:2301.04104.
- Hansen et al. (2024). *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR.
- Henderson et al. (2018). *Deep Reinforcement Learning that Matters.* AAAI.
- Kidambi, Rajeswaran, Netrapalli, Joachims (2020). *MOReL: Model-Based Offline Reinforcement Learning.* NeurIPS.
- LeCun (2022). *A Path Towards Autonomous Machine Intelligence.* Open Review.
- Liu et al. (2023). *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.*
- Maes, Le Lidec, Facury, Massaudi, Chaurasia, Capuano, Gao, Gillin, Haramati, Scieur, LeCun, Balestriero (2026). *stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation.* arXiv:2605.21800.
- Micheli, Alonso, Fleuret (2023). *Transformers are Sample-Efficient World Models (IRIS).* ICLR.
- Newcombe (1998). *Interval Estimation for the Difference Between Independent Proportions: Comparison of Eleven Methods.* Statistics in Medicine.
- Park et al. (2024). *OGBench: Benchmarking Offline Goal-Conditioned RL.*
- Schäfer, Udluft, Zimmermann (2007). *The Recurrent Control Neural Network.* Engineering Applications of Artificial Intelligence.
- Schrittwieser et al. (2020). *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero).* Nature.
- Tassa et al. (2018). *DeepMind Control Suite.*
- Tobin, Fong, Ray, Schneider, Zaremba, Abbeel (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS.
- Tunyasuvunakool et al. (2020). *dm_control: Software and Tasks for Continuous Control.* Software Impacts.
- Wilson (1927). *Probable Inference, the Law of Succession, and Statistical Inference.* JASA.
- Yu, Thomas, Yu, Ermon, Zou, Levine, Finn, Ma (2020). *MOPO: Model-Based Offline Policy Optimization.* NeurIPS.
