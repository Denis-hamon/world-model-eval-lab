# RoboLab-120 leaderboard: a calibration audit

A no-GPU reading of a published generalist-policy leaderboard through this
framework's honesty layer. A point-estimate ranking invites the question its
numbers leave open: **at the reported sample size, which pairwise orderings are
statistically resolved, and how many episodes would the close ones need?**

```bash
python -m experiments.robolab_audit.audit
```

Writes `results/robolab_audit/leaderboard_audit.json`.

## What it does

Takes the published RoboLab-120 *overall* success rates (arXiv:2604.09860 --
five policies, `N=10` episodes per task over 120 tasks, so `n=1200` trials per
model) and, for every pair of models, computes the Agresti--Caffo 95% interval
on the difference of success rates and asks whether it clears zero. The
machinery is the same `wmel.metrics` power-analysis used for CPG
(`ac_ci_half_width`, `detectable_gap_at_n`); nothing here is run or trained.

RoboLab-120 is the simulation benchmark on which the open Cosmos 3 DROID policy
also reports task success rates, so this audit is the lens through which a later
paired re-run -- running policies ourselves on shared initial states and ranking
them with `paired_bradley_terry_ranking` -- should be read.

## What it finds

The headline ordering is robust: the top model is separated from the rest by
intervals that clear zero by a wide margin. But the ranking is **not fully
resolved at the reported sample size** -- the closest tail pair's small overall
gap does not clear zero at `n=1200`, and the audit reports the per-arm episode
count that would actually separate it (using the verdict gate itself, not a
half-width proxy). Adding the interval to each pairwise gap is the whole point:
it distinguishes the orderings the data supports from the ones it does not.

## Scope and a caveat, stated honestly

- **Overall only.** The paper also reports per-competency-axis rates (visual /
  procedural / relational), but its tasks are *multi-labeled* across
  competencies, so the per-axis trial counts are unequal and not a clean
  120/3 split. Rather than guess them, the per-axis audit is deferred to the
  paired re-run (Stage 1), where the episode count per cell is controlled
  directly.
- **The overall figure is not i.i.d.** The 1200 trials span heterogeneous
  tasks, not i.i.d. Bernoulli draws of a single success probability. Treating
  them as i.i.d. for the interval is the *optimistic* case: task-level
  clustering only widens the true interval, so every pair this audit calls
  *within noise* stays within noise under the proper clustered analysis. The
  result is a conservative lower bound on how much of the ordering is noise.

## Source

RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist
Policies, [arXiv:2604.09860](https://arxiv.org/abs/2604.09860). Overall success
rates and the N=10 / 120-task protocol transcribed from the main results table
and evaluation-protocol section; verify against the source before reuse.
