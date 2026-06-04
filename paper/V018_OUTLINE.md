# v0.18 revision outline (heterogeneity + self-correction framing)

Planning doc for the v0.18 paper rewrite. NOT part of the paper. The result
numbers below are slotted in once the clean Cartpole init-only ablation lands
(see `experiments/RERUN_VARIED_INIT.md`); Acrobot and Reacher are already final
(clean init-only ablations, checkpoints reused). Delete this file at tag time.

## What changed since v0.17

The v0.17 worked examples were all evaluated at a single fixed initial state
(`task_kwargs={"random":0}`, fresh env per episode -> every episode, every
"seed", same start). The task-distribution re-run (#36, `--varied-init`)
inverts part of the headline:

| Env | v0.17 (fixed init) | v0.18 (task-level) | Clean ablation? |
|---|---|---|---|
| Acrobot (pooled n=150, CEM) | +0.88 MODEL BOTTLENECK | oracle ~0.033, gap ~+0.01 -> **PLANNER BOTTLENECK** | yes (ckpt reused) |
| Acrobot flagship (n=10, RS) | +0.30 INCONCLUSIVE | 0.10/0.10 -> +0.00 INCONCLUSIVE | yes |
| Cartpole size=5 CEM x TD-MPC2 | MODEL BOTTLENECK | gap -0.267, CI [-0.483,-0.017] -> **LEARNED OUTPERFORMS ORACLE** | PENDING clean re-run* |
| Reacher CEM (mlp / tdmpc2) | +0.667 / +0.367 MB | +0.333 / +0.233 -> **MODEL BOTTLENECK** (weaker) | yes |

\* Cartpole's #36 numbers used from-scratch retrained checkpoints (originals
deleted), so fixed-vs-varied there confounds init with checkpoint. The
`--out-suffix _fixedinit` ablation (PR #39) fixes this. Until it lands, treat
Cartpole numbers as provisional / report task-level only without the
fixed-vs-varied contrast.

## New framing: the metric caught its own authors' artifact

Thesis of v0.18 is no longer "MODEL BOTTLENECK across three envs." It is:

> A decision-grade metric is only as honest as the distribution it is measured
> over. Evaluated at a single fixed start, CPG reported a large model gap on
> Acrobot; sampling the task distribution, the same gap is **planner**-bound,
> not model-bound. Across three DMC tasks the gated verdict is **heterogeneous**
> -- PLANNER BOTTLENECK (Acrobot), LEARNED OUTPERFORMS ORACLE (Cartpole),
> MODEL BOTTLENECK (Reacher) -- and the metric's gate flips between four
> branches on real data. The headline is the self-correction: the framework's
> own calibrated-honesty machinery surfaced a config-sensitivity its first
> draft missed.

This **strengthens** the contribution and resolves the referee's main weakness
(the "verdict changes on evidence" property no longer rests on a single cell --
four verdict branches now fire across envs).

## Proposed title

Drop "Decoupling Model Error from Planner Capacity" (now doubly wrong: the
re-run shows the bottleneck is often NOT model error). Candidate:

> Counterfactual Planning Gap: A Gated, Interval-Reported Statistic for
> Diagnosing World-Model Bottlenecks under a Fixed Planner

## Section plan

1. **Intro** -- reorder contributions: lead with the gated decision rule (was
   contribution 3); fold the CPG definition into it as the operationalization
   of the known model-exploitation gap; demote the four-method contract to a
   paragraph. Add the self-correction as a contribution.
2. **Related work** -- DONE in prep: value-equivalence / decision-aware
   paragraph (Grimm, Farahmand, D'Oro); AC-vs-RLiable-bootstrap justification;
   Wang benchmarking; Tobin sim-to-real. Add the RLiable-analogy sentence to
   the swm paragraph (CPG : success-rate platforms :: RLiable : RL benchmarks).
3. **Contract + metric (§3)** -- mostly unchanged. Reconcile §3.1 "property of
   the model" with the planner-relative caveat: CPG is a per-(model, planner)
   quantity. Add the paired-design note (see Methods note below).
4. **Empirical (§5)** -- REWRITE to task-level. New structure: per-env
   subsections each reporting the task-level verdict; a cross-env synthesis
   table that is the heterogeneity headline. Keep the single-config numbers
   only as an explicit fixed-vs-varied contrast on Acrobot (the cleanest
   artifact demonstration).
5. **Power analysis (§ unchanged math)** -- keep; numbers re-derived at the new
   task-level rates where the gate lands near zero.
6. **Discussion** -- the self-correction narrative; threats-to-validity updated
   (paired design now genuinely paired under varied-init -> a paired bootstrap
   / McNemar becomes the right interval; report it for the non-degenerate
   cells, Acrobot pooled + Reacher + Cartpole).

## Methods note (now that the design is genuinely paired)

Under `--varied-init`, episode k starts from the same state in both arms
(shared base seed) -- a paired design. So the AC independent-proportions CI is
now mildly conservative for the non-degenerate cells, and a paired bootstrap /
McNemar CI is the right tool there. Per-episode outcomes are persisted in the
seed-level JSONs (`*_full/results`), so this is computable without re-running.
Do this in v0.18 for: Acrobot pooled, Reacher (both arms non-zero), Cartpole
size=5 (where LEARNED OUTPERFORMS -- a tighter paired CI matters for whether
the negative gap clears zero).

## Referee-fix checklist

- [x] Value-equivalence / decision-aware citations (prep, this branch).
- [x] AC-vs-RLiable-bootstrap justification (prep, this branch).
- [x] Wang benchmarking + Tobin sim-to-real citations (prep).
- [ ] Retitle to "...under a fixed planner" (full rewrite, for coherence).
- [ ] Reorder §1 contributions; demote the contract.
- [ ] Trim the abstract to ~200-250 words (currently ~840).
- [ ] Reconcile §3.1 vs the planner-relative caveat.
- [ ] Paired bootstrap / McNemar CI on the non-degenerate cells.
- [ ] Sync `docs/paper.md` (HTML mirror) Related Work + numbers; drop the
      status banner once numbers are integrated.
- [ ] Fix the stale repo version tags (v0.11/v0.12) leaking into prose.
- [ ] De-duplicate the recurring slogan sentences; standardize
      "learned-dynamics arm".
- [ ] Add a methods note that the re-run design is paired; report per-seed
      dispersion (the seed-level JSONs make this cheap).

## Gating

The only blocker is the clean Cartpole ablation (PR #39 command). Everything
else (Acrobot, Reacher, framing, citations, methods, abstract, title) can be
written now; Cartpole's cells + the cross-env synthesis table slot in last.
