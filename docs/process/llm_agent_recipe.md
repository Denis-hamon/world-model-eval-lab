---
prev:
  title: "05 - 30-day study plan"
  url: ../05_30_day_prototype_plan.html
next:
  title: "Back to home"
  url: /
---
# Recipe: executing the 30-day plan with an LLM coding agent

> This page is process meta-content - it documents how this repository was built. If you are here to read about world-model evaluation, the technical pages are [00 - Thesis](../00_thesis.html) through [06 - Reading a scorecard](../06_demo.html). This page is for someone who wants to use a similar recipe on a different project.

This repository was built end-to-end with an LLM coding agent in the loop (Claude Code in this case; Codex, Cursor, Aider, or any equivalent agent that can read files, run shell, edit code, and spawn sub-agents works the same way). The bottleneck was never "the model cannot code". It was **specification clarity** and **review discipline**.

The recipe below is reproducible. Each step has been used to ship the v0.3 → v0.7 cycle of this repo.

## 0. Setup

- Pick a coding agent that can read files, run shell, edit code, and spawn sub-agents.
- Make sure the agent has read access to `AGENTS.md` (hard rules) and `CONTRIBUTING.md` (style + extension procedure). Both files exist for the agent as much as for a human contributor.
- Set up a working directory and confine the agent to it. Do not let it modify `~` or system files.
- Decide upfront: which user actions you will run by hand (creating remote repos, force-pushes, releases) versus which the agent may take autonomously (file edits, commits, pushes to a feature branch).

## 1. Hand it the week, not the task

Each week of the plan is a stand-alone brief. Drop the entire week's section into the agent verbatim and ask:

> Plan the implementation. List every file you will create or modify. Do not write code yet. When you list a metric or a test, state the invariant it locks in.

Approve the plan only when it lists the same files you would. If the agent proposes "improvements" outside the week's scope, push back. **Scope creep is the failure mode**, not under-delivery.

## 2. Implementation in one round

Once the plan is approved:

> Implement the plan. Write tests alongside each new module. After every file is written, run `pytest -q` and any example scripts the README mentions. Report failures rather than working around them.

A good agent will use a visible todo list, run tests before claiming success, and surface unexpected behaviour rather than papering over it.

## 3. Pre-tag adversarial review (the step that paid off 4-for-4)

Before tagging any release, spawn an **independent** review agent with a fresh context. The independence is what makes this work; the agent does not inherit your implementation rationalisations.

Prompt template:

> Adversarially review the diff at HEAD against the brief above. The previous review caught [insert prior failure modes; for this repo it was per-call vs per-episode latency confusion, perturbation accounting overcounting the denominator, dead reporting paths, fragile test heuristics, missing invariants]. Look specifically for: math errors, doc/code mismatches, missing invariants, silent fallbacks, performance regressions. If you find nothing, say so explicitly. Output `severity (critical / major / minor) | one-line description | file:line evidence | suggested fix`.

Track record on this repo:

- **v0.3 review**: 2 real bugs (per-call latency, perturbation accounting) + 2 minors. Both bugs fixed in v0.3.1.
- **v0.4 review**: 2 majors (test docstring arithmetic wrong, missing compute column in sweep markdown) + 5 minors. All addressed before tag.
- **v0.5 review**: 6 minors, including a dead `Perturbation.name` in the reporting path and an O(n) `list.pop(0)` regression in the runner. All addressed before tag.
- **v0.7 review**: 1 critical (CLI imported `examples.*` which is not part of the installed package), 3 majors (schema version inconsistency, two-room CLI silently scored 0% without a waypoint, `--plan-horizons` parser crashed on bad input), 3 minors. All addressed before tag.

Zero releases shipped with metric-correctness bugs after this pattern was adopted. The review usually catches *one* finding that would have been embarrassing in public.

## 4. Tag, push, release

Only after review findings are addressed. Each tag corresponds to a working green-CI state. Release notes are drafted by the agent from the commit history plus the relevant doc-page updates, then read over by a human before publishing.

## 5. Soft passes (no version bump)

Documentation polishing, vocabulary tightening, rebranding, design-system changes, and Pages-only updates are **doc-only passes**. They do not need a version bump. They still deserve a single self-contained commit message that explains the why.

This repo has shipped several: the "soften framing" pass, the GitHub Pages landing, the IBM Plex design system, the interactive hero, the formula popovers. None bumped the version.

## What to keep human-in-the-loop

The agent should not autonomously decide:

- **Strategic scope**: ship X or cut X. Especially "should this be a side project or be folded into the day job?".
- **Vocabulary that signals intent**: a repo positioned as a "product wedge" reads very differently from one positioned as a "methodology study". The agent will not catch this for you unless you ask it to.
- **Tagging decisions** and the human-facing release narrative.
- **Going public**: making the repo public, posting on social media, sharing internally. The agent can prepare the artifact; the human owns the broadcast.

## Anti-patterns this recipe explicitly avoids

- Asking the agent to "improve" the repo without a concrete target.
- Accepting code without tests for the invariants that actually matter (i.e. tests that would fail if the metric were silently wrong, not tests that just exercise the happy path).
- Skipping the adversarial review because `pytest` is green. Green tests do not catch doc/code mismatches or numerical confusion in unmeasured directions.
- Letting the agent rationalise a doc/code mismatch instead of fixing one side. The doc is usually right; if it is wrong, fix the doc and re-test the code.
- Bumping the version on doc-only passes. Releases should mean a real surface change.
- Optimising for polish-on-the-Pages-site rather than for substance in the framework itself. The polish has a sharp diminishing return after the first investment; technical depth does not.

## Reading order if you are setting this recipe up on a new project

1. The technical pages [00 - Thesis](../00_thesis.html) → [06 - Reading a scorecard](../06_demo.html).
2. `AGENTS.md` (hard rules: scope, dependencies, non-affiliation, style).
3. `CONTRIBUTING.md` (how a new metric / benchmark card / adapter / perturbation should be structured).
4. One existing release commit message in `git log` (for example `v0.6.0` or `v0.7.0`) to calibrate the level of detail the agent should produce.

## How long does it actually take?

Per week, one focused sitting end-to-end (plan, implement, review, fix, commit, tag). Most of the wall-clock time is test runs and waiting on review cycles, not model think-time. The recipe favours **fewer, deeper cycles** over many shallow ones - exactly the discipline that catches metric-correctness bugs early.
