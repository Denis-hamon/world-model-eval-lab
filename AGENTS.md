# AGENTS.md

Project-specific instructions for AI coding agents (Codex, Claude Code, etc.) working in this repository.

## Mission

This repository is a **research-to-product** exploration of evaluation for action-conditioned world models. The product wedge is the **evaluation layer**, not the model itself. Every change should sharpen that wedge.

## Hard rules

1. **No affiliation claims.** Do not state or imply this project is from AMI, Meta, Yann LeCun, the LeWorldModel authors, or any of their organizations. References to JEPA-style or LeWorldModel ideas are conceptual only.
2. **No reimplementation of LeWorldModel** or any specific proprietary world model. Adapters define a contract; they do not ship someone else's weights or code.
3. **No heavyweight ML dependencies.** No `torch`, `tensorflow`, `jax`, `transformers`, `gymnasium`, `mujoco`, etc., in core code unless explicitly requested by a human. The runtime must stay stdlib-only.
4. **No datasets, no checkpoints.** Do not download or embed model weights or training data.
5. **CPU-only.** Everything must run on a laptop without a GPU.

## Soft rules

- Keep modules small and readable. Use `dataclasses` and `typing`.
- Prefer clear, boring Python over clever Python.
- Prefer extending existing interfaces in `src/wmel/adapters/base.py` over adding parallel ones.
- Every new metric must have: a definition, why it matters, an example measurement, and a test on synthetic data.
- Every new benchmark card must answer a **product question**, not just a technical one.
- Docs are public-facing. Treat them like a landing page, not a notebook.
- Runnable examples beat narrative claims. If a doc page mentions a capability, there should be code that demonstrates it.

## When in doubt

- If a change risks turning this into a training framework or a model zoo, stop and ask.
- If a change introduces a dependency, justify it in the PR description and consider whether a stub or a protocol would do.
- If a change makes the repo look like an official LeWorldModel artifact, revise the wording.

## Style

- Markdown: no emojis, no decorative headings.
- Python: 4-space indent, type hints on public functions, docstrings on public classes.
- Tests: `pytest`, no marker magic, deterministic seeds.
