"""Per-episode environment seeding for task-distribution sampling.

By default the DMC env adapters load with ``task_kwargs={"random": 0}``, and
the :class:`~wmel.benchmark_runner.BenchmarkRunner` builds a fresh env per
episode, so *every* episode starts from the same initial state (and, for
Reacher, the same target geom). Under that default a reported success rate
estimates ``P(success | one fixed configuration, planner noise)`` -- not the
task's initial-state distribution -- and the two CPG arms are coupled only
through shared planner seeding, not through shared per-episode start states.

``episode_varied_factory`` returns a zero-arg env factory that assigns each
constructed env a distinct, reproducible task seed derived from ``base_seed``.
Two factories built with the *same* ``base_seed`` emit the *same* seed
sequence, so the oracle and learned arms see identical per-episode initial
states: a paired design that samples the task distribution. Pooling several
``base_seed`` values then yields disjoint sets of task instances.

This is opt-in: drivers default to the historical fixed-init factory so that
committed results remain reproducible. Pass ``--varied-init`` (and re-run) to
sample the task distribution instead.
"""

from __future__ import annotations

import itertools
from typing import Callable

# Stride between base seeds. Each arm draws episode seeds
# ``base_seed * _SEED_STRIDE + episode_index``; the stride must exceed the
# largest per-arm episode count so two base seeds never collide.
_SEED_STRIDE = 100_000


def episode_varied_factory(
    env_ctor: Callable,
    base_seed: int,
    **env_kwargs,
) -> Callable[[], object]:
    """Return a zero-arg factory yielding envs with per-episode task seeds.

    Parameters
    ----------
    env_ctor
        An env class/callable accepting a ``task_kwargs`` keyword (all DMC
        adapters in ``wmel.envs`` do).
    base_seed
        Per-arm base seed. The oracle and learned arms of one CPG comparison
        must share the same ``base_seed`` to stay paired (identical
        per-episode initial states); distinct pooled seeds must use distinct
        ``base_seed`` values.
    **env_kwargs
        Forwarded to ``env_ctor`` (e.g. ``discrete_levels=`` for Cartpole).
    """
    counter = itertools.count()

    def factory():
        episode_index = next(counter)
        task_seed = base_seed * _SEED_STRIDE + episode_index
        return env_ctor(task_kwargs={"random": task_seed}, **env_kwargs)

    return factory


# Training-data collection must draw from a DISJOINT block of initial states
# from evaluation, so the model is never trained on the exact start states it
# is scored on. Eval uses base_seed = `seed`; training uses
# base_seed = `seed + _TRAIN_BASE_OFFSET`. With _SEED_STRIDE = 100_000 and
# pooled seeds in {0, 1, 2}, eval task seeds live in [0, 3 * 100_000) while
# training lives at >= 10_000 * 100_000, so the two never overlap.
_TRAIN_BASE_OFFSET = 10_000


def eval_varied_factory(env_ctor: Callable, seed: int, **env_kwargs):
    """Per-episode-varying factory for an evaluation arm (paired across arms
    sharing the same ``seed``)."""
    return episode_varied_factory(env_ctor, base_seed=seed, **env_kwargs)


def train_varied_factory(env_ctor: Callable, seed: int, **env_kwargs):
    """Per-episode-varying factory for training-data collection, drawing from
    a block of initial states disjoint from :func:`eval_varied_factory`."""
    return episode_varied_factory(env_ctor, base_seed=_TRAIN_BASE_OFFSET + seed, **env_kwargs)
