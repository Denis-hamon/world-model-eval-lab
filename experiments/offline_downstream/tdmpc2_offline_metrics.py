"""Offline metrics vs downstream CPG for the committed TD-MPC2 cells (CPU).

Stage 2 of the keystone, and its discriminating test. The maze proof of
concept (``maze_quality_sweep.py``) lives in a regime where prediction and
decision quality move together, so the naive one-step metric and the
decision-aware ranking metric correlate with downstream success about equally.
This script asks the sharper question on the real cells: across the committed
TD-MPC2 world models, does a *decision-aware* offline metric (M3) track the
downstream Counterfactual Planning Gap where a *naive prediction-error* metric
(M1/M2) does not?

For each committed TD-MPC2 cell we load the trained latent dynamics on CPU and
compute three planner-free offline metrics against the matched oracle dynamics,
on a fixed seeded sample of states, then pair them with that cell's already
committed downstream CPG:

  M1  one-step L2 error   : mean ||learned(s,a) - oracle(s,a)|| over the
                            sampled (state, action) pairs. The naive
                            reconstruction-style metric -- the foil.
  M2  k-step L2 divergence: mean L2 between learned and oracle open-loop
                            rollouts under one fixed action sequence
                            (compounding error).
  M3  action agreement    : mean Kendall tau between the oracle and learned
                            *ranking of actions by the env score* of their
                            predicted successor -- the decision-aware metric
                            (does the model rank actions the way the planner
                            needs?). Unitless, hence cross-env comparable.
  downstream gap          : the committed CPG (oracle minus learned planning
                            success), read from results/dmc_*/tdmpc2_cpg*.json.

Compute is CPU (TD-MPC2 inference defaults to CPU; the planner is not run
here -- only the dynamics callable). It is *checkpoint-gated*: the ``.pt``
files are gitignored and not in the repo. Run this where they exist (on the
training box, or after ``scp``-ing them into results/dmc_*/), having done the
TD-MPC2 setup (``scripts/setup_tdmpc2.sh`` + the deps in
``experiments/dmc_reacher/tdmpc2_cpg.py``). Cells whose checkpoint is absent
are reported as skipped, so the script still runs (and tells you what is
missing) on a checkout without the weights.

Writes results/offline_downstream/tdmpc2_offline_scores.json. Feed it to the
analysis with::

    python -m experiments.offline_downstream.correlate \\
        --bundle results/offline_downstream/tdmpc2_offline_scores.json \\
        --downstream gap

IMPORTANT (scale confound): M1/M2 are L2 distances in each env's own state
space, so their magnitudes are NOT comparable across environments; pooling
Cartpole and Reacher M1/M2 into one correlation is scale-confounded and would
unfairly handicap them against the unitless M3 (the same kind of apples-to-
oranges comparison the maze PoC review flagged). The fair head-to-head is
WITHIN an environment. This script therefore prints a within-env correlation
for every env with at least three cells; the pooled cross-env correlate.py
output is meaningful only for M3.

Usage:
    python -m experiments.offline_downstream.tdmpc2_offline_metrics
    python -m experiments.offline_downstream.tdmpc2_offline_metrics --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

# Stdlib-only at module scope so the pure metric helpers below are importable
# (and unit-testable with synthetic callables) on a checkout with no torch and
# no dm_control. wmel.metrics is stdlib. The heavy env/adapter/torch imports
# are deferred into the cell-evaluation functions.
from wmel.metrics import bootstrap_correlation_ci, kendall_tau  # noqa: E402
from wmel.report import report_envelope_metadata  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure, env-agnostic metric helpers (no torch, no env) -- unit-testable.       #
# A "dynamics" is any callable (state_tuple, action_tuple) -> next_state_tuple;#
# a "score" is any callable (state_tuple) -> float.                            #
# --------------------------------------------------------------------------- #

def _l2(a, b) -> float:
    if len(a) != len(b):
        raise ValueError(f"state length mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def mean_one_step_l2(oracle_dyn, learned_dyn, states, actions) -> float:
    """M1: mean L2 between learned and oracle one-step predictions over the
    Cartesian product of ``states`` x ``actions``."""
    if not states or not actions:
        raise ValueError("need at least one state and one action")
    total = 0.0
    n = 0
    for s in states:
        for a in actions:
            total += _l2(learned_dyn(s, a), oracle_dyn(s, a))
            n += 1
    return total / n


def mean_kstep_divergence(oracle_dyn, learned_dyn, states, action_seq) -> float:
    """M2: mean L2 between learned and oracle open-loop rollouts at the END of
    a fixed ``action_seq``, averaged over starting ``states``. Compounding
    error -- each model is fed its own previous prediction."""
    if not states or not action_seq:
        raise ValueError("need at least one state and one action in the sequence")
    divs = []
    for s in states:
        so, sl = s, s
        for a in action_seq:
            so = oracle_dyn(so, a)
            sl = learned_dyn(sl, a)
        divs.append(_l2(sl, so))
    return sum(divs) / len(divs)


def mean_action_ranking_agreement(oracle_dyn, learned_dyn, states, actions, score_fn):
    """M3: mean Kendall tau between the oracle and learned ranking of actions
    by the score of their predicted successor, over states where the oracle
    ranking is non-degenerate. Returns None (-> JSON null) if no such state.

    Score direction is irrelevant: tau measures concordance of the two
    orderings, so a lower-is-better or higher-is-better score gives the same
    agreement as long as the SAME score_fn is applied to both arms.
    """
    if len(actions) < 2:
        raise ValueError("action ranking needs at least two actions")
    agreements = []
    for s in states:
        oracle_scores = [score_fn(oracle_dyn(s, a)) for a in actions]
        learned_scores = [score_fn(learned_dyn(s, a)) for a in actions]
        try:
            agreements.append(kendall_tau(oracle_scores, learned_scores))
        except ValueError:
            # Degenerate ranking on either side (e.g. an action-blind learned
            # model whose predicted successors all score equally). Undefined
            # here, which is itself informative; skip the state.
            continue
    if not agreements:
        return None
    return sum(agreements) / len(agreements)


# --------------------------------------------------------------------------- #
# Cell registry. Each cell pairs a committed downstream-CPG JSON with the      #
# checkpoint that produced it, plus the env-specific oracle / score / actions. #
# Builders are lazy (called only when the checkpoint is present) so importing  #
# this module never pulls in torch or dm_control.                              #
# --------------------------------------------------------------------------- #

def _reacher_pieces():
    from wmel.envs.dmc_reacher import (
        DMCReacherEnv,
        make_reacher_oracle_dynamics,
        reacher_reach_score,
    )
    env = DMCReacherEnv()
    return env, make_reacher_oracle_dynamics(), reacher_reach_score, env.action_space


def _cartpole_pieces():
    from wmel.envs.dmc_cartpole import (
        DMCCartpoleEnv,
        cartpole_upright_score,
        make_cartpole_oracle_dynamics,
    )
    env = DMCCartpoleEnv()
    return env, make_cartpole_oracle_dynamics(), cartpole_upright_score, env.action_space


def _acrobot_pieces():
    from wmel.adapters.mlp_world_model import acrobot_upright_score
    from wmel.envs.dmc_acrobot import DMCAcrobotEnv, make_acrobot_oracle_dynamics
    levels = (-1.0, -0.5, 0.0, 0.5, 1.0)
    env = DMCAcrobotEnv(discrete_levels=levels)
    return env, make_acrobot_oracle_dynamics(), acrobot_upright_score, env.action_space


# (env_tag, results_dir, pieces_builder, [(downstream_json, checkpoint_pt), ...])
CELLS = [
    ("dmc_reacher", "results/dmc_reacher", _reacher_pieces, [
        ("tdmpc2_cpg.json", "tdmpc2_reacher.pt"),
        ("tdmpc2_cpg_seed1.json", "tdmpc2_reacher_seed1.pt"),
        ("tdmpc2_cpg_seed2.json", "tdmpc2_reacher_seed2.pt"),
    ]),
    ("dmc_cartpole", "results/dmc_cartpole", _cartpole_pieces, [
        ("tdmpc2_cpg.json", "tdmpc2_cartpole.pt"),
        ("tdmpc2_cpg_seed1.json", "tdmpc2_cartpole_seed1.pt"),
        ("tdmpc2_cpg_seed2.json", "tdmpc2_cartpole_seed2.pt"),
        ("tdmpc2_cpg_size5_seed0.json", "tdmpc2_cartpole_size5_seed0.pt"),
        ("tdmpc2_cpg_size5_seed1.json", "tdmpc2_cartpole_size5_seed1.pt"),
        ("tdmpc2_cpg_size5_seed2.json", "tdmpc2_cartpole_size5_seed2.pt"),
    ]),
    ("dmc_acrobot", "results/dmc_acrobot", _acrobot_pieces, [
        ("tdmpc2_cpg.json", "tdmpc2_acrobot.pt"),
    ]),
]


def collect_states(env, action_space, n_states: int, seed: int) -> list:
    """A fixed, seeded random-action rollout through the real env, returning
    the flat observations visited. A reproducible proxy for the on-task state
    distribution the planner traverses; resets periodically to keep coverage
    broad rather than trailing off down one trajectory."""
    rng = random.Random(seed)
    states = []
    obs = env.reset()
    for i in range(n_states):
        states.append(tuple(float(x) for x in obs))
        if i == n_states - 1:
            break  # no need to advance past the last collected state
        if (i + 1) % 25 == 0:
            obs = env.reset()
        else:
            obs = env.step(rng.choice(action_space))
    return states


def evaluate_cell(results_dir, downstream_json, checkpoint_pt, pieces_builder,
                  n_states, k_steps, seed):
    """Compute M1/M2/M3 for one cell and pair with its committed CPG. Returns a
    row dict, or a skip dict if the checkpoint or the downstream JSON is
    missing."""
    json_path = _REPO_ROOT / results_dir / downstream_json
    ckpt_path = _REPO_ROOT / results_dir / checkpoint_pt

    if not json_path.exists():
        return {"skipped": f"downstream JSON not found: {json_path.relative_to(_REPO_ROOT)}"}
    bundle = json.loads(json_path.read_text())
    cpg = bundle.get("cpg", {})
    training = bundle.get("training", {})

    if not ckpt_path.exists():
        return {
            "env": results_dir.split("/")[-1],
            "downstream_json": downstream_json,
            "gap": cpg.get("gap"),
            "skipped": (f"checkpoint not found: {ckpt_path.relative_to(_REPO_ROOT)} "
                        "(gitignored; run on the training box or scp the .pt here)"),
        }

    # Heavy path: load env + oracle + learned dynamics (CPU) and measure.
    from wmel.adapters.tdmpc2_adapter import make_tdmpc2_dynamics
    env, oracle_dyn, score_fn, action_space = pieces_builder()
    learned_dyn = make_tdmpc2_dynamics(ckpt_path, device="cpu")

    states = collect_states(env, action_space, n_states, seed)
    seq_rng = random.Random(seed + 1)
    action_seq = [seq_rng.choice(action_space) for _ in range(k_steps)]

    m1 = mean_one_step_l2(oracle_dyn, learned_dyn, states, action_space)
    m2 = mean_kstep_divergence(oracle_dyn, learned_dyn, states, action_seq)
    m3 = mean_action_ranking_agreement(oracle_dyn, learned_dyn, states, action_space, score_fn)

    return {
        "env": results_dir.split("/")[-1],
        "model_size": training.get("tdmpc2_model_size"),
        "seed": bundle.get("seed"),
        "verdict": cpg.get("verdict"),
        "m1_l2_onestep": round(m1, 6),
        "m2_l2_kstep": round(m2, 6),
        "m3_action_agreement": None if m3 is None else round(m3, 6),
        "gap": cpg.get("gap"),
        # Context (dicts/strings are ignored by correlate.py's numeric auto-detect).
        "context": {
            "downstream_json": downstream_json,
            "oracle_success_rate": cpg.get("oracle_success_rate"),
            "learned_success_rate": cpg.get("learned_success_rate"),
            "training_steps": training.get("training_steps"),
            "n_states": len(states),
            "k_steps": k_steps,
        },
    }


def _within_env_report(rows):
    """For each env with >=3 evaluated cells, the Spearman correlation of each
    offline metric with the downstream gap (the fair, same-units comparison)."""
    out = {}
    by_env = {}
    for r in rows:
        by_env.setdefault(r["env"], []).append(r)
    for env, cells in by_env.items():
        if len(cells) < 3:
            continue
        env_block = {"n_cells": len(cells), "metrics": {}}
        for key in ("m1_l2_onestep", "m2_l2_kstep", "m3_action_agreement"):
            pairs = [(c[key], c["gap"]) for c in cells
                     if isinstance(c.get(key), (int, float)) and isinstance(c.get("gap"), (int, float))]
            if len(pairs) < 3:
                env_block["metrics"][key] = {"n": len(pairs), "skipped": "fewer than 3 usable cells"}
                continue
            try:
                res = bootstrap_correlation_ci([p[0] for p in pairs], [p[1] for p in pairs],
                                               method="spearman", n_boot=10_000, alpha=0.05, seed=0)
                env_block["metrics"][key] = {
                    "n": res.n_pairs, "rho": round(res.rho, 4),
                    "ci": [round(res.ci_low, 4), round(res.ci_high, 4)],
                    "clears_zero": res.ci_low > 0 or res.ci_high < 0,
                }
            except ValueError as exc:
                env_block["metrics"][key] = {"n": len(pairs), "skipped": str(exc)}
        out[env] = env_block
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny state sample for wiring validation")
    ap.add_argument("--n-states", type=int, default=None)
    ap.add_argument("--k-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_states = args.n_states if args.n_states is not None else (8 if args.smoke else 150)
    k_steps = args.k_steps if args.k_steps is not None else (4 if args.smoke else 10)

    rows = []
    skipped = []
    for env_tag, results_dir, pieces_builder, cells in CELLS:
        for downstream_json, checkpoint_pt in cells:
            rec = evaluate_cell(results_dir, downstream_json, checkpoint_pt,
                                pieces_builder, n_states, k_steps, args.seed)
            if "skipped" in rec:
                skipped.append({"cell": f"{env_tag}/{downstream_json}", "reason": rec["skipped"]})
                m = rec.get("gap")
                print(f"  SKIP {env_tag}/{downstream_json:<28} gap={m}  ({rec['skipped']})")
                continue
            rows.append(rec)
            m3 = rec["m3_action_agreement"]
            m3s = "n/a   " if m3 is None else f"{m3:+.3f}"
            print(f"  {rec['env']:<12} size={rec['model_size']} seed={rec['seed']}  "
                  f"M1={rec['m1_l2_onestep']:.4f} M2={rec['m2_l2_kstep']:.4f} "
                  f"M3={m3s} gap={rec['gap']:+.3f}")

    within_env = _within_env_report(rows)
    if within_env:
        print("\nWithin-env Spearman(offline metric, downstream gap) "
              "[the fair, same-units comparison]:")
        for env, block in within_env.items():
            print(f"  {env} (n={block['n_cells']}):")
            for key, m in block["metrics"].items():
                if "skipped" in m:
                    print(f"    {key:<22} skipped: {m['skipped']}")
                else:
                    tag = "RESOLVED    " if m["clears_zero"] else "within noise"
                    print(f"    {key:<22} n={m['n']} rho={m['rho']:+.3f} "
                          f"CI [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}] -> {tag}")

    report = {
        **report_envelope_metadata(),
        "metric": "tdmpc2_offline_downstream_sweep",
        "note": (
            "Stage 2 of the offline->downstream keystone: M1 one-step L2 (naive), "
            "M2 k-step L2 divergence (compounding), M3 action-ranking agreement "
            "(decision-aware, unitless) for the committed TD-MPC2 cells, vs the "
            "committed downstream CPG (gap). M1/M2 are L2 in each env's own state "
            "space and are NOT comparable across envs -- the fair head-to-head is "
            "within_env_correlation below; pooled cross-env correlate.py output is "
            "valid only for the unitless M3. Checkpoints (.pt) are gitignored, so "
            "cells without the weights are listed under skipped."
        ),
        "config": {"n_states": n_states, "k_steps": k_steps, "state_sample_seed": args.seed},
        "within_env_correlation": within_env,
        "skipped": skipped,
        "cells": rows,
    }
    out = _REPO_ROOT / "results" / "offline_downstream" / "tdmpc2_offline_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")  # null, never NaN
    print(f"\nWrote {out.relative_to(_REPO_ROOT)} ({len(rows)} cells, {len(skipped)} skipped)")
    if not rows:
        print("No checkpoints found -- this is the expected output on a checkout "
              "without the .pt weights. Run where the TD-MPC2 checkpoints live.")


if __name__ == "__main__":
    main()
