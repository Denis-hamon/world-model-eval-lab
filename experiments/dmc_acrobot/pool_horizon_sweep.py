"""Merge partial cem_cpg_horizon_sweep JSONs into a single pooled report.

Why: the horizon sweep is launched as N parallel processes (one per
seed, split across GPUs) for wall-clock parallelism. Each process
writes a partial JSON with its own per-seed cells already pooled at
n=`episodes_per_cell`. This script concatenates the partial JSONs into
one merged report whose `cells` array is repooled across all seeds.

Usage
-----
    python -m experiments.dmc_acrobot.pool_horizon_sweep \
        --inputs results/dmc_acrobot/cem_cpg_horizon_sweep_seed0.json \
                 results/dmc_acrobot/cem_cpg_horizon_sweep_seed1.json \
                 results/dmc_acrobot/cem_cpg_horizon_sweep_seed2.json \
        --output results/dmc_acrobot/cem_cpg_horizon_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wmel.metrics import EpisodeResult, counterfactual_planning_gap, cpg_verdict
from wmel.report import report_envelope_metadata


def _episodes_from(cell: dict, full_key: str) -> list[EpisodeResult] | None:
    full = cell.get(full_key)
    if full is None:
        return None
    out: list[EpisodeResult] = []
    for r in full["results"]:
        out.append(EpisodeResult(
            success=bool(r["success"]),
            steps=int(r["steps"]),
            planning_latencies_ms=tuple(r.get("planning_latencies_ms", ())),
            perturbed=bool(r.get("perturbed", False)),
            recovered=bool(r.get("recovered", False)),
            compute_per_decision=r.get("compute_per_decision"),
        ))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inputs", nargs="+", required=True, help="Partial horizon-sweep JSONs to merge.")
    p.add_argument("--output", required=True, help="Output JSON path.")
    args = p.parse_args()

    payloads = [json.loads(Path(p).read_text()) for p in args.inputs]
    horizons = sorted({H for pl in payloads for H in pl["horizons"]})
    seeds = [s for pl in payloads for s in pl["seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds across inputs: {seeds}")

    cells_by_H: dict[int, dict] = {}
    for H in horizons:
        oracle_pool: list[EpisodeResult] = []
        mlp_pool: list[EpisodeResult] = []
        tdmpc2_pool: list[EpisodeResult] | None = []
        for pl in payloads:
            cell = next((c for c in pl["cells"] if c["plan_horizon"] == H), None)
            if cell is None:
                continue
            oracle_pool.extend(_episodes_from(cell, "oracle_full") or [])
            mlp_pool.extend(_episodes_from(cell, "mlp_full") or [])
            tdmpc2 = _episodes_from(cell, "tdmpc2_full")
            if tdmpc2 is None:
                tdmpc2_pool = None
            elif tdmpc2_pool is not None:
                tdmpc2_pool.extend(tdmpc2)

        cpgs: dict[str, dict] = {}
        for name, pool in [("mlp_on_data", mlp_pool), ("tdmpc2", tdmpc2_pool)]:
            if pool is None:
                continue
            cpg = counterfactual_planning_gap(oracle_pool, pool)
            cpgs[name] = {**asdict(cpg), "verdict": cpg_verdict(cpg)}

        cells_by_H[H] = {
            "plan_horizon": H,
            "pooled_n_per_arm": len(oracle_pool),
            "cpgs": cpgs,
        }

    merged = {
        **report_envelope_metadata(),
        "environment": "dmc_acrobot_swingup",
        "metric": "counterfactual_planning_gap",
        "planner": "cem",
        "sweep_axis": "plan_horizon",
        "mlp_data_source": payloads[0]["mlp_data_source"],
        "tdmpc2_agent_seed_held_fixed": payloads[0].get("tdmpc2_agent_seed_held_fixed"),
        "seeds": sorted(set(seeds)),
        "horizons": horizons,
        "episodes_per_cell": payloads[0]["episodes_per_cell"],
        "merged_from": [str(p) for p in args.inputs],
        "cells": [cells_by_H[H] for H in horizons],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"Wrote {out_path}")
    print(f"Summary (pooled n={cells_by_H[horizons[0]]['pooled_n_per_arm']} per arm):")
    for H in horizons:
        cell = cells_by_H[H]
        for arm, cpg in cell["cpgs"].items():
            print(f"  H={H:>2d} oracle vs {arm:>15s}: oracle={cpg['oracle_success_rate']:.3f}  learned={cpg['learned_success_rate']:.3f}  "
                  f"gap={cpg['gap']:+.3f}  CI [{cpg['gap_ci_low']:+.3f}, {cpg['gap_ci_high']:+.3f}]  {cpg['verdict']}")


if __name__ == "__main__":
    main()
