"""Pool per-seed Reacher CPG JSONs into a single pooled-N report.

Reads per-seed JSONs produced by `tdmpc2_cpg.py`, `coverage_mlp_on_tdmpc2.py`,
and `cem_cpg.py`, reconstructs the per-episode result lists, concatenates
them, and runs `wmel.metrics.counterfactual_planning_gap` on the
concatenation to obtain a pooled-N estimate with a tighter CI.

Usage
-----
    python -m experiments.dmc_reacher.pool_cpg --model-size 5 --seeds 0 1 2
    python -m experiments.dmc_reacher.pool_cpg --model-size 1 --seeds 0 1 2

Writes (for --model-size 5 --seeds 0 1 2):
    results/dmc_reacher/tdmpc2_cpg_size5_pooled.json
    results/dmc_reacher/coverage_mlp_on_tdmpc2_cpg_size5_pooled.json
    results/dmc_reacher/cem_cpg_size5_pooled.json
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


def _suffix(model_size: int, seed: int) -> str:
    if model_size == 1 and seed == 0:
        return ""
    if model_size == 1:
        return f"_seed{seed}"
    return f"_size{model_size}_seed{seed}"


def _episodes_from(json_payload: dict, arm_key: str) -> list[EpisodeResult]:
    """Reconstruct EpisodeResult objects from the persisted `arm_key + '_full'.results`."""
    arm = json_payload[arm_key]
    out: list[EpisodeResult] = []
    for r in arm["results"]:
        out.append(EpisodeResult(
            success=bool(r["success"]),
            steps=int(r["steps"]),
            planning_latencies_ms=tuple(r.get("planning_latencies_ms", ())),
            perturbed=bool(r.get("perturbed", False)),
            recovered=bool(r.get("recovered", False)),
            compute_per_decision=r.get("compute_per_decision"),
        ))
    return out


def _cpg_dict(oracle: list[EpisodeResult], learned: list[EpisodeResult]) -> dict:
    cpg = counterfactual_planning_gap(oracle, learned)
    return {**asdict(cpg), "verdict": cpg_verdict(cpg)}


def _pool_tdmpc2_cpg(payloads: list[dict]) -> dict:
    oracle = [e for p in payloads for e in _episodes_from(p, "oracle_full")]
    learned = [e for p in payloads for e in _episodes_from(p, "learned_full")]
    return {
        "environment": payloads[0]["environment"],
        "metric": "counterfactual_planning_gap",
        "learned_model": payloads[0]["learned_model"],
        "pooling": {"seeds": [int(p["seed"]) for p in payloads], "n_per_seed": 10, "n_total": len(oracle)},
        "cpg": _cpg_dict(oracle, learned),
        "training": payloads[0].get("training"),
    }


def _pool_coverage_mlp(payloads: list[dict]) -> dict:
    oracle = [e for p in payloads for e in _episodes_from(p, "oracle_full")]
    learned = [e for p in payloads for e in _episodes_from(p, "learned_full")]
    return {
        "environment": payloads[0]["environment"],
        "metric": "counterfactual_planning_gap",
        "learned_model": payloads[0]["learned_model"],
        "data_source": payloads[0].get("data_source"),
        "pooling": {"seeds": [int(p["seed"]) for p in payloads], "n_per_seed": 10, "n_total": len(oracle)},
        "cpg": _cpg_dict(oracle, learned),
    }


def _pool_cem(payloads: list[dict]) -> dict:
    oracle = [e for p in payloads for e in _episodes_from(p, "oracle_full")]
    mlp = [e for p in payloads for e in _episodes_from(p, "mlp_full")]
    tdmpc2 = [e for p in payloads for e in _episodes_from(p, "tdmpc2_full")] if payloads[0].get("tdmpc2_full") else None
    out = {
        "environment": payloads[0]["environment"],
        "metric": "counterfactual_planning_gap",
        "planner": payloads[0]["planner"],
        "mlp_data_source": payloads[0].get("mlp_data_source"),
        "pooling": {"seeds": [int(p["seed"]) for p in payloads], "n_per_seed": 10, "n_total": len(oracle)},
        "cpgs": {"mlp_on_data": _cpg_dict(oracle, mlp)},
    }
    if tdmpc2 is not None:
        out["cpgs"]["tdmpc2"] = _cpg_dict(oracle, tdmpc2)
    return out


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-size", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    args = p.parse_args()

    results_dir = _REPO_ROOT / "results" / "dmc_reacher"
    size_tag = f"size{args.model_size}_" if args.model_size != 1 else ""
    pooled_tag = f"{size_tag}pooled"

    pieces = [
        ("tdmpc2_cpg", _pool_tdmpc2_cpg),
        ("coverage_mlp_on_tdmpc2_cpg", _pool_coverage_mlp),
        ("cem_cpg", _pool_cem),
    ]
    for stem, pooler in pieces:
        per_seed_paths = [results_dir / f"{stem}{_suffix(args.model_size, s)}.json" for s in args.seeds]
        missing = [str(p) for p in per_seed_paths if not p.exists()]
        if missing:
            print(f"[skip] {stem}: missing {missing}")
            continue
        payloads = [_read(p) for p in per_seed_paths]
        pooled = pooler(payloads)
        out_path = results_dir / f"{stem}_{pooled_tag}.json"
        out_path.write_text(json.dumps(pooled, indent=2) + "\n")
        cpg_repr = pooled.get("cpg") or pooled.get("cpgs")
        print(f"[ok] wrote {out_path.relative_to(_REPO_ROOT)} (n={pooled['pooling']['n_total']})")
        if isinstance(cpg_repr, dict) and "gap" in cpg_repr:
            print(f"       gap={cpg_repr['gap']:+.3f} CI=[{cpg_repr['gap_ci_low']:+.3f}, {cpg_repr['gap_ci_high']:+.3f}] verdict={cpg_repr['verdict']}")
        else:
            for arm, c in cpg_repr.items():
                print(f"       {arm:>15s}: gap={c['gap']:+.3f} CI=[{c['gap_ci_low']:+.3f}, {c['gap_ci_high']:+.3f}] verdict={c['verdict']}")


if __name__ == "__main__":
    main()
