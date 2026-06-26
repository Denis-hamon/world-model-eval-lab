"""Pool per-seed DreamerV3 CPG JSONs into a single pooled-N report.

Reads the per-seed JSONs produced by the per-env `dreamerv3_cpg.py` scripts
(seed 0 has no suffix; seeds N>0 use `--out-suffix _seedN`), reconstructs the
per-episode result lists from `oracle_full`/`learned_full`, concatenates them,
and re-runs `wmel.metrics.counterfactual_planning_gap` on the concatenation to
obtain a pooled-N estimate with a tighter CI. Mirrors
`experiments.dmc_cartpole.pool_cpg` for the DreamerV3 arm.

Usage
-----
    python -m experiments.pool_dreamerv3_cpg --env dmc_acrobot --seeds 0 1 2
    python -m experiments.pool_dreamerv3_cpg --env dmc_cartpole --seeds 0 1 2
    python -m experiments.pool_dreamerv3_cpg --env dmc_reacher --seeds 0 1 2

Writes:
    results/<env>/dreamerv3_cpg_pooled.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wmel.metrics import EpisodeResult, counterfactual_planning_gap, cpg_verdict


def _suffix(seed: int) -> str:
    return "" if seed == 0 else f"_seed{seed}"


def _episodes_from(payload: dict, arm_key: str) -> list[EpisodeResult]:
    arm = payload[arm_key]
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", required=True, help="e.g. dmc_acrobot, dmc_cartpole, dmc_reacher")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--stem", default="dreamerv3_cpg", help="Per-seed JSON stem (default dreamerv3_cpg).")
    args = p.parse_args()

    results_dir = _REPO_ROOT / "results" / args.env
    paths = [results_dir / f"{args.stem}{_suffix(s)}.json" for s in args.seeds]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing per-seed JSONs: {missing}")

    payloads = [json.loads(p.read_text()) for p in paths]
    oracle = [e for p in payloads for e in _episodes_from(p, "oracle_full")]
    learned = [e for p in payloads for e in _episodes_from(p, "learned_full")]
    cpg = counterfactual_planning_gap(oracle, learned)
    verdict = cpg_verdict(cpg)

    pooled = {
        "environment": payloads[0]["environment"],
        "metric": "counterfactual_planning_gap",
        "learned_model": payloads[0].get("learned_model", "dreamerv3"),
        "pooling": {
            "seeds": [int(p["seed"]) for p in payloads],
            "n_per_seed": [len(_episodes_from(p, "oracle_full")) for p in payloads],
            "n_total": len(oracle),
        },
        "cpg": {**asdict(cpg), "verdict": verdict},
        "training": payloads[0].get("training"),
        "varied_init": payloads[0].get("varied_init"),
    }
    out_path = results_dir / f"{args.stem}_pooled.json"
    out_path.write_text(json.dumps(pooled, indent=2) + "\n")
    print(f"[ok] wrote {out_path.relative_to(_REPO_ROOT)} (n={len(oracle)})")
    print(f"     oracle={cpg.oracle_success_rate:.3f} learned={cpg.learned_success_rate:.3f} "
          f"gap={cpg.gap:+.3f} CI=[{cpg.gap_ci_low:+.3f}, {cpg.gap_ci_high:+.3f}] verdict={verdict}")


if __name__ == "__main__":
    main()
