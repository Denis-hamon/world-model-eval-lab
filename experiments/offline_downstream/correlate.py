"""Correlate offline metrics against downstream performance (no GPU, stdlib).

Stage T2.1.c of the keystone: read an offline-scores bundle (one row per
evaluated cell, each with one or more planner-free offline metrics and a
downstream value) and report, per offline metric, its rank correlation with the
downstream target plus a bootstrap confidence interval. This is the analysis
that answers "does a cheap offline metric predict control usefulness, and which
one best?" -- the external-validity test for the lab.

Pure arithmetic over a committed JSON bundle (produced by, e.g.,
``maze_quality_sweep.py`` on CPU, or a future DMC/TD-MPC2 sweep on GPU); reuses
``wmel.metrics.bootstrap_correlation_ci``. No model, no environment, no torch.

Usage:
    python -m experiments.offline_downstream.correlate                       # maze bundle
    python -m experiments.offline_downstream.correlate --bundle <path> --downstream cpg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.metrics import bootstrap_correlation_ci  # noqa: E402
from wmel.report import report_envelope_metadata  # noqa: E402

# Keys that describe the cell, not an offline metric to be correlated.
DEFAULT_EXCLUDE = {"epochs", "seed", "hidden", "data_size", "model_size",
                   "env", "planner", "model", "variant", "cell"}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def correlate_bundle(
    cells: list[dict],
    downstream_key: str,
    *,
    exclude: set[str] | None = None,
    method: str = "spearman",
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Per-offline-metric rank correlation against the downstream target.

    Auto-detects offline metric keys (numeric, not the downstream key, not a
    cell descriptor). For each, correlates the cells with a finite value for
    both that metric and the downstream target; metrics with fewer than three
    usable cells are reported as skipped rather than correlated.
    """
    exclude = (exclude or DEFAULT_EXCLUDE) | {downstream_key}
    metric_keys = sorted(
        {k for c in cells for k, v in c.items() if k not in exclude and _is_number(v)}
    )
    rows = []
    for key in metric_keys:
        pairs = [
            (c[key], c[downstream_key])
            for c in cells
            if _is_number(c.get(key)) and _is_number(c.get(downstream_key))
        ]
        if len(pairs) < 3:
            rows.append({"metric": key, "n": len(pairs), "skipped": "fewer than 3 usable cells"})
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        try:
            res = bootstrap_correlation_ci(xs, ys, method=method, n_boot=n_boot, alpha=alpha, seed=seed)
        except ValueError as exc:
            rows.append({"metric": key, "n": len(pairs), "skipped": str(exc)})
            continue
        rows.append({
            "metric": key,
            "n": res.n_pairs,
            "rho": round(res.rho, 4),
            "ci": [round(res.ci_low, 4), round(res.ci_high, 4)],
            "clears_zero": res.ci_low > 0 or res.ci_high < 0,
            "n_boot": res.n_boot,
        })

    # Fair cross-metric comparison: restrict every metric to the COMMON subset
    # of cells where ALL metrics (and the downstream) are finite. The per-metric
    # rows above each use that metric's own usable cells, which differ when a
    # metric is undefined for some cells (e.g. an action-blind model has no
    # ranking metric) -- so comparing their magnitudes across different samples
    # is invalid. Only the common-subset block below is apples-to-apples.
    common = [
        c for c in cells
        if _is_number(c.get(downstream_key)) and all(_is_number(c.get(k)) for k in metric_keys)
    ]
    common_rows = []
    if metric_keys and len(common) >= 3:
        ys = [c[downstream_key] for c in common]
        for key in metric_keys:
            xs = [c[key] for c in common]
            try:
                res = bootstrap_correlation_ci(xs, ys, method=method, n_boot=n_boot, alpha=alpha, seed=seed)
                common_rows.append({
                    "metric": key, "n": res.n_pairs, "rho": round(res.rho, 4),
                    "ci": [round(res.ci_low, 4), round(res.ci_high, 4)],
                    "clears_zero": res.ci_low > 0 or res.ci_high < 0,
                })
            except ValueError as exc:
                common_rows.append({"metric": key, "n": len(common), "skipped": str(exc)})

    return {
        "downstream": downstream_key,
        "method": method,
        "n_cells": len(cells),
        "metrics": rows,
        "common_subset": {"n": len(common), "metrics": common_rows},
        "comparability_note": (
            "Per-metric rows use each metric's own usable (finite) cells, whose "
            "counts differ; their magnitudes are NOT comparable across metrics. "
            "Only common_subset (cells where every metric is finite) is a fair "
            "head-to-head."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/offline_downstream/maze_offline_scores.json")
    ap.add_argument("--downstream", default="success_rate")
    ap.add_argument("--method", default="spearman", choices=["spearman", "kendall"])
    args = ap.parse_args()

    path = _REPO_ROOT / args.bundle
    if not path.exists():
        print(f"bundle not found: {path}\nRun an offline-scores sweep first "
              "(e.g. python -m experiments.offline_downstream.maze_quality_sweep).")
        raise SystemExit(1)

    bundle = json.loads(path.read_text())
    cells = bundle["cells"]
    result = correlate_bundle(cells, args.downstream, method=args.method)

    print(f"# Offline metric -> downstream ({args.downstream}) rank correlation "
          f"[{args.method}, {result['n_cells']} cells]")
    print(f"# bundle: {args.bundle}")
    print()
    print("Per-metric (each on its own usable cells; magnitudes NOT comparable across metrics):")
    for r in result["metrics"]:
        if "skipped" in r:
            print(f"  {r['metric']:<24} n={r['n']:<3} skipped: {r['skipped']}")
        else:
            tag = "RESOLVED    " if r["clears_zero"] else "within noise"
            print(f"  {r['metric']:<24} n={r['n']:<3} rho={r['rho']:+.3f}  "
                  f"CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  -> {tag}")
    cs = result["common_subset"]
    print(f"\nCommon subset (n={cs['n']}; the only fair cross-metric comparison):")
    for r in cs["metrics"]:
        if "skipped" in r:
            print(f"  {r['metric']:<24} skipped: {r['skipped']}")
        else:
            tag = "RESOLVED    " if r["clears_zero"] else "within noise"
            print(f"  {r['metric']:<24} n={r['n']:<3} rho={r['rho']:+.3f}  "
                  f"CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  -> {tag}")
    print()

    report = {
        **report_envelope_metadata(),
        "metric": "offline_vs_downstream_correlation",
        "source_bundle": args.bundle,
        "note": (
            "Rank correlation of each offline metric with the downstream target, "
            "with a paired bootstrap CI. A metric whose CI clears zero predicts "
            "downstream performance at this sample of cells; sign follows the "
            "metric's direction (error metrics positive vs a gap target, "
            "negative vs a success target)."
        ),
        "result": result,
    }
    out = _REPO_ROOT / "results" / "offline_downstream" / "offline_vs_downstream.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
