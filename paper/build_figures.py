"""Regenerate the paper's headline table values from the committed JSON.

The paper hard-codes the Acrobot scorecards and the CPG numbers in
`paper/main.tex`. If `results/dmc_acrobot/cpg.json` is regenerated (e.g.
after a re-run of `experiments/dmc_acrobot/cpg.py`), this script prints
the matching LaTeX-ready values so the table can be updated by hand.

Stdlib-only. Runs in under a second.

Usage:

    python -m paper.build_figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _fmt(value, decimals: int = 3, default: str = "n/a") -> str:
    if value is None:
        return default
    return f"{value:.{decimals}f}"


def main() -> None:
    cpg_path = _REPO_ROOT / "results" / "dmc_acrobot" / "cpg.json"
    if not cpg_path.exists():
        print(
            f"Missing {cpg_path.relative_to(_REPO_ROOT)}. Run "
            "`python -m experiments.dmc_acrobot.cpg` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = json.loads(cpg_path.read_text())
    cpg = report["cpg"]
    o = report["oracle_scorecard"]
    l = report["learned_scorecard"]

    print("# Values to paste into paper/main.tex Table 1 (\\label{tab:cpg}).")
    print(f"# Source: {cpg_path.relative_to(_REPO_ROOT)}")
    print(f"# wmel_version: {report.get('wmel_version')}, "
          f"generated_at: {report.get('generated_at')}")
    print()
    print(f"Episodes per arm   : {o['episodes']} / {l['episodes']}")
    print(f"Oracle success     : {_fmt(o['success_rate'])} ({int(round(o['success_rate'] * o['episodes']))}/{o['episodes']})")
    print(f"Learned success    : {_fmt(l['success_rate'])} ({int(round(l['success_rate'] * l['episodes']))}/{l['episodes']})")
    print(f"Oracle avg steps   : {_fmt(o['average_steps_to_success'], 1)}")
    print(f"Learned avg steps  : {_fmt(l['average_steps_to_success'], 1)}")
    print(f"Oracle latency/ms  : {_fmt(o['average_planning_latency_ms'], 1)}")
    print(f"Learned latency/ms : {_fmt(l['average_planning_latency_ms'], 1)}")
    print(f"Oracle compute/dec : {_fmt(o['average_compute_per_decision'], 1)}")
    print(f"Learned compute/dec: {_fmt(l['average_compute_per_decision'], 1)}")
    print()
    print(f"Raw CPG            : {cpg['gap']:+.3f}")
    print(f"AC 95% CI low      : {cpg['gap_ci_low']:+.3f}")
    print(f"AC 95% CI high     : {cpg['gap_ci_high']:+.3f}")
    print(f"Verdict            : {cpg['verdict']}")


if __name__ == "__main__":
    main()
