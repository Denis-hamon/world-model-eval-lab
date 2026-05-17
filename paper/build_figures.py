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

    # ----------------------------------------------------------------
    # Section 5.5 / Table 2 (multi-seed CPG sweep across training-set
    # sizes). Generated only if the sweep JSON exists.
    # ----------------------------------------------------------------
    sweep_path = _REPO_ROOT / "results" / "dmc_acrobot" / "cpg_sweep.json"
    if not sweep_path.exists():
        print()
        print(f"# (No sweep JSON at {sweep_path.relative_to(_REPO_ROOT)} - "
              "Section 5.5 table will not be regenerated.)")
        return

    sweep = json.loads(sweep_path.read_text())
    print()
    print("# Values to paste into paper/main.tex Table 2 (\\label{tab:sweep}).")
    print(f"# Source: {sweep_path.relative_to(_REPO_ROOT)}")
    print(f"# wmel_version: {sweep.get('wmel_version')}, "
          f"generated_at: {sweep.get('generated_at')}, seeds: {sweep.get('seeds')}")
    print()
    print(f"{'data':>6}  {'val_mse':>8}  {'oracle':>11}  {'learned':>11}  "
          f"{'CPG':>7}  {'CI low':>7}  {'CI hi':>7}  verdict")
    for cell in sweep["cells"]:
        per_seed = cell["per_seed"]
        avg_val_mse = sum(s["val_mse"] for s in per_seed) / len(per_seed)
        o_succ = sum(s["oracle_successes"] for s in per_seed)
        o_n = sum(s["n_oracle"] for s in per_seed)
        l_succ = sum(s["learned_successes"] for s in per_seed)
        l_n = sum(s["n_learned"] for s in per_seed)
        p = cell["pooled_cpg"]
        print(
            f"{cell['data_size']:>6}  {avg_val_mse:>8.4f}  "
            f"{o_succ:>3}/{o_n:<3} ={p['oracle_success_rate']:>5.3f}  "
            f"{l_succ:>3}/{l_n:<3} ={p['learned_success_rate']:>5.3f}  "
            f"{p['gap']:>+.3f}  {p['gap_ci_low']:>+.3f}  {p['gap_ci_high']:>+.3f}  "
            f"{p['verdict']}"
        )

    # ----------------------------------------------------------------
    # Section 5.6.1 coverage receipt. Generated only if coverage.json
    # exists.
    # ----------------------------------------------------------------
    cov_path = _REPO_ROOT / "results" / "dmc_acrobot" / "coverage.json"
    if not cov_path.exists():
        print()
        print(f"# (No coverage JSON at {cov_path.relative_to(_REPO_ROOT)} - "
              "Section 5.6.1 receipt will not be regenerated.)")
        return

    cov = json.loads(cov_path.read_text())
    print()
    print("# Values for paper/main.tex Section 5.6.1 'Empirical receipt for the coverage claim'.")
    print(f"# Source: {cov_path.relative_to(_REPO_ROOT)}")
    print()
    print(f"{'dataset':>18}  {'n':>5}  {'mean u':>7}  {'max u':>7}  {'frac>1.0':>9}  {'frac>1.5':>9}")
    for d in cov["datasets"]:
        print(
            f"{d['label']:>18}  {d['n_states']:>5}  "
            f"{d['mean_uprightness']:>+7.3f}  {d['max_uprightness']:>+7.3f}  "
            f"{d['frac_above_1_0']*100:>8.2f}%  {d['frac_above_1_5']*100:>8.2f}%"
        )


if __name__ == "__main__":
    main()
