"""Power analysis for the Counterfactual Planning Gap.

Turns the CPG verdict from a post-hoc label into a planning tool: before
running a comparison, how many episodes per arm are needed for the
Agresti-Caffo interval to reach a target precision, and is a given gap even
detectable at a given n? Pure arithmetic on the same AC standard error the
CPG CI uses (see ``wmel.metrics``); zero environment interaction, zero GPU.

The motivating observation: point-estimate leaderboards report success rates
without intervals, so a reader cannot tell whether a reported ranking gap is
real or noise at the sample size used. This script produces the table that
answers that question, and traces one practitioner decision end-to-end.

Usage:

    python -m experiments.power_analysis

Writes:

    results/power_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.metrics import (  # noqa: E402
    ac_ci_half_width,
    detectable_gap_at_n,
    required_n_for_half_width,
)
from wmel.report import report_envelope_metadata  # noqa: E402

# Per-arm episode counts a practitioner realistically chooses.
N_GRID = [10, 30, 50, 100, 150, 300, 500]
# Hypothesised oracle rates spanning the regimes seen in the paper.
P_GRID = [0.3, 0.5, 0.7, 0.9]
# Target half-widths: 0.10 = +/-10pp (decide a coarse gap), 0.05 = +/-5pp.
TARGET_HALF_WIDTHS = [0.10, 0.05, 0.02]


def half_width_table() -> dict:
    """AC half-width as a function of (oracle rate, n), learned arm at 0.

    The learned arm is fixed at 0 because that is the regime the worked
    examples operate in; it is also the widest-variance learned arm for a
    given oracle rate, so the table is a conservative (upper) bound on n.
    """
    rows = []
    for p in P_GRID:
        rows.append(
            {
                "oracle_rate": p,
                "learned_rate": 0.0,
                "half_width_by_n": {
                    str(n): round(ac_ci_half_width(p, 0.0, n), 4) for n in N_GRID
                },
            }
        )
    return {"n_grid": N_GRID, "rows": rows}


def required_n_table() -> dict:
    """Per-arm n needed to reach each target half-width, learned arm at 0."""
    rows = []
    for p in P_GRID:
        rows.append(
            {
                "oracle_rate": p,
                "learned_rate": 0.0,
                "required_n": {
                    str(t): required_n_for_half_width(p, 0.0, t)
                    for t in TARGET_HALF_WIDTHS
                },
            }
        )
    return {"targets": TARGET_HALF_WIDTHS, "rows": rows}


def leaderboard_gap_audit() -> list:
    """Whether reported point-estimate ranking gaps are detectable at their n.

    Applies the verdict gate to pairs of success rates a leaderboard would
    report without intervals. Demonstrates that a small reported gap at a
    typical evaluation budget can be statistically indistinguishable from
    noise. The pairs are illustrative success-rate contrasts of the kind
    point-estimate world-model leaderboards report at n = 100 per arm.
    """
    pairs = [
        {"label": "top-2 near-tie (0.94 vs 0.92)", "a": 0.94, "b": 0.92},
        {"label": "moderate gap (0.94 vs 0.78)", "a": 0.94, "b": 0.78},
        {"label": "wide gap (0.86 vs 0.62)", "a": 0.86, "b": 0.62},
        {"label": "mid-table tie (0.78 vs 0.75)", "a": 0.78, "b": 0.75},
    ]
    n_reported = 100
    out = []
    for pr in pairs:
        a, b = pr["a"], pr["b"]
        out.append(
            {
                "label": pr["label"],
                "rate_a": a,
                "rate_b": b,
                "n_reported": n_reported,
                "detectable_at_n_reported": detectable_gap_at_n(a, b, n_reported),
                "half_width_at_n_reported": round(ac_ci_half_width(a, b, n_reported), 4),
                "n_for_half_width_0_05": required_n_for_half_width(a, b, 0.05),
            }
        )
    return out


def traced_decision() -> dict:
    """One practitioner decision end-to-end on a real INCONCLUSIVE cell.

    The Cartpole model_size=1, CEM, TD-MPC2 cell returned INCONCLUSIVE at
    n=30 (oracle 0.5, learned 0.533, CI [-0.276, +0.214]). The power table
    prescribes the n needed to resolve it to a half-width of 0.05.
    """
    oracle, learned, n_now = 0.5, 0.533, 30
    hw_now = ac_ci_half_width(oracle, learned, n_now)
    n_needed = required_n_for_half_width(oracle, learned, 0.05)
    return {
        "cell": "dmc_cartpole model_size=1 CEM TD-MPC2",
        "observed": {
            "oracle_rate": oracle,
            "learned_rate": learned,
            "n_per_arm": n_now,
            "half_width": round(hw_now, 4),
            "verdict": "INCONCLUSIVE",
        },
        "prescription": {
            "target_half_width": 0.05,
            "n_per_arm_needed": n_needed,
            "note": "Episodes per arm to shrink the AC interval to +/-0.05; "
            "whether the verdict then commits depends on the true rates the "
            "larger sample reveals.",
        },
    }


def main() -> None:
    report = {
        **report_envelope_metadata(),
        "metric": "cpg_power_analysis",
        "half_width_table": half_width_table(),
        "required_n_table": required_n_table(),
        "leaderboard_gap_audit": leaderboard_gap_audit(),
        "traced_decision": traced_decision(),
    }

    print("# CPG power analysis (AC plus-4, z=1.96)")
    print()
    print("Half-width vs n (learned arm at 0):")
    hdr = "  oracle  " + "".join(f"{n:>8}" for n in N_GRID)
    print(hdr)
    for row in report["half_width_table"]["rows"]:
        cells = "".join(
            f"{row['half_width_by_n'][str(n)]:>8.3f}" for n in N_GRID
        )
        print(f"  {row['oracle_rate']:>6.2f}  {cells}")
    print()
    print("Required n per arm for target half-width (learned arm at 0):")
    print("  oracle   hw<=0.10  hw<=0.05  hw<=0.02")
    for row in report["required_n_table"]["rows"]:
        r = row["required_n"]
        print(f"  {row['oracle_rate']:>6.2f}   {str(r['0.1']):>8}  {str(r['0.05']):>8}  {str(r['0.02']):>8}")
    print()
    print("Leaderboard gap audit (n=100 per arm, no-CI point estimates):")
    for a in report["leaderboard_gap_audit"]:
        verdict = "DECIDABLE" if a["detectable_at_n_reported"] else "WITHIN NOISE"
        print(
            f"  {a['label']:<32} hw={a['half_width_at_n_reported']:.3f}  "
            f"-> {verdict}  (n for hw<=0.05: {a['n_for_half_width_0_05']})"
        )
    print()
    td = report["traced_decision"]
    print(f"Traced decision: {td['cell']}")
    print(
        f"  observed n={td['observed']['n_per_arm']} half-width={td['observed']['half_width']} "
        f"-> {td['observed']['verdict']}"
    )
    print(
        f"  to reach half-width {td['prescription']['target_half_width']}: "
        f"n={td['prescription']['n_per_arm_needed']} per arm"
    )

    out_path = _REPO_ROOT / "results" / "power_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
