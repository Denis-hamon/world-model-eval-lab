"""Paired-aware re-analysis of the non-degenerate CPG cells (no GPU).

The CPG design is *paired*: episode k starts from the same initial state in the
oracle and learned arms. The headline interval (Agresti-Caffo) treats the two
arms as *independent* proportions and so ignores that pairing. This script
recomputes every non-degenerate cell with the paired-aware estimators added in
`wmel.metrics` -- the exact McNemar test and Newcombe's paired-difference CI --
alongside the existing AC interval and the paired bootstrap, and applies a Holm
family-wise correction across the McNemar tests.

It is a pure offline recompute: the per-episode, index-aligned success arrays
are committed in the per-seed result JSONs (`oracle_full` / `learned_full` /
`mlp_full` / `tdmpc2_full`, each with a `results` list). No planner is re-run,
no GPU, no torch.

Why it matters. Whether the independence assumption in AC distorts anything is
an empirical question about the within-pair correlation phi, which this audit
measures per cell. Here phi is ~0 on Reacher and slightly negative on Cartpole,
so the paired-design intervals are essentially interchangeable with AC (Newcombe
is within a few points of AC's width in every cell): the corroboration is that
all three intervals still agree on whether each interval clears zero, not that
the paired estimators tighten it. The audit also surfaces where the verdict is
method-sensitive at the reported n (the exact McNemar test, Holm-corrected).

Usage:

    python -m experiments.paired_intervals_audit

Writes:

    results/paired_intervals/paired_vs_ac.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import stdev

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.metrics import (  # noqa: E402
    EpisodeResult,
    counterfactual_planning_gap,
    cpg_verdict,
    holm_correction,
    mcnemar_exact,
    newcombe_paired_diff_ci,
    paired_bootstrap_gap_ci,
)
from wmel.report import report_envelope_metadata  # noqa: E402

# Non-degenerate cells (learned arm has > 0 successes), labelled to match the
# paper. (env_dir, file_prefix, model_size, learned_arm_key). The two Cartpole
# size-5 MLP arms are 0/n boundary cells -- AC's home turf -- and are excluded.
CELLS = [
    ("Reacher-easy | random-shooting | MLP-on-TD-MPC2", "dmc_reacher", "coverage_mlp_on_tdmpc2_cpg", 1, "learned_full"),
    ("Reacher-easy | random-shooting | TD-MPC2", "dmc_reacher", "tdmpc2_cpg", 1, "learned_full"),
    ("Reacher-easy | CEM | MLP-on-TD-MPC2", "dmc_reacher", "cem_cpg", 1, "mlp_full"),
    ("Reacher-easy | CEM | TD-MPC2", "dmc_reacher", "cem_cpg", 1, "tdmpc2_full"),
    ("Cartpole-swingup | random-shooting | TD-MPC2 (size5)", "dmc_cartpole", "tdmpc2_cpg", 5, "learned_full"),
    ("Cartpole-swingup | CEM | TD-MPC2 (size5)", "dmc_cartpole", "cem_cpg", 5, "tdmpc2_full"),
]


def _seed_files(prefix: str, size: int) -> list[str]:
    """Per-seed filenames for a cell, in a fixed seed order (0, 1, 2)."""
    if size == 1:
        return [f"{prefix}.json", f"{prefix}_seed1.json", f"{prefix}_seed2.json"]
    return [f"{prefix}_size{size}_seed{s}.json" for s in (0, 1, 2)]


def _load_arm_by_seed(
    env_dir: str, prefix: str, size: int, arm_key: str
) -> list[list[EpisodeResult]]:
    """One arm's per-episode results, grouped per seed file (fixed seed order).

    Strict: a missing seed file or arm key is an error, so a silently truncated
    cell fails loudly rather than reporting an undersized n as if intended.
    """
    base = _REPO_ROOT / "results" / env_dir
    per_seed: list[list[EpisodeResult]] = []
    for fn in _seed_files(prefix, size):
        path = base / fn
        if not path.exists():
            raise FileNotFoundError(f"expected seed file is missing: {path}")
        payload = json.loads(path.read_text())
        if arm_key not in payload:
            raise KeyError(f"arm {arm_key!r} not in {path}")
        per_seed.append([
            EpisodeResult(success=bool(r["success"]), steps=int(r.get("steps", 0)))
            for r in payload[arm_key]["results"]
        ])
    return per_seed


def _load_arm(env_dir: str, prefix: str, size: int, arm_key: str) -> list[EpisodeResult]:
    """Pool one arm across the 3 seed files, in seed order (pairing preserved)."""
    return [e for seed in _load_arm_by_seed(env_dir, prefix, size, arm_key) for e in seed]


def audit_cell(env_dir: str, prefix: str, size: int, learned_key: str) -> dict:
    oracle_by_seed = _load_arm_by_seed(env_dir, prefix, size, "oracle_full")
    learned_by_seed = _load_arm_by_seed(env_dir, prefix, size, learned_key)
    # Per-seed CPG (each on that seed's n_per_seed episodes): the dispersion the
    # pooled point estimate hides. With equal episodes per seed the pooled gap is
    # the mean of these, so it lies within [min, max].
    per_seed_gaps = [
        counterfactual_planning_gap(o, l).gap
        for o, l in zip(oracle_by_seed, learned_by_seed)
    ]
    oracle = [e for seed in oracle_by_seed for e in seed]
    learned = [e for seed in learned_by_seed for e in seed]
    cpg = counterfactual_planning_gap(oracle, learned)
    verdict = cpg_verdict(cpg)
    pb_gap, pb_lo, pb_hi = paired_bootstrap_gap_ci(oracle, learned)
    nc_diff, nc_lo, nc_hi = newcombe_paired_diff_ci(oracle, learned)
    mc = mcnemar_exact(oracle, learned)
    # Within-pair correlation (phi) from the 2x2 table: 0 means the paired and
    # independent analyses coincide; this is the quantity that decides whether
    # AC's independence assumption matters at all.
    a, b, c, d = mc.both, mc.oracle_only, mc.learned_only, mc.neither
    margin = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(margin) if margin > 0 else 0.0
    # AC and Newcombe half-widths, to show the paired interval is not tighter here.
    ac_hw = (cpg.gap_ci_high - cpg.gap_ci_low) / 2.0
    nc_hw = (nc_hi - nc_lo) / 2.0
    return {
        "n": len(oracle),
        "oracle_rate": round(cpg.oracle_success_rate, 4),
        "learned_rate": round(cpg.learned_success_rate, 4),
        "gap": round(cpg.gap, 4),
        "per_seed_gaps": [round(g, 4) for g in per_seed_gaps],
        "gap_range": [round(min(per_seed_gaps), 4), round(max(per_seed_gaps), 4)],
        "gap_std": round(stdev(per_seed_gaps), 4),
        "phi": round(phi, 4),
        "ac_half_width": round(ac_hw, 4),
        "newcombe_half_width": round(nc_hw, 4),
        "ac_ci": [round(cpg.gap_ci_low, 4), round(cpg.gap_ci_high, 4)],
        "paired_bootstrap_ci": [round(pb_lo, 4), round(pb_hi, 4)],
        "newcombe_ci": [round(nc_lo, 4), round(nc_hi, 4)],
        "mcnemar": {
            "table": {"both": mc.both, "oracle_only": mc.oracle_only,
                      "learned_only": mc.learned_only, "neither": mc.neither},
            "n_discordant": mc.n_discordant,
            "p_value": round(mc.p_value, 4),
        },
        "verdict": verdict,
    }


def _clears_zero(lo: float, hi: float) -> bool:
    return lo > 0.0 or hi < 0.0


def main() -> None:
    cells = []
    for label, env_dir, prefix, size, learned_key in CELLS:
        rec = audit_cell(env_dir, prefix, size, learned_key)
        rec["cell"] = label
        cells.append(rec)

    # Holm family-wise correction across the McNemar tests.
    holm = holm_correction([c["mcnemar"]["p_value"] for c in cells])
    for c, h in zip(cells, holm):
        c["mcnemar"]["p_holm"] = round(h, 4)

    report = {
        **report_envelope_metadata(),
        "metric": "paired_intervals_audit",
        "note": (
            "Paired-aware re-analysis of the non-degenerate CPG cells. AC treats the "
            "arms as independent; the measured within-pair correlation phi (per cell) "
            "is ~0 on Reacher and slightly negative on Cartpole, so the paired "
            "intervals are not tighter than AC here -- the corroboration is that all "
            "three intervals agree on clearing zero. McNemar p-values carry a Holm "
            "family-wise correction across cells. per_seed_gaps reports the CPG on "
            "each of the three seeds separately (the dispersion the pooled estimate "
            "hides); the pooled gap is their mean and gap_std is the sample (n-1) "
            "standard deviation over the three seeds."
        ),
        "cells": cells,
    }

    print("# Paired-aware re-analysis of the non-degenerate CPG cells")
    print("# AC = independent-proportions; pBoot/Newcombe/McNemar = paired-design-correct")
    print()
    header = (f"  {'cell':<52} {'n':>3} {'orc':>5} {'lrn':>5} {'gap':>7} {'phi':>6} "
              f"{'AC CI':>17} {'pBoot CI':>17} {'Newcombe CI':>17} {'McN p':>6} {'Holm':>6}")
    print(header)
    for c in cells:
        ac = f"[{c['ac_ci'][0]:+.2f},{c['ac_ci'][1]:+.2f}]"
        pb = f"[{c['paired_bootstrap_ci'][0]:+.2f},{c['paired_bootstrap_ci'][1]:+.2f}]"
        nc = f"[{c['newcombe_ci'][0]:+.2f},{c['newcombe_ci'][1]:+.2f}]"
        print(f"  {c['cell']:<52} {c['n']:>3} {c['oracle_rate']:>5.2f} {c['learned_rate']:>5.2f} "
              f"{c['gap']:>+7.3f} {c['phi']:>+6.2f} {ac:>17} {pb:>17} {nc:>17} "
              f"{c['mcnemar']['p_value']:>6.3f} {c['mcnemar']['p_holm']:>6.3f}")
    print()

    # Per-seed CPG dispersion: the spread the pooled point estimate hides (3 seeds).
    print("Per-seed CPG (3 seeds x n_per_seed) -- pooled gap is their mean, so it lies in [min, max]:")
    for c in cells:
        seeds = ", ".join(f"{g:+.2f}" for g in c["per_seed_gaps"])
        print(f"  {c['cell']:<52} pooled {c['gap']:>+7.3f}  per-seed [{seeds}]  "
              f"range [{c['gap_range'][0]:+.2f},{c['gap_range'][1]:+.2f}]  std(n-1) {c['gap_std']:.3f}")
    print()

    # Agreement summary: does each paired estimator agree with AC on clearing zero?
    agree_pb = sum(1 for c in cells if _clears_zero(*c["paired_bootstrap_ci"]) == _clears_zero(*c["ac_ci"]))
    agree_nc = sum(1 for c in cells if _clears_zero(*c["newcombe_ci"]) == _clears_zero(*c["ac_ci"]))
    print(f"Clears-zero agreement with AC: paired-bootstrap {agree_pb}/{len(cells)}, "
          f"Newcombe {agree_nc}/{len(cells)}.")
    borderline = [c for c in cells if _clears_zero(*c["ac_ci"]) and c["mcnemar"]["p_holm"] > 0.05]
    if borderline:
        print("Method-sensitive at the reported n (AC clears zero but Holm-McNemar p > 0.05):")
        for c in borderline:
            print(f"  - {c['cell']}: McNemar p={c['mcnemar']['p_value']:.3f}, "
                  f"Holm p={c['mcnemar']['p_holm']:.3f} -- warrants the n=150 follow-up.")

    out_path = _REPO_ROOT / "results" / "paired_intervals" / "paired_vs_ac.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
