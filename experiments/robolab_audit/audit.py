"""Calibrated reading of the RoboLab-120 generalist-policy leaderboard (no GPU).

This is a *calibration audit*: it takes the published overall success rates of a
real, current generalist-policy benchmark and applies this framework's honesty
layer -- Agresti--Caffo intervals and the CPG power-analysis helpers
(``wmel.metrics``) -- to ask which leaderboard gaps are statistically resolved
at the reported sample size, and how many episodes the close ones would need.
Pure arithmetic on published numbers; zero environment interaction, zero GPU.

Why this benchmark. RoboLab-120 is a high-fidelity simulation benchmark for
task-generalist robot policies, and is the simulation benchmark on which the
open Cosmos 3 DROID policy also reports task success rates. Auditing its
published leaderboard is the lens through which a later paired re-run (running
policies ourselves on shared initial states and ranking them with
``paired_bradley_terry_ranking``) should be read.

Source of the numbers (cite before reuse; transcribed verbatim):
  RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task
  Generalist Policies, arXiv:2604.09860, main results table and evaluation
  protocol. Protocol: N=10 episodes per task across 120 tasks -> n=1200 trials
  per model overall.

Scope, stated honestly. We audit only the *overall* leaderboard (n=1200). The
paper also reports per-competency-axis rates (visual / procedural / relational),
but its tasks are *multi-labeled* across competencies (a task can carry several
axis labels), so the per-axis trial counts are unequal and not a clean
120/3 split. Rather than guess them, the per-axis audit is deferred to the
paired re-run (Stage 1), where we control the episode count per cell directly.

Caveat about the overall figure. The 1200 trials span *heterogeneous tasks*,
not i.i.d. Bernoulli draws of a single success probability. Treating them as
i.i.d. for the Agresti--Caffo interval is the *optimistic* case: task-level
clustering (a design effect) only widens the true interval. So any pair this
audit calls *indistinguishable* stays indistinguishable under the proper
clustered analysis -- the result is a conservative lower bound on how much of
the ordering is noise, not an upper bound.

Usage:

    python -m experiments.robolab_audit.audit

Writes:

    results/robolab_audit/leaderboard_audit.json
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from wmel.metrics import (  # noqa: E402
    ac_ci_half_width,
    detectable_gap_at_n,
)
from wmel.report import report_envelope_metadata  # noqa: E402

# --- Published RoboLab-120 numbers (arXiv:2604.09860) ------------------------
# Overall success rate per model (n = 1200 trials each: 10 episodes x 120 tasks).
N_OVERALL = 1200
OVERALL = {
    "pi0.5": 0.280,
    "pi0-FAST": 0.155,
    "GR00T N1.6": 0.072,
    "pi0": 0.050,
    "PaliGemma": 0.034,
}
SOURCE = "RoboLab-120, arXiv:2604.09860 (N=10 episodes/task, 120 tasks, n=1200/model)"


def _tilde_gap(p_a: float, p_b: float, n: int) -> float:
    """Agresti--Caffo plus-4 point estimate of the difference (same as metrics)."""
    tilde_a = (p_a * n + 1.0) / (n + 2.0)
    tilde_b = (p_b * n + 1.0) / (n + 2.0)
    return tilde_a - tilde_b


def n_to_separate(p_a: float, p_b: float, n_max: int = 200_000) -> int | None:
    """Smallest equal per-arm n at which the AC gate clears zero for this pair.

    Uses the verdict gate itself (``detectable_gap_at_n``), not a half-width
    proxy, so the returned n genuinely separates the pair. The gate is monotone
    in n for fixed rates -- the AC tilde-gap grows toward the raw gap while the
    half-width shrinks -- so a binary search is exact. Returns None if even
    ``n_max`` does not separate them (e.g. identical rates).
    """
    if p_a == p_b or not detectable_gap_at_n(p_a, p_b, n_max):
        return None
    lo, hi = 1, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if detectable_gap_at_n(p_a, p_b, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def audit_scope(rates: dict[str, float], n_per_arm: int) -> dict:
    """Pairwise calibrated reading of a leaderboard.

    For every pair of models, reports the raw gap, the Agresti--Caffo 95% CI on
    the difference of success rates at the reported ``n_per_arm``, whether that
    interval clears zero (so the pairwise ordering is statistically resolved),
    and -- for the pairs it does not resolve -- the per-arm episode count that
    would.
    """
    ranking = sorted(rates, key=lambda m: rates[m], reverse=True)
    rows = []
    for better, worse in combinations(ranking, 2):
        p_a, p_b = rates[better], rates[worse]
        hw = ac_ci_half_width(p_a, p_b, n_per_arm)
        tg = _tilde_gap(p_a, p_b, n_per_arm)
        separable = detectable_gap_at_n(p_a, p_b, n_per_arm)
        rows.append(
            {
                "better": better,
                "worse": worse,
                "rate_better": p_a,
                "rate_worse": p_b,
                "raw_gap": round(p_a - p_b, 4),
                "ac_ci": [round(tg - hw, 4), round(tg + hw, 4)],
                "separable": separable,
                "n_to_separate": None if separable else n_to_separate(p_a, p_b),
            }
        )
    return {"n_per_arm": n_per_arm, "ranking": ranking, "pairs": rows}


def main() -> None:
    overall = audit_scope(OVERALL, N_OVERALL)
    n_pairs = len(overall["pairs"])
    n_separable = sum(1 for r in overall["pairs"] if r["separable"])
    within_noise = [r for r in overall["pairs"] if not r["separable"]]

    report = {
        **report_envelope_metadata(),
        "metric": "robolab_leaderboard_audit",
        "source": SOURCE,
        "scope": "overall leaderboard only (per-axis deferred: tasks are multi-labeled)",
        "caveat": (
            "Trials span heterogeneous tasks, not i.i.d. Bernoulli. The i.i.d. "
            "Agresti-Caffo interval is the optimistic case; task clustering only "
            "widens it, so 'within noise' pairs are conservative."
        ),
        "overall": overall,
        "summary": {
            "pairs": n_pairs,
            "separable": n_separable,
            "within_noise": n_pairs - n_separable,
        },
    }

    print(f"# Calibrated reading of the RoboLab-120 leaderboard ({SOURCE})")
    print("# AC plus-4 95% CI on the difference of success rates; gate = does the CI clear zero?")
    print()
    print(f"Ranking: {' > '.join(overall['ranking'])}")
    for r in overall["pairs"]:
        tag = "RESOLVED    " if r["separable"] else "WITHIN NOISE"
        extra = "" if r["separable"] else f"  (separates at n~{r['n_to_separate']}/model)"
        print(
            f"  {r['better']:>11} vs {r['worse']:<11} "
            f"gap {r['raw_gap']:+.3f}  CI [{r['ac_ci'][0]:+.3f}, {r['ac_ci'][1]:+.3f}]  "
            f"-> {tag}{extra}"
        )
    print()
    print(
        f"Summary: {n_separable}/{n_pairs} pairwise orderings are statistically resolved "
        f"at n={N_OVERALL}. "
        + (
            "All pairs resolved."
            if not within_noise
            else "Not resolved: "
            + ", ".join(f"{r['better']} vs {r['worse']}" for r in within_noise)
            + ". The leaderboard top is robust; the closest tail pair is not "
            "separated at the reported sample size."
        )
    )

    out_path = _REPO_ROOT / "results" / "robolab_audit" / "leaderboard_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
