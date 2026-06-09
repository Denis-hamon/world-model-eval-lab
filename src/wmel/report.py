"""Reporting helpers: print a scorecard, dump a JSON report, render Markdown."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Sequence

from wmel.metrics import EpisodeResult, Scorecard


# Schema version of the JSON report envelope. Bump on breaking changes; keep
# the envelope additive (consumers should ignore unknown keys).
REPORT_SCHEMA_VERSION = "1.0"


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def print_scorecard(scorecard: Scorecard) -> None:
    """Print a human-readable scorecard to stdout."""
    header = f"Scorecard: {scorecard.policy_name}"
    if scorecard.perturbation_name:
        header += f"  (perturbation: {scorecard.perturbation_name})"
    print(header)
    print("-" * 48)
    print(f"  episodes                       : {scorecard.episodes}")
    print(f"  action success rate            : {_fmt(scorecard.success_rate)}")
    print(f"  average steps to success       : {_fmt(scorecard.average_steps_to_success)}")
    print(f"  planning latency per call (ms) : {_fmt(scorecard.average_planning_latency_ms)}")
    print(f"  perturbation recovery rate     : {_fmt(scorecard.perturbation_recovery_rate)}")
    print(f"  average compute per decision   : {_fmt(scorecard.average_compute_per_decision)}")
    for name, value in scorecard.extras.items():
        print(f"  {name:<31}: {_fmt(value)}")
    print()


def report_envelope_metadata() -> dict:
    """Return the standard top-level versioning fields for any report dict.

    Use it to stamp `schema_version`, `wmel_version`, and `generated_at`
    on the outer wrapper of scripts that bundle multiple per-run envelopes
    (e.g., `examples/maze_toy/run_baseline.py` returns one dict containing
    three `runs`, and the outer dict should be versioned just like each
    inner one).

    Typical usage:

        report = {
            **report_envelope_metadata(),
            "environment": "maze_toy",
            "runs": {...},
        }
    """
    from wmel import __version__ as _wmel_version

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "wmel_version": _wmel_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def to_json_report(
    results: Sequence[EpisodeResult],
    scorecard: Scorecard,
    *,
    extra_metadata: dict | None = None,
) -> dict:
    """Return a JSON-serializable dict combining raw results and the scorecard.

    The envelope is versioned (`schema_version`) and stamped with the wmel
    version that produced it plus a UTC ISO-8601 timestamp. Downstream
    consumers (a future public scoreboard, for instance) can rely on the
    `schema_version` to handle format evolution; bumps will be additive
    whenever possible.

    Pass `extra_metadata` to attach run-level fields (env name, seed, episode
    count, perturbation strategy, anything else worth round-tripping) under
    a top-level `metadata` block. The block is omitted when None.
    """
    from wmel import __version__ as _wmel_version

    envelope: dict = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "wmel_version": _wmel_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorecard": asdict(scorecard),
        "results": [asdict(r) for r in results],
    }
    if extra_metadata:
        envelope["metadata"] = dict(extra_metadata)
    return envelope


def _md_value(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def to_markdown_scorecard(scorecard: Scorecard) -> str:
    """Render a `Scorecard` as a Markdown table, paste-ready for a PR or doc."""
    heading = f"### Scorecard: `{scorecard.policy_name}`"
    if scorecard.perturbation_name:
        heading += f" (perturbation: `{scorecard.perturbation_name}`)"
    lines = [
        heading,
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| episodes | {scorecard.episodes} |",
        f"| action success rate | {_md_value(scorecard.success_rate)} |",
        f"| average steps to success | {_md_value(scorecard.average_steps_to_success, decimals=1)} |",
        f"| planning latency per call (ms) | {_md_value(scorecard.average_planning_latency_ms)} |",
        f"| perturbation recovery rate | {_md_value(scorecard.perturbation_recovery_rate)} |",
        f"| average compute per decision | {_md_value(scorecard.average_compute_per_decision)} |",
    ]
    for name, value in scorecard.extras.items():
        lines.append(f"| {name} | {_md_value(value)} |")
    return "\n".join(lines) + "\n"


def to_markdown_report(scorecards: Sequence[Scorecard], heading: str | None = None) -> str:
    """Render one or more `Scorecard`s as a single Markdown document.

    Useful for dropping a comparison directly into a pull request body or a
    docs page.
    """
    parts: list[str] = []
    if heading:
        parts.append(f"# {heading}\n")
    for sc in scorecards:
        parts.append(to_markdown_scorecard(sc))
    return "\n".join(parts)


@dataclass(frozen=True)
class ModelTableRow:
    """One (environment, model, planner, init) cell of the multi-model CPG table."""

    environment: str
    model: str
    planner: str
    init: str  # "fixed" | "varied"
    n_per_arm: int
    oracle_success_rate: float
    learned_success_rate: float
    gap: float
    gap_ci_low: float
    gap_ci_high: float
    verdict: str
    source: str = ""


def _init_label(report: dict) -> str:
    """Map a report to its initial-state regime ("fixed" | "varied").

    The pooled reports committed before the `varied_init` stamp existed do
    not carry the key, but they pool the task-level (varied-init) per-seed
    runs: every committed `*_seed*.json` they aggregate has
    `varied_init: true`, and the pooled success rates equal the per-seed
    success sums (e.g. cartpole tdmpc2 seeds 0-2: oracle 10+9+9 = 28/30 =
    0.933, learned 7+1+1 = 9/30 = 0.300). A pooled report without the stamp
    is therefore a varied-init row, not a fixed-init one.
    """
    if "varied_init" in report:
        return "varied" if report["varied_init"] else "fixed"
    if report.get("pooling"):
        return "varied"
    return "fixed"


def _capacity_suffix(report: dict, source: str) -> str:
    """Distinguish TD-MPC2 capacity variants so they never share a cell.

    The size lives in `training.tdmpc2_model_size` where the producing
    script stamped it; the CEM and coverage size-5 reports carry it only in
    their filename (`*_size5_*`), so the source path is the documented
    fallback. The default size 1 stays unsuffixed.
    """
    size = (report.get("training") or {}).get("tdmpc2_model_size")
    if size is None and "size5" in source.rsplit("/", 1)[-1]:
        size = 5
    if size is not None and int(size) != 1:
        return f" (size={int(size)})"
    return ""


def _row_from_cpg_dict(
    report: dict, cpg: dict, model: str, source: str
) -> ModelTableRow:
    return ModelTableRow(
        environment=str(report.get("environment", "unknown")),
        model=model + _capacity_suffix(report, source),
        planner=str(report.get("planner") or "random-shooting"),
        init=_init_label(report),
        n_per_arm=int(cpg.get("n_episodes_oracle", 0)),
        oracle_success_rate=float(cpg["oracle_success_rate"]),
        learned_success_rate=float(cpg["learned_success_rate"]),
        gap=float(cpg["gap"]),
        gap_ci_low=float(cpg["gap_ci_low"]),
        gap_ci_high=float(cpg["gap_ci_high"]),
        verdict=str(cpg.get("verdict", "")),
        source=source,
    )


def _fallback_model_name(report: dict) -> str:
    """Best-effort model label for early reports that predate `learned_model`.

    Falls back to the parenthetical of the learned arm's policy name, e.g.
    `tabular-world-model (learned dynamics)` -> `learned`.
    """
    name = report.get("learned_model")
    if name:
        name = str(name)
        # The coverage reports label the arm `mlp_world_model` and put the
        # training-data source in `data_source`; the CEM reports call the
        # same arm `mlp_on_<source>_data`. Normalise to the latter so one
        # arm has one name across planners.
        data_source = report.get("data_source")
        if name == "mlp_world_model" and data_source:
            return f"mlp_on_{data_source}_data"
        return name
    policy = (report.get("learned_scorecard") or {}).get("policy_name", "")
    match = re.search(r"\((.+?)(?: dynamics)?\)", policy)
    return match.group(1) if match else "unspecified"


def model_table_rows_from_report(report: dict, source: str = "") -> list[ModelTableRow]:
    """Extract multi-model table rows from one CPG report dict.

    Handles the two committed report shapes:

    - single-arm reports (`tdmpc2_cpg.py`, `dreamerv3_cpg.py`, `cpg.py`):
      one `cpg` dict plus a `learned_model` field;
    - multi-arm reports (`cem_cpg.py` and the pooled variants): a `cpgs`
      dict keyed by model arm, where the `mlp_on_data` arm is renamed via
      `mlp_data_source` (e.g. `mlp_on_tdmpc2_data`).

    Sweep reports (a `cells` matrix), smoke runs, and non-CPG metrics return
    no rows: the table is a headline summary, not an ablation dump.
    """
    if report.get("metric") != "counterfactual_planning_gap":
        return []
    if report.get("smoke_mode"):
        return []
    if "cells" in report:
        return []

    rows: list[ModelTableRow] = []
    cpg = report.get("cpg")
    if isinstance(cpg, dict):
        rows.append(_row_from_cpg_dict(report, cpg, _fallback_model_name(report), source))
    cpgs = report.get("cpgs")
    if isinstance(cpgs, dict):
        data_source = report.get("mlp_data_source")
        for arm_name, arm_cpg in cpgs.items():
            if not isinstance(arm_cpg, dict):
                continue
            model = (
                f"mlp_on_{data_source}_data"
                if arm_name == "mlp_on_data" and data_source
                else str(arm_name)
            )
            rows.append(_row_from_cpg_dict(report, arm_cpg, model, source))
    return rows


def dedupe_model_table_rows(rows: Sequence[ModelTableRow]) -> list[ModelTableRow]:
    """Keep, per (environment, model, planner, init), the row with the most
    episodes per arm -- pooled results supersede their per-seed runs."""
    best: dict[tuple[str, str, str, str], ModelTableRow] = {}
    for row in rows:
        key = (row.environment, row.model, row.planner, row.init)
        current = best.get(key)
        if current is None or row.n_per_arm > current.n_per_arm:
            best[key] = row
    return sorted(
        best.values(), key=lambda r: (r.environment, r.model, r.planner, r.init)
    )


def to_markdown_model_table(
    rows: Sequence[ModelTableRow], heading: str | None = None
) -> str:
    """Render multi-model CPG rows as a Markdown table.

    One line per (environment, model, planner, init) cell: success rates of
    both arms, the gap with its Agresti--Caffo 95% interval, and the gated
    verdict. This is the cross-model summary a reader should be able to cite
    without opening any JSON.
    """
    lines: list[str] = []
    if heading:
        lines.extend([f"## {heading}", ""])
    lines.extend(
        [
            "| Environment | Model | Planner | Init | n/arm | Oracle | Learned | CPG | 95% AC CI | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in rows:
        ci = f"[{r.gap_ci_low:+.3f}, {r.gap_ci_high:+.3f}]"
        lines.append(
            f"| {r.environment} | {r.model} | {r.planner} | {r.init} | {r.n_per_arm} "
            f"| {r.oracle_success_rate:.3f} | {r.learned_success_rate:.3f} "
            f"| {r.gap:+.3f} | {ci} | {r.verdict or 'n/a'} |"
        )
    return "\n".join(lines) + "\n"
