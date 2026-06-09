"""Tests for the multi-model CPG table (wmel.report) and its generator script.

Stdlib-only: runs in the no-torch CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmel.report import (
    ModelTableRow,
    dedupe_model_table_rows,
    model_table_rows_from_report,
    to_markdown_model_table,
)


def _cpg_dict(oracle: float = 0.9, learned: float = 0.1, n: int = 30) -> dict:
    return {
        "oracle_success_rate": oracle,
        "learned_success_rate": learned,
        "gap": oracle - learned,
        "n_episodes_oracle": n,
        "n_episodes_learned": n,
        "gap_ci_low": oracle - learned - 0.1,
        "gap_ci_high": oracle - learned + 0.1,
        "verdict": "MODEL BOTTLENECK",
    }


def _single_report(**overrides) -> dict:
    report = {
        "environment": "dmc_acrobot_swingup",
        "metric": "counterfactual_planning_gap",
        "learned_model": "dreamerv3",
        "cpg": _cpg_dict(),
        "smoke_mode": False,
        "varied_init": False,
    }
    report.update(overrides)
    return report


def test_single_arm_report_yields_one_row() -> None:
    rows = model_table_rows_from_report(_single_report(), source="a.json")
    assert len(rows) == 1
    row = rows[0]
    assert row.environment == "dmc_acrobot_swingup"
    assert row.model == "dreamerv3"
    assert row.planner == "random-shooting"  # default when the field is absent
    assert row.init == "fixed"
    assert row.n_per_arm == 30
    assert row.verdict == "MODEL BOTTLENECK"
    assert row.source == "a.json"


def test_multi_arm_report_renames_mlp_on_data() -> None:
    report = {
        "environment": "dmc_cartpole_swingup",
        "metric": "counterfactual_planning_gap",
        "planner": "cem",
        "mlp_data_source": "tdmpc2",
        "cpgs": {
            "mlp_on_data": _cpg_dict(0.5, 0.1),
            "tdmpc2": _cpg_dict(0.5, 0.4),
        },
    }
    rows = model_table_rows_from_report(report)
    assert {r.model for r in rows} == {"mlp_on_tdmpc2_data", "tdmpc2"}
    assert all(r.planner == "cem" for r in rows)


def test_smoke_sweep_and_foreign_metrics_yield_no_rows() -> None:
    assert model_table_rows_from_report(_single_report(smoke_mode=True)) == []
    assert model_table_rows_from_report(_single_report(cells={"h1": {}})) == []
    assert model_table_rows_from_report({"metric": "cpg_power_analysis"}) == []
    assert model_table_rows_from_report({"metric": "counterfactual_planning_gap"}) == []


def test_varied_init_is_a_separate_row() -> None:
    fixed = model_table_rows_from_report(_single_report())[0]
    varied = model_table_rows_from_report(_single_report(varied_init=True))[0]
    assert fixed.init == "fixed"
    assert varied.init == "varied"
    deduped = dedupe_model_table_rows([fixed, varied])
    assert len(deduped) == 2  # never collapsed into one cell


def test_pooled_report_without_stamp_is_varied() -> None:
    """The committed pooled reports predate the varied_init stamp but pool
    varied-init seed runs; they must land in the varied column, not fixed."""
    report = _single_report(pooling={"seeds": [0, 1, 2], "n_total": 30})
    del report["varied_init"]
    assert model_table_rows_from_report(report)[0].init == "varied"


def test_tdmpc2_capacity_variants_get_distinct_cells() -> None:
    size1 = model_table_rows_from_report(
        _single_report(training={"tdmpc2_model_size": 1})
    )[0]
    size5_field = model_table_rows_from_report(
        _single_report(training={"tdmpc2_model_size": 5})
    )[0]
    size5_filename = model_table_rows_from_report(
        _single_report(), source="results/dmc_cartpole/cem_cpg_size5_pooled.json"
    )[0]
    assert size1.model == "dreamerv3"  # default capacity stays unsuffixed
    assert size5_field.model == "dreamerv3 (size=5)"
    assert size5_filename.model == "dreamerv3 (size=5)"
    assert len(dedupe_model_table_rows([size1, size5_field])) == 2


def test_single_arm_mlp_world_model_is_renamed_by_data_source() -> None:
    """The coverage reports and the CEM reports name the same arm differently;
    both must render as mlp_on_<source>_data."""
    report = _single_report(learned_model="mlp_world_model", data_source="tdmpc2")
    assert model_table_rows_from_report(report)[0].model == "mlp_on_tdmpc2_data"


def test_fallback_model_name_uses_policy_parenthetical() -> None:
    report = _single_report()
    del report["learned_model"]
    report["learned_scorecard"] = {
        "policy_name": "tabular-world-model (learned dynamics)"
    }
    assert model_table_rows_from_report(report)[0].model == "learned"


def test_dedupe_keeps_highest_n_per_cell() -> None:
    seed_row = model_table_rows_from_report(
        _single_report(cpg=_cpg_dict(n=10)), source="seed0.json"
    )[0]
    pooled_row = model_table_rows_from_report(
        _single_report(cpg=_cpg_dict(n=30)), source="pooled.json"
    )[0]
    deduped = dedupe_model_table_rows([seed_row, pooled_row, seed_row])
    assert len(deduped) == 1
    assert deduped[0].n_per_arm == 30
    assert deduped[0].source == "pooled.json"


def test_markdown_table_renders_every_row_and_signs_the_gap() -> None:
    rows = [
        ModelTableRow(
            environment="dmc_acrobot_swingup",
            model="dreamerv3",
            planner="random-shooting",
            init="fixed",
            n_per_arm=150,
            oracle_success_rate=0.9,
            learned_success_rate=0.1,
            gap=0.8,
            gap_ci_low=0.7,
            gap_ci_high=0.9,
            verdict="MODEL BOTTLENECK",
        ),
        ModelTableRow(
            environment="dmc_cartpole_swingup",
            model="tdmpc2",
            planner="cem",
            init="varied",
            n_per_arm=30,
            oracle_success_rate=0.5,
            learned_success_rate=0.77,
            gap=-0.27,
            gap_ci_low=-0.48,
            gap_ci_high=-0.02,
            verdict="LEARNED OUTPERFORMS ORACLE",
        ),
    ]
    md = to_markdown_model_table(rows, heading="Multi-model CPG")
    assert "## Multi-model CPG" in md
    assert "| dmc_acrobot_swingup | dreamerv3 | random-shooting | fixed | 150 " in md
    assert "+0.800" in md
    assert "-0.270" in md
    assert "[-0.480, -0.020]" in md
    assert "LEARNED OUTPERFORMS ORACLE" in md
    # header + separator + 2 rows
    assert len([line for line in md.strip().splitlines() if line.startswith("|")]) == 4


def test_generator_ingests_the_committed_results() -> None:
    """Integration: every committed result JSON must parse, and the known
    published arms must survive extraction and dedupe."""
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    rows = []
    for path in sorted(results_dir.rglob("*.json")):
        report = json.loads(path.read_text())
        if isinstance(report, dict):
            rows.extend(
                model_table_rows_from_report(report, source=str(path.relative_to(repo_root)))
            )
    deduped = dedupe_model_table_rows(rows)

    assert len(deduped) >= 5
    models = {r.model for r in deduped}
    assert "tdmpc2" in models
    # capacity variants must not collide into one cell
    assert "tdmpc2 (size=5)" in models
    # the coverage arm and the CEM arm of the same MLP share one name
    assert "mlp_on_tdmpc2_data" in models
    assert "mlp_world_model" not in models
    environments = {r.environment for r in deduped}
    assert {"dmc_acrobot_swingup", "dmc_cartpole_swingup", "dmc_reacher_easy"} <= environments
    # The pooled reports aggregate the varied-init seed runs, so dedupe must
    # land them in the varied column at the pooled n, superseding the n=10
    # single-seed runs of the same cell.
    cartpole_tdmpc2 = [
        r for r in deduped
        if r.environment == "dmc_cartpole_swingup"
        and r.model == "tdmpc2"
        and r.planner == "random-shooting"
    ]
    assert cartpole_tdmpc2 == [r for r in cartpole_tdmpc2 if r.init == "varied"]
    assert all(r.n_per_arm >= 30 for r in cartpole_tdmpc2)
