from __future__ import annotations

import json
from pathlib import Path


def test_results_notebook_is_clean_staged_and_hard_gated() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "05_analyse_mechanism_results.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    for index, cell in enumerate(code_cells):
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"results-cell-{index}", "exec")

    for stage in (
        "Stage 0",
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4",
        "Stage 5",
        "Stage 6",
        "Stage 7",
        "Stage 8",
        "Stage 9",
        "Stage 10",
        "Stage 11",
        "Stage 12",
    ):
        assert stage in markdown, f"missing {stage} header"

    # Exact deliberate full-analysis confirmation.
    assert "FULL_ANALYSIS_CONFIRMATION" in code
    assert "RUN FULL RESULTS ANALYSIS" in code

    # Calls into result_views/result_plots exist -- this notebook is an
    # orchestrator over those modules, not a reimplementation.
    assert "result_views" in code or "import ausgrid_analysis.result_views" in code
    assert "result_plots" in code or "import ausgrid_analysis.result_plots" in code
    assert "rv." in code
    assert "rp." in code

    # Assertions and displays/plots exist throughout, not just at the end.
    assert code.count("assert ") >= 10
    assert "display(" in code
    assert "plt.show()" in code or "plt.subplots" in code

    # This notebook never rebuilds a Delivery 4 result table.
    for forbidden in (
        "build_voltvar_results",
        "build_voltwatt_results",
        "build_response_observability",
    ):
        assert forbidden not in code, f"results notebook must never call {forbidden}"

    # No full raw/canonical pandas load: the notebook must only read the
    # three bounded Delivery 4 result tables through result_views.py, never
    # touch canonical_phase or the raw telemetry parquet directly.
    assert "canonical_phase" not in code
    assert "telemetry_parquet" not in code
    assert "read_parquet(" not in code

    # No local timestamp used as a uniqueness/grouping key anywhere.
    assert "timestamp_local" not in code

    # No counterfactual/curtailment calculation -- only the fixed
    # unavailable panel.
    assert "build_counterfactual" not in code
    assert "curtailment_energy" not in code
    assert "curtailment_rate" not in code
    assert "plot_curtailment_unavailable" in code
    assert "gate 7" in markdown.lower() or "gate-7" in markdown.lower()

    # No blended fleet score anywhere in code; markdown may only discuss the
    # concept to explicitly disclaim it.
    assert "score" not in code.lower()
    assert "blended" not in code.lower()

    # 'pass'/'fail'/'conforming' must never be used as a proxy or
    # observability label; the only legitimate use of these words in this
    # notebook is view/gate reconciliation status ('status', 'passed'),
    # not a per-row magnitude classification.
    assert "'conforming'" not in code
    assert '"conforming"' not in code

    # Both phase_scope_basis tracks are inspected, per the explicit user
    # decision to extend this delivery beyond the original single-track
    # acceptance spec.
    assert "der_inferred" in code
    assert "all_phases" in code
