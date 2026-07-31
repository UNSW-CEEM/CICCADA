from __future__ import annotations

import json
from pathlib import Path


def test_new_notebooks_are_clean_compilable_and_gated() -> None:
    root = Path(__file__).resolve().parents[1]
    sources: dict[str, str] = {}
    for name in (
        "03a_build_analysis_cohort.ipynb",
        "03b_assess_irradiance_coverage.ipynb",
    ):
        notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        sources[name] = "\n".join("".join(cell["source"]) for cell in code_cells)
        for index, cell in enumerate(code_cells):
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"{name}-cell-{index}", "exec")

    coverage = sources["03b_assess_irradiance_coverage.ipynb"]
    assert "RUN BOM COVERAGE AUDIT" in coverage
    assert "ACCEPT IRRADIANCE MAPPING FOR DECOMPOSITION EXPERIMENTS" in coverage
    assert "database='bom_nci'" in coverage
