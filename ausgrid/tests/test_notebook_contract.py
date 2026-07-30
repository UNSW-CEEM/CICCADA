from __future__ import annotations

import json
from pathlib import Path


def test_foundation_notebook_has_stages_visuals_and_full_run_gate() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "01_foundation_pipeline.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    code = "\n".join("".join(cell["source"]) for cell in code_cells)

    for stage in (
        "Stage 0 — source inventory",
        "Stage 1 — canonical metadata",
        "Stage 2 — duplicate-key audit",
        "Stage 3 — canonical phase telemetry",
        "Stage 4 — exact row accounting",
        "Optional full-dataset build",
    ):
        assert stage in markdown

    assert 'FULL_RUN_CONFIRMATION == "RUN FULL DATASET"' in code
    assert "plt.subplots" in code
    assert "display(" in code
    assert "assert sample_decision" in code

    for index, cell in enumerate(code_cells):
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
