from __future__ import annotations

import json
from pathlib import Path


def test_mechanism_notebook_is_clean_staged_and_hard_gated() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "04_build_mechanism_results.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    for index, cell in enumerate(code_cells):
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"mechanism-cell-{index}", "exec")

    for stage in (
        "Stage 0",
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4",
        "Stage 5",
        "Stage 6",
        "Optional full-dataset build",
    ):
        assert stage in markdown
    assert "SIGN REVIEW COMPLETE" in code
    assert "RUN MECHANISM RESULTS FULL" in code
    assert "FULL_RUN_CONFIRMATION" in code
    assert "display(" in code
    assert "plt.subplots" in code
    assert "assert " in code
    assert "build_counterfactual" not in code
    assert "curtailment is intentionally not built" in markdown.lower()
