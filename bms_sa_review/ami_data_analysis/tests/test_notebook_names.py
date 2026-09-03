"""
Every name a notebook uses must be bound earlier in that same notebook.

Ported verbatim in spirit from ``oem_analysis/tests/test_notebook_names.py``,
which exists because of a real failure: a section was appended to
``04_bom_extract.ipynb`` without adding its module to the import cell, so the
notebook raised ``NameError`` on first run -- after the expensive Athena extract
had already completed.

That failure mode is worse here, not better. These notebooks are run by hand,
one at a time, against a paid query engine, and Phase 4 is a long resumable
extract. A ``NameError`` three cells after an UNLOAD is exactly the thing this
test is for.

Static check, no execution: parse each code cell, collect what it binds, and
flag any load of a name that nothing earlier bound. Deliberately conservative --
it only reports names never bound anywhere before use, so notebooks are still
free to depend on execution order.
"""

from __future__ import annotations

import ast
import builtins
import pathlib

import pytest

try:
    import nbformat
except ImportError:  # pragma: no cover
    nbformat = None

NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parents[1] / "notebooks"

#: Scratchpads, not pipeline notebooks.
EXCLUDE = {"draft.ipynb"}

#: IPython injects these; no cell defines them.
IPYTHON_GLOBALS = {"display", "get_ipython", "In", "Out", "exit", "quit"}

SAFE_NAMES = set(dir(builtins)) | IPYTHON_GLOBALS


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name this tree binds: assignment, import, def, arg, except-as."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, ast.alias):
            out.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, ast.Global):
            out.update(node.names)
    return out


def _code_source(cell) -> str:
    """Cell source with IPython magics and shell escapes stripped."""
    return "\n".join(
        line for line in cell.source.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )


def _notebooks() -> list[pathlib.Path]:
    if not NOTEBOOK_DIR.is_dir():
        return []
    return sorted(p for p in NOTEBOOK_DIR.glob("*.ipynb") if p.name not in EXCLUDE)


@pytest.mark.skipif(nbformat is None, reason="nbformat not installed")
@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.name)
def test_no_undefined_names(path: pathlib.Path) -> None:
    nb = nbformat.read(path, as_version=4)

    seen: set[str] = set()
    problems: list[str] = []

    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(_code_source(cell))
        except SyntaxError as exc:
            problems.append(f"cell {index}: syntax error -- {exc.msg}")
            continue

        here = _bound_names(tree)
        used = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for name in sorted(used - seen - here - SAFE_NAMES):
            problems.append(f"cell {index}: `{name}` used but never bound earlier")
        seen |= here

    assert not problems, (
        f"{path.name} references names nothing defines:\n  "
        + "\n  ".join(problems)
        + "\n\nUsually a cell was appended without updating the import cell."
    )


@pytest.mark.skipif(nbformat is None, reason="nbformat not installed")
def test_notebooks_are_discoverable() -> None:
    """Guard the guard: a bad glob would make every test above vacuously pass."""
    found = _notebooks()
    assert found, f"no notebooks found in {NOTEBOOK_DIR}"
    assert any(p.name.startswith("00_") for p in found), (
        f"expected the numbered pipeline notebooks, found {[p.name for p in found]}"
    )
