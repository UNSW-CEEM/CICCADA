"""
Every name a notebook uses must be bound earlier in that same notebook.

Exists because of a real failure: section 5 was appended to
``04_bom_extract.ipynb`` without adding ``se_counterfactual`` to the import cell,
so the notebook raised ``NameError: name 'ctf' is not defined`` on first run --
after the expensive Athena extract had already completed. ``config`` was missing
too and would have been the next error.

Appending cells to a notebook is exactly the edit that breaks this, and it is
invisible in review because the new cell looks self-contained. Static check, no
execution: parse each code cell, collect what it binds, and flag any load of a
name that nothing earlier bound.

Deliberately conservative -- it only reports names that are never bound anywhere
before use, so a cell that defines and uses a variable in one go is fine, and
notebooks are still free to depend on execution order.
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

#: `draft.ipynb` is the user's scratchpad, not a pipeline notebook.
EXCLUDE = {"draft.ipynb"}

#: IPython injects these; they are not defined by any cell.
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
    assert len(found) >= 8, f"expected the 00-07 pipeline notebooks, found {found}"
