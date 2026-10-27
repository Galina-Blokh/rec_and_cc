"""docs/requirements: every story has a traceability row, every listed test exists, and every
test file declares its stories ("Stories: none." for tooling tests).

Stories: none.
"""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "docs" / "requirements"
TESTS = ROOT / "tests"


def story_ids() -> list[str]:
    return re.findall(r"^### (US-\d+) ", (REQ / "stories.md").read_text("utf-8"), re.M)


def summary_ids() -> list[str]:
    return re.findall(r"^\| (US-\d+) \|", (REQ / "stories.md").read_text("utf-8"), re.M)


def table_lines() -> list[tuple[str, str]]:
    return [
        (m[1], m[2])
        for line in (REQ / "traceability.md").read_text("utf-8").splitlines()
        if (m := re.match(r"\| (US-\d+) \|[^|]*\|(.*)\|$", line))
    ]


REF = re.compile(r"test_\w+\.py::(\*|test_\w+)")


def split_refs(cell: str) -> tuple[list[str], list[str]]:
    """(well-formed, malformed) backticked entries of a Tests cell."""
    refs = re.findall(r"`([^`]+)`", cell)
    return [r for r in refs if REF.fullmatch(r)], [r for r in refs if not REF.fullmatch(r)]


def table_cells() -> dict[str, str]:
    return dict(table_lines())


def trace_rows() -> dict[str, list[str]]:
    return {story: split_refs(cell)[0] for story, cell in table_cells().items()}


@cache
def parsed(file: str) -> ast.Module:
    return ast.parse((TESTS / file).read_text("utf-8"))


def functions_named_test(file: str) -> set[str]:
    return {
        n.name
        for n in parsed(file).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }


def declared_stories(file: str) -> set[str] | None:
    m = re.search(r"^Stories: (.+)\.$", ast.get_docstring(parsed(file)) or "", re.M)
    if not m:
        return None
    return set() if m[1] == "none" else set(m[1].split(", "))


def test_every_story_is_traced_and_ids_are_unique():
    ids = story_ids()
    assert ids, "no stories found"
    assert len(ids) == len(set(ids))
    assert summary_ids() == ids, "summary table and story sections differ"
    traced = [story for story, _ in table_lines()]
    assert len(traced) == len(set(traced)), "duplicate traceability rows"
    rows = trace_rows()
    assert sorted(rows) == sorted(ids)
    assert all(rows[i] for i in ids)


def test_split_refs_rejects_malformed_entries():
    good, bad = split_refs(
        "`test_a.py::test_x`, `test_a.py::*`, `test_a.py::test-x`, `sub/test_b.py::t`"
    )
    assert good == ["test_a.py::test_x", "test_a.py::*"]
    assert bad == ["test_a.py::test-x", "sub/test_b.py::t"]


def test_every_table_reference_is_well_formed():
    bad = {story: split_refs(cell)[1] for story, cell in table_cells().items()}
    assert not any(bad.values()), {s: b for s, b in bad.items() if b}


def test_traced_tests_exist():
    missing = []
    for refs in trace_rows().values():
        for ref in refs:
            file, func = ref.split("::")
            names = functions_named_test(file) if (TESTS / file).exists() else set()
            if not names if func == "*" else func not in names:
                missing.append(ref)
    assert not missing, missing


def test_every_test_file_declares_the_stories_it_is_traced_to():
    traced: dict[str, set[str]] = {}
    for story, refs in trace_rows().items():
        for ref in refs:
            traced.setdefault(ref.split("::")[0], set()).add(story)
    wrong = {}
    for path in sorted(TESTS.glob("test_*.py")):
        declared = declared_stories(path.name)
        if declared is None:
            wrong[path.name] = "no 'Stories:' docstring line (use 'Stories: none.' for tooling)"
        elif declared != traced.get(path.name, set()):
            wrong[path.name] = (
                f"docstring {sorted(declared)} != table {sorted(traced.get(path.name, set()))}"
            )
    assert not wrong, wrong
