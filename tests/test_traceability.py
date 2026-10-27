"""docs/requirements: every story has a traceability row, and every listed test exists."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "docs" / "requirements"
TESTS = ROOT / "tests"


def story_ids() -> list[str]:
    return re.findall(r"^### (US-\d+) ", (REQ / "stories.md").read_text("utf-8"), re.M)


def trace_rows() -> dict[str, list[str]]:
    rows = {}
    for line in (REQ / "traceability.md").read_text("utf-8").splitlines():
        m = re.match(r"\| (US-\d+) \|[^|]*\|(.*)\|$", line)
        if m:
            rows[m[1]] = re.findall(r"`(test_\w+\.py::[\w*]+)`", m[2])
    return rows


def defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_every_story_is_traced_and_ids_are_unique():
    ids = story_ids()
    assert ids, "no stories found"
    assert len(ids) == len(set(ids))
    rows = trace_rows()
    assert sorted(rows) == sorted(ids)
    assert all(rows[i] for i in ids)


def test_traced_tests_exist():
    missing = []
    for refs in trace_rows().values():
        for ref in refs:
            file, func = ref.split("::")
            path = TESTS / file
            if not path.exists() or (func != "*" and func not in defined_functions(path)):
                missing.append(ref)
    assert not missing, missing


def test_test_docstrings_name_their_stories():
    want: dict[str, set[str]] = {}
    for story, refs in trace_rows().items():
        for ref in refs:
            want.setdefault(ref.split("::")[0], set()).add(story)
    for file, stories in want.items():
        doc = ast.get_docstring(ast.parse((TESTS / file).read_text("utf-8"))) or ""
        m = re.search(r"^Stories: (.+)\.$", doc, re.M)
        assert m, f"{file}: docstring has no 'Stories:' line"
        assert set(m[1].split(", ")) == stories, file
