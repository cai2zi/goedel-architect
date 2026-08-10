"""Whole-file, side-by-side candidate diffs for the readonly reviewer."""
from __future__ import annotations

import difflib
from typing import Any


def whole_file_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Return aligned full-file lines with VS-Code-like add/delete classes."""
    before = str(left.get("lean", "")).splitlines()
    after = str(right.get("lean", "")).splitlines()
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    rows: list[dict[str, Any]] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(left_end - left_start):
                rows.append({"left": {"line": left_start + offset + 1, "text": before[left_start + offset], "kind": "unchanged"},
                             "right": {"line": right_start + offset + 1, "text": after[right_start + offset], "kind": "unchanged"}})
        elif tag == "delete":
            for index in range(left_start, left_end):
                rows.append({"left": {"line": index + 1, "text": before[index], "kind": "removed"}, "right": None})
        elif tag == "insert":
            for index in range(right_start, right_end):
                rows.append({"left": None, "right": {"line": index + 1, "text": after[index], "kind": "added"}})
        else:  # replace: retain alignment but mark the two sides independently.
            width = max(left_end - left_start, right_end - right_start)
            for offset in range(width):
                old = left_start + offset
                new = right_start + offset
                rows.append({"left": {"line": old + 1, "text": before[old], "kind": "removed"} if old < left_end else None,
                             "right": {"line": new + 1, "text": after[new], "kind": "added"} if new < right_end else None})
    return {"leftCandidateId": left.get("candidateId"), "rightCandidateId": right.get("candidateId"), "rows": rows}
