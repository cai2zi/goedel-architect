"""Mechanical boundary anchors and lossless source-unit reconstruction."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


OPEN_MARKER = "[[SOURCE_UNITS_V1]]"
CLOSE_MARKER = "[[/SOURCE_UNITS_V1]]"
_ANCHOR_RE = re.compile(r"B[0-9]{4,}")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|\*\*(?:step|case|part|answer|conclusion)\b)", re.I,
)
_LIST_RE = re.compile(r"^\s*(?:[-+*]|\d{1,3}[.)])\s+")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")


class SourceUnitError(ValueError):
    """A splitter response or reconstructed partition is invalid."""


def _protected_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for pattern in (
        re.compile(r"`+[^`]*`+"),
        re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.S),
        re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]", re.S),
    ):
        ranges.extend((match.start(), match.end()) for match in pattern.finditer(text))
    return ranges


def _sentence_starts(line: str) -> list[int]:
    protected = _protected_ranges(line)
    starts = [0]
    for index, char in enumerate(line):
        if char not in ".?!;。！？；":
            continue
        if any(start <= index < end for start, end in protected):
            continue
        if (
            char == "."
            and index > 0
            and index + 1 < len(line)
            and line[index - 1].isdigit()
            and line[index + 1].isdigit()
        ):
            continue
        next_index = index + 1
        while next_index < len(line) and line[next_index].isspace():
            next_index += 1
        if next_index < len(line) and next_index > index + 1:
            starts.append(next_index)
    return sorted(set(starts))


def make_boundary_anchors(source: str) -> list[dict[str, Any]]:
    """Create dense coordinates whose text concatenates to the exact source."""
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    lines = source.splitlines(keepends=True) or [source]
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    seeds: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        body = lines[index].rstrip("\r\n")
        start = offsets[index]
        if not body.strip() or _SEPARATOR_RE.fullmatch(body):
            index += 1
            continue
        fence = _FENCE_RE.match(body)
        if fence:
            marker = fence.group(1)
            final = index
            for candidate in range(index + 1, len(lines)):
                close = _FENCE_RE.match(lines[candidate].rstrip("\r\n"))
                if close and close.group(1)[0] == marker[0]:
                    final = candidate
                    break
            seeds.append((start, "code_block"))
            index = final + 1
            continue
        stripped = body.strip()
        if _HEADING_RE.match(body):
            kind, starts = "heading", [0]
        elif _TABLE_RE.match(body):
            kind, starts = "table_row", [0]
        elif _LIST_RE.match(body):
            kind, starts = "list_item", [0]
        elif (
            (stripped.startswith("$$") and stripped.endswith("$$"))
            or (stripped.startswith(r"\[") and stripped.endswith(r"\]"))
        ):
            kind, starts = "display_math", [0]
        else:
            kind, starts = "prose", _sentence_starts(body)
        seeds.extend((start + relative, kind) for relative in starts)
        index += 1

    if not seeds:
        seeds = [(0, "text")]
    seeds = sorted(set(seeds))
    if seeds[0][0] != 0:
        seeds[0] = (0, seeds[0][1])
    width = max(4, len(str(len(seeds))))
    anchors: list[dict[str, Any]] = []
    for number, (start, kind) in enumerate(seeds, start=1):
        end = seeds[number][0] if number < len(seeds) else len(source)
        anchors.append({
            "anchor_id": f"B{number:0{width}d}",
            "kind": kind,
            "source_start": start,
            "source_end": end,
            "source_text": source[start:end],
        })
    if "".join(item["source_text"] for item in anchors) != source:
        raise AssertionError("boundary anchors do not reconstruct the exact COT")
    return anchors


def parse_boundaries(
    content: str,
    anchors: Sequence[Mapping[str, Any]],
    *,
    max_units: int | None = None,
) -> list[str]:
    """Parse one strict marker block containing increasing end boundaries."""
    if max_units is not None and max_units < 1:
        raise ValueError("max_units must be positive")
    raw = str(content or "")
    if raw.count(OPEN_MARKER) != 1 or raw.count(CLOSE_MARKER) != 1:
        raise SourceUnitError("response must contain exactly one marker block")
    start = raw.find(OPEN_MARKER) + len(OPEN_MARKER)
    end = raw.find(CLOSE_MARKER)
    if end < start:
        raise SourceUnitError("marker block is misordered")
    prefix, suffix = raw[: raw.find(OPEN_MARKER)], raw[end + len(CLOSE_MARKER):]
    if prefix.strip() or suffix.strip():
        raise SourceUnitError("response must contain only the marker block")
    values = [line.strip() for line in raw[start:end].splitlines() if line.strip()]
    if not values or any(not _ANCHOR_RE.fullmatch(value) for value in values):
        raise SourceUnitError("marker block contains an invalid boundary")
    ids = [str(anchor["anchor_id"]) for anchor in anchors]
    positions = {value: index for index, value in enumerate(ids)}
    if any(value not in positions for value in values):
        raise SourceUnitError("response contains an unknown boundary")
    indices = [positions[value] for value in values]
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise SourceUnitError("boundaries must be unique and strictly increasing")
    if values[-1] != ids[-1]:
        raise SourceUnitError(f"final boundary must be {ids[-1]}")
    if max_units is not None and len(values) > max_units:
        raise SourceUnitError(
            f"source unit count must be at most {max_units}; received {len(values)}"
        )
    for value in values[:-1]:
        anchor = anchors[positions[value]]
        if str(anchor.get("kind")) == "heading":
            raise SourceUnitError(f"heading-only boundary is forbidden: {value}")
        if str(anchor.get("source_text") or "").rstrip().endswith(":"):
            raise SourceUnitError(f"colon lead-in boundary is forbidden: {value}")
    return values


def source_units_from_boundaries(
    source: str,
    anchors: Sequence[Mapping[str, Any]],
    boundaries: Sequence[str],
) -> list[dict[str, Any]]:
    """Construct exact, adjacent S-units from selected end boundaries."""
    positions = {str(anchor["anchor_id"]): index for index, anchor in enumerate(anchors)}
    units: list[dict[str, Any]] = []
    first_anchor = 0
    for number, boundary in enumerate(boundaries, start=1):
        if boundary not in positions:
            raise SourceUnitError(f"unknown boundary: {boundary}")
        final_anchor = positions[boundary]
        if final_anchor < first_anchor:
            raise SourceUnitError("boundaries overlap or are out of order")
        source_start = int(anchors[first_anchor]["source_start"])
        source_end = int(anchors[final_anchor]["source_end"])
        units.append({
            "unit_id": f"S{number:03d}",
            "source_start": source_start,
            "source_end": source_end,
            "source_text": source[source_start:source_end],
        })
        first_anchor = final_anchor + 1
    if first_anchor != len(anchors):
        raise SourceUnitError("boundaries omit trailing anchors")
    if "".join(unit["source_text"] for unit in units) != source:
        raise SourceUnitError("source units do not reconstruct the exact COT")
    for left, right in zip(units, units[1:]):
        if left["source_end"] != right["source_start"]:
            raise SourceUnitError("source units are not adjacent")
    return units
