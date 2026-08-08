from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openai import AsyncOpenAI


SPLITTER_VERSION = "llm-cot-boundary-v6"
PROMPT_VERSION = "cot-split-boundaries-v6"
ATOMIZER_VERSION = "cot-atomizer-v1"
NORMALIZER_VERSION = "boundary-normalizer-v6.1"
RESULTS_FILENAME = "llm_cot_splits.jsonl"
SUMMARY_FILENAME = "llm_cot_split_summary.json"

_OPEN_MARKER = "[[COT_SPLIT_V1]]"
_CLOSE_MARKER = "[[/COT_SPLIT_V1]]"
_ATOM_ID_RE = re.compile(r"A[0-9]{4,}")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")
_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|\*\*(?:step|case|part|answer|conclusion)\b)",
    re.IGNORECASE,
)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-+*]|\d{1,3}[.)])\s+")
_PURE_NUMBERED_BOLD_TITLE_RE = re.compile(
    r"^\s*(?:"
    r"\*\*\s*(?:\(?\d{1,3}\)?[.):]|[IVXLC]+[.):])\s+[^\n]+?\*\*"
    r"|(?:\(?\d{1,3}\)?[.)]|[IVXLC]+[.)])\s+\*\*[^\n]+?\*\*"
    r")\s*:?\s*$",
    re.IGNORECASE,
)
_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "mr.", "mrs.", "ms.", "dr.", "prof.",
    "eq.", "fig.", "no.", "vs.", "w.l.o.g.",
}


class SplitFormatError(ValueError):
    """The model response is not a valid lossless boundary annotation."""

    def __init__(self, reason: str, raw_content: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_content = raw_content


@dataclass(frozen=True)
class LLMCotSplitterConfig:
    model: str
    openai_base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "EMPTY"
    concurrency: int = 24
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 600.0
    max_format_attempts: int = 2
    enable_thinking: bool = False
    atomizer_version: str = ATOMIZER_VERSION
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("splitter model must be non-empty")
        if self.concurrency < 1:
            raise ValueError("splitter concurrency must be >= 1")
        if self.max_tokens < 1:
            raise ValueError("splitter max_tokens must be >= 1")
        if self.timeout_s <= 0:
            raise ValueError("splitter timeout_s must be > 0")
        if not 1 <= self.max_format_attempts <= 2:
            raise ValueError("max_format_attempts must be 1 or 2")
        if self.atomizer_version != ATOMIZER_VERSION:
            raise ValueError(
                f"configured atomizer_version={self.atomizer_version!r} does not match "
                f"implementation {ATOMIZER_VERSION!r}"
            )
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"configured prompt_version={self.prompt_version!r} does not match "
                f"implementation {PROMPT_VERSION!r}"
            )

    @classmethod
    def from_value(
        cls,
        value: "LLMCotSplitterConfig | Mapping[str, Any] | Any",
    ) -> "LLMCotSplitterConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            data = dict(value)
        else:
            names = (
                "model", "openai_base_url", "base_url", "api_key", "concurrency",
                "temperature", "max_tokens", "timeout_s", "max_format_attempts",
                "max_attempts", "enable_thinking", "atomizer_version",
                "prompt_version",
            )
            data = {name: getattr(value, name) for name in names if hasattr(value, name)}
        if "base_url" in data and "openai_base_url" not in data:
            data["openai_base_url"] = data.pop("base_url")
        if "max_attempts" in data and "max_format_attempts" not in data:
            data["max_format_attempts"] = data.pop("max_attempts")
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: data[key] for key in data if key in allowed})


@dataclass
class SplitResult:
    row_id: str
    status: str
    source_sha256: str
    cache_key: str
    prompt_content_sha256: str = ""
    boundaries: list[str] = field(default_factory=list)
    atoms: list[dict[str, Any]] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None
    error_type: str | None = None
    cached: bool = False
    normalizer_replayed: bool = False
    replayed_from_normalizer_version: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_artifact_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({
            "ID": self.row_id,
            "splitter_version": SPLITTER_VERSION,
            "prompt_version": PROMPT_VERSION,
            "atomizer_version": ATOMIZER_VERSION,
            "normalizer_version": NORMALIZER_VERSION,
            "attempt_count": len(self.attempts),
            "raw_responses": [
                {
                    "attempt": attempt.get("attempt"),
                    "content": attempt.get("raw_content"),
                    "reasoning_content": attempt.get("reasoning_content"),
                    "finish_reason": attempt.get("finish_reason"),
                    "response_id": attempt.get("response_id"),
                    "request_id": attempt.get("request_id"),
                }
                for attempt in self.attempts
                if attempt.get("raw_content") is not None
            ],
            "finished_at": _utc_now(),
        })
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stripped_bounds(source: str) -> tuple[int, int]:
    left = len(source) - len(source.lstrip())
    right = len(source.rstrip())
    return left, right


def _line_body(line: str) -> str:
    return line.rstrip("\r\n")


def _protected_inline_ranges(text: str) -> list[tuple[int, int]]:
    """Return spans in which sentence punctuation must not create an atom."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index] == "`":
            width = 1
            while index + width < len(text) and text[index + width] == "`":
                width += 1
            close = text.find("`" * width, index + width)
            if close >= 0:
                spans.append((index, close + width))
                index = close + width
                continue
        if text.startswith(r"\(", index):
            close = text.find(r"\)", index + 2)
            if close >= 0:
                spans.append((index, close + 2))
                index = close + 2
                continue
        if text.startswith(r"\[", index):
            close = text.find(r"\]", index + 2)
            if close >= 0:
                spans.append((index, close + 2))
                index = close + 2
                continue
        if text[index] == "$" and (index == 0 or text[index - 1] != "\\"):
            width = 2 if text.startswith("$$", index) else 1
            needle = "$" * width
            close = text.find(needle, index + width)
            if close >= 0:
                spans.append((index, close + width))
                index = close + width
                continue
        index += 1
    return spans


def _inside_ranges(index: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in ranges)


def _looks_like_abbreviation(text: str, period_index: int) -> bool:
    prefix = text[:period_index + 1].lower()
    return any(prefix.endswith(value) for value in _ABBREVIATIONS)


def _sentence_starts(line_body: str) -> list[int]:
    protected = _protected_inline_ranges(line_body)
    starts = [0]
    for index, char in enumerate(line_body):
        if char not in ".?!;。！？；" or _inside_ranges(index, protected):
            continue
        if char == "." and _looks_like_abbreviation(line_body, index):
            continue
        next_index = index + 1
        while next_index < len(line_body) and line_body[next_index].isspace():
            next_index += 1
        if next_index >= len(line_body):
            continue
        # ASCII full stops split only at a conventional sentence boundary;
        # this avoids decimal numbers and most dotted identifiers. Other
        # punctuation is unambiguous enough to split without this restriction.
        if char == "." and next_index == index + 1:
            continue
        if next_index not in starts:
            starts.append(next_index)
    return starts


def _display_block_end(lines: Sequence[str], start: int) -> int | None:
    """Return the inclusive final line of a multiline math block, if any."""
    body = _line_body(lines[start])
    if body.count("$$") % 2 == 1:
        for index in range(start + 1, len(lines)):
            if _line_body(lines[index]).count("$$") % 2 == 1:
                return index
        return len(lines) - 1
    if r"\[" in body and r"\]" not in body[body.find(r"\[") + 2:]:
        for index in range(start + 1, len(lines)):
            if r"\]" in _line_body(lines[index]):
                return index
        return len(lines) - 1
    begin = re.search(r"\\begin\{([^}]+)\}", body)
    if begin:
        close = rf"\end{{{begin.group(1)}}}"
        if close not in body[begin.end():]:
            for index in range(start + 1, len(lines)):
                if close in _line_body(lines[index]):
                    return index
            return len(lines) - 1
    return None


def atomize_cot(source: str) -> list[dict[str, Any]]:
    """Create immutable, contiguous source atoms without interpreting the COT.

    The atoms cover exactly ``source.strip()``. Sentence splitting is used only
    for ordinary prose. Fenced code, multiline display math, inline math and
    Markdown structural lines are never split internally.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    left, right = _stripped_bounds(source)
    if left >= right:
        return []

    core = source[left:right]
    lines = core.splitlines(keepends=True)
    line_starts: list[int] = []
    cursor = left
    for line in lines:
        line_starts.append(cursor)
        cursor += len(line)

    seeds: list[tuple[int, str]] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        body = _line_body(line)
        if not body.strip() or _SEPARATOR_RE.fullmatch(body):
            line_index += 1
            continue
        absolute_start = line_starts[line_index]

        fence = _FENCE_RE.match(body)
        if fence:
            marker = fence.group(1)
            final = line_index
            for candidate in range(line_index + 1, len(lines)):
                close = re.match(r"^\s*(`{3,}|~{3,})", _line_body(lines[candidate]))
                if close and close.group(1)[0] == marker[0] and len(close.group(1)) >= len(marker):
                    final = candidate
                    break
            else:
                final = len(lines) - 1
            seeds.append((absolute_start, "code_block"))
            line_index = final + 1
            continue

        display_end = _display_block_end(lines, line_index)
        if display_end is not None:
            seeds.append((absolute_start, "display_math"))
            line_index = display_end + 1
            continue

        stripped = body.strip()
        if _TABLE_ROW_RE.match(body):
            kind = "table_row"
            relative_starts = [0]
        elif _HEADING_RE.match(body):
            kind = "heading"
            relative_starts = [0]
        elif _LIST_ITEM_RE.match(body):
            kind = "list_item"
            relative_starts = [0]
        elif (
            (stripped.startswith("$$") and stripped.endswith("$$"))
            or (stripped.startswith(r"\[") and stripped.endswith(r"\]"))
            or (stripped.startswith(r"\(") and stripped.endswith(r"\)"))
        ):
            kind = "display_math"
            relative_starts = [0]
        else:
            kind = "prose"
            relative_starts = _sentence_starts(body)
        for relative_start in relative_starts:
            seeds.append((absolute_start + relative_start, kind))
        line_index += 1

    if not seeds:
        seeds = [(left, "text")]
    if seeds[0][0] != left:
        # Global leading whitespace has already been stripped. This case is
        # possible only for an indented first non-empty line; keep indentation
        # inside the first immutable atom.
        seeds[0] = (left, seeds[0][1])

    width = max(4, len(str(len(seeds))))
    atoms: list[dict[str, Any]] = []
    for index, (start, kind) in enumerate(seeds):
        end = seeds[index + 1][0] if index + 1 < len(seeds) else right
        if end <= start:
            raise AssertionError("atomizer produced an empty or overlapping atom")
        text = source[start:end]
        atoms.append({
            "atom_id": f"A{index + 1:0{width}d}",
            "source_start": start,
            "source_end": end,
            "source_text": text,
            "source_sha256": _sha256(text),
            "kind": kind,
        })
    _validate_atoms(source, atoms)
    return atoms


def _validate_atoms(source: str, atoms: Sequence[Mapping[str, Any]]) -> None:
    left, right = _stripped_bounds(source)
    if left >= right:
        if atoms:
            raise ValueError("empty source must not have atoms")
        return
    if not atoms:
        raise ValueError("non-empty source has no atoms")
    expected_start = left
    seen: set[str] = set()
    for atom in atoms:
        atom_id = str(atom.get("atom_id") or "")
        if not _ATOM_ID_RE.fullmatch(atom_id):
            raise ValueError(f"invalid atom id: {atom_id!r}")
        if atom_id in seen:
            raise ValueError(f"duplicate atom id: {atom_id}")
        seen.add(atom_id)
        start = int(atom.get("source_start", -1))
        end = int(atom.get("source_end", -1))
        if start != expected_start or end <= start:
            raise ValueError(f"non-contiguous atom span at {atom_id}")
        actual = source[start:end]
        if actual != str(atom.get("source_text") or ""):
            raise ValueError(f"source text mismatch at {atom_id}")
        expected_sha = str(atom.get("source_sha256") or "")
        if expected_sha and expected_sha != _sha256(actual):
            raise ValueError(f"source hash mismatch at {atom_id}")
        expected_start = end
    if expected_start != right:
        raise ValueError("atoms do not cover the complete stripped source")
    if "".join(str(atom["source_text"]) for atom in atoms) != source.strip():
        raise ValueError("atom concatenation differs from stripped source")


def _validate_boundaries(
    boundaries: Sequence[str],
    atoms: Sequence[Mapping[str, Any]],
) -> list[str]:
    if not atoms:
        raise SplitFormatError("cannot split an empty atom inventory")
    if not boundaries:
        raise SplitFormatError("boundary list is empty")
    atom_ids = [str(atom.get("atom_id") or "") for atom in atoms]
    positions = {atom_id: index for index, atom_id in enumerate(atom_ids)}
    unknown = [boundary for boundary in boundaries if boundary not in positions]
    if unknown:
        raise SplitFormatError(f"unknown atom id(s): {','.join(unknown)}")
    indices = [positions[boundary] for boundary in boundaries]
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise SplitFormatError("boundary atom ids must be unique and strictly increasing")
    if boundaries[-1] != atom_ids[-1]:
        raise SplitFormatError(f"final boundary must be {atom_ids[-1]}")
    return list(boundaries)


def _canonicalize_structural_boundaries(
    boundaries: Sequence[str],
    atoms: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """Relocate presentation-only boundaries without changing source text.

    Headings and colon-led displays/lists/tables are mechanically incomplete
    steps.  Models often still place a boundary there even after an explicit
    instruction.  If prior substantive content is present, the boundary must
    move *before* the heading/intro so that it attaches to the following claim;
    merely deleting it would incorrectly merge two reasoning actions.  When a
    previous boundary is already there (or the prefix starts the document), the
    incomplete boundary is simply removed.  Both operations are deterministic,
    lossless and recorded for audit.
    """
    atom_ids = [str(atom.get("atom_id") or "") for atom in atoms]
    positions = {atom_id: index for index, atom_id in enumerate(atom_ids)}
    normalized: list[str] = []
    normalized_positions: list[int] = []
    warnings: list[str] = []

    def attach_prefix_to_next(index: int, boundary: str, warning_kind: str) -> None:
        candidate = index - 1
        candidate_text = (
            str(atoms[candidate].get("source_text") or "").strip()
            if candidate >= 0 else ""
        )
        candidate_is_complete = (
            candidate >= 0
            and str(atoms[candidate].get("kind") or "") != "heading"
            and _PURE_NUMBERED_BOLD_TITLE_RE.fullmatch(candidate_text) is None
        )
        if (
            candidate_is_complete
            and (not normalized_positions or candidate > normalized_positions[-1])
        ):
            shifted = atom_ids[candidate]
            normalized.append(shifted)
            normalized_positions.append(candidate)
            warnings.append(f"shifted_{warning_kind}_boundary:{boundary}->{shifted}")
        else:
            warnings.append(f"removed_{warning_kind}_boundary:{boundary}")

    for boundary in boundaries[:-1]:
        index = positions[boundary]
        atom = atoms[index]
        next_atom = atoms[index + 1]
        if (
            str(atom.get("kind") or "") == "heading"
            or _PURE_NUMBERED_BOLD_TITLE_RE.fullmatch(
                str(atom.get("source_text") or "").strip()
            )
        ):
            attach_prefix_to_next(index, boundary, "heading_only")
            continue
        if (
            str(atom.get("source_text") or "").rstrip().endswith(":")
            and str(next_atom.get("kind") or "")
            in {"display_math", "list_item", "table_row"}
        ):
            attach_prefix_to_next(
                index, boundary, f"intro_{next_atom.get('kind')}"
            )
            continue
        if (
            str(atom.get("kind") or "") == "table_row"
            and str(next_atom.get("kind") or "") == "table_row"
        ):
            next_cells = [
                cell.strip()
                for cell in str(next_atom.get("source_text") or "").strip().split("|")[1:-1]
            ]
            if next_cells and not next_cells[0]:
                warnings.append(
                    f"removed_table_continuation_boundary:{boundary}"
                )
                continue
        normalized.append(boundary)
        normalized_positions.append(index)
    normalized.append(boundaries[-1])
    return normalized, warnings


def parse_split_boundaries(
    content: str,
    atoms: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Parse the sole permitted output block and validate every boundary."""
    boundaries, _warnings = _parse_split_boundaries_with_warnings(content, atoms)
    return boundaries


def _parse_split_boundaries_with_warnings(
    content: str,
    atoms: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    raw = str(content or "")
    body, _prefix, _suffix = _split_block_parts(raw)
    lines = body.strip().splitlines()
    if not lines:
        raise SplitFormatError("COT_SPLIT_V1 block has no boundaries", raw)
    boundaries: list[str] = []
    for line in lines:
        value = line.strip()
        if not _ATOM_ID_RE.fullmatch(value):
            raise SplitFormatError(f"invalid boundary line: {line!r}", raw)
        boundaries.append(value)
    try:
        checked = _validate_boundaries(boundaries, atoms)
        return _canonicalize_structural_boundaries(checked, atoms)
    except SplitFormatError as exc:
        raise SplitFormatError(exc.reason, raw) from exc


def _split_block_parts(content: str) -> tuple[str, str, str]:
    """Extract one complete marker block while retaining surrounding text.

    The model is still instructed to emit no surrounding prose. We do not
    discard an otherwise valid annotation merely for that envelope violation:
    the caller records both the raw response and the outside text for audit.
    """
    raw = str(content or "")
    if raw.count(_OPEN_MARKER) != 1 or raw.count(_CLOSE_MARKER) != 1:
        raise SplitFormatError("response must contain exactly one COT_SPLIT_V1 block", raw)
    open_at = raw.find(_OPEN_MARKER)
    close_at = raw.find(_CLOSE_MARKER)
    body_start = open_at + len(_OPEN_MARKER)
    if close_at < body_start:
        raise SplitFormatError("COT_SPLIT_V1 markers are misordered", raw)
    return raw[body_start:close_at], raw[:open_at], raw[close_at + len(_CLOSE_MARKER):]


def sections_from_boundaries(
    source: str,
    atoms: Sequence[Mapping[str, Any]],
    boundaries: Sequence[str],
) -> list[dict[str, Any]]:
    """Reconstruct exact adjacent source slices selected by model boundaries."""
    _validate_atoms(source, atoms)
    checked = _validate_boundaries(boundaries, atoms)
    by_id = {str(atom["atom_id"]): index for index, atom in enumerate(atoms)}
    sections: list[dict[str, Any]] = []
    first_atom = 0
    for boundary in checked:
        final_atom = by_id[boundary]
        group = atoms[first_atom:final_atom + 1]
        start = int(group[0]["source_start"])
        end = int(group[-1]["source_end"])
        sections.append({
            "source_text": source[start:end],
            "source_start": start,
            "source_end": end,
            "atom_ids": [str(atom["atom_id"]) for atom in group],
            "atoms": [{
                "atom_id": str(atom["atom_id"]),
                "kind": str(atom.get("kind") or "text"),
                "source_start": int(atom["source_start"]) - start,
                "source_end": int(atom["source_end"]) - start,
            } for atom in group],
        })
        first_atom = final_atom + 1
    if first_atom != len(atoms):
        raise SplitFormatError("boundaries did not consume every atom")
    if "".join(section["source_text"] for section in sections) != source.strip():
        raise AssertionError("section reconstruction is not lossless")
    return sections


def build_split_messages(
    source: str,
    atoms: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    inventory = "\n".join(
        json.dumps(
            {
                "atom_id": atom["atom_id"],
                "kind": atom["kind"],
                "text": atom["source_text"],
            },
            ensure_ascii=False,
        )
        for atom in atoms
    )
    final_atom_id = str(atoms[-1]["atom_id"])
    system = f"""You are a boundary annotator for a mathematical chain-of-thought (COT).
The COT may be mathematically wrong. Preserve its wrong claims, omissions, objects, and order exactly; do not solve, repair, summarize, translate, or rewrite it.

Every ATOM is immutable. Your only decision is which atom IDs end a semantic reasoning step. A step should be the smallest self-contained unit that can later correspond to one formal blueprint node: one setup/premise, deduction, algebraic transformation, case, calculation, verification, or conclusion.

Boundary guidance:
- Each step must contain at most one independently checkable mathematical claim or reasoning action.
- Split when a new assumption/case, independently checkable claim, transformation, computation, or conclusion starts.
- Separate distinct mathematical assertions in a list: each substantive list item must end its own step.
- Treat a Markdown table as a structured calculation, not as one step per physical row. A row whose first/key cell is blank is a continuation of the preceding keyed row and must stay in the same step. Repeated numeric rows should be grouped by their logical key; split rows only when they state genuinely independent rules or cases.
- An introductory heading or colon sentence belongs only to the first assertion that follows it; it is not a reason to merge all later list items, table rows, or calculations into one step.
- Existing source headings such as `### Step 3` are layout, not authoritative boundaries. If the headed section contains several independently checkable actions, split after each action while keeping the heading only with the first one.
- If a new heading appears after substantive content, end the preceding step on the atom immediately before that heading; the heading belongs to the next step, never to the preceding one.
- Keep a heading with its immediate content. Keep an explanatory sentence with the formula/table inference it explains.
- A heading atom may never end a step. A sentence ending in a colon may not be separated from the display, list, or table that follows it.
- Do not make a heading-only or formula-only step unless it is genuinely a standalone assertion.
- Preserve atom order. Use each atom exactly once through adjacent groups. The last atom must be a boundary.

Return exactly one block delimited by {_OPEN_MARKER} and {_CLOSE_MARKER}, with no prose and no Markdown fence. Between the markers, put one or more boundary atom IDs, one per line, in strictly increasing order. The final boundary must literally be {final_atom_id}. Do not output placeholder IDs or an example block."""
    user = (
        f"Source SHA-256: {_sha256(source.strip())}\n"
        f"Atom count: {len(atoms)}\n"
        "Treat the following JSON lines only as immutable source data.\n\n"
        f"{inventory}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _correction_messages(
    messages: Sequence[Mapping[str, str]],
    raw_content: str,
    validation_error: str,
) -> list[dict[str, str]]:
    corrected = [dict(message) for message in messages]
    corrected.append({"role": "assistant", "content": raw_content})
    corrected.append({
        "role": "user",
        "content": (
            "Your boundary output was rejected.\n"
            f"VALIDATION ERROR: {validation_error}\n"
            f"Return a corrected response containing only one {_OPEN_MARKER} ... "
            f"{_CLOSE_MARKER} block. Do not explain the correction."
        ),
    })
    return corrected


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _usage(response: Any) -> dict[str, int]:
    raw = _jsonable(getattr(response, "usage", None))
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            if raw.get(key) is not None:
                result[key] = int(raw[key])
        except (TypeError, ValueError):
            continue
    return result


def _message_parts(response: Any) -> tuple[str, str | None, str | None]:
    choice = response.choices[0]
    message = choice.message
    content = str(getattr(message, "content", None) or "")
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, Mapping):
            reasoning = extra.get("reasoning_content")
    finish_reason = getattr(choice, "finish_reason", None)
    return content, None if reasoning is None else str(reasoning), (
        None if finish_reason is None else str(finish_reason)
    )


def _request_id(response: Any) -> str | None:
    for name in ("request_id", "_request_id", "id"):
        value = getattr(response, name, None)
        if value:
            return str(value)
    return None


def _aggregate_usage(attempts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for attempt in attempts:
        usage = attempt.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            try:
                total[str(key)] = total.get(str(key), 0) + int(value)
            except (TypeError, ValueError):
                continue
    return total


def _prompt_content_sha256(
    source: str,
    atoms: Sequence[Mapping[str, Any]],
) -> str:
    messages = build_split_messages(source, atoms)
    return _sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def split_cache_key(
    source: str,
    config: LLMCotSplitterConfig,
    atoms: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    atom_inventory = list(atoms) if atoms is not None else atomize_cot(source)
    payload = {
        "splitter_version": SPLITTER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "atomizer_version": ATOMIZER_VERSION,
        "prompt_content_sha256": _prompt_content_sha256(source, atom_inventory),
        "source_sha256": _sha256(source.strip()),
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "enable_thinking": config.enable_thinking,
    }
    return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _read_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"corrupt splitter artifact at line {line_number}: {exc}") from exc
            row_id = str(row.get("row_id") or row.get("ID") or "")
            if not row_id:
                raise RuntimeError(f"splitter artifact line {line_number} has no row ID")
            latest[row_id] = row
    return latest


def _result_from_cache(
    row: Mapping[str, Any],
    source: str,
    atoms: list[dict[str, Any]],
) -> SplitResult | None:
    if str(row.get("status") or "") != "ok":
        return None
    stored_boundaries = [str(value) for value in (row.get("boundaries") or [])]
    attempts = [
        dict(value) for value in (row.get("attempts") or [])
        if isinstance(value, Mapping)
    ]
    boundaries = stored_boundaries
    replayed_attempts = [dict(value) for value in attempts]
    for attempt_index in range(len(replayed_attempts) - 1, -1, -1):
        attempt = replayed_attempts[attempt_index]
        if str(attempt.get("status") or "") != "ok" or not attempt.get("raw_content"):
            continue
        try:
            boundaries, structural_warnings = _parse_split_boundaries_with_warnings(
                str(attempt["raw_content"]), atoms,
            )
        except (SplitFormatError, ValueError, TypeError):
            return None
        nonstructural = [
            str(value) for value in (attempt.get("format_warnings") or [])
            if str(value) == "content_outside_marker_block"
        ]
        attempt["format_warnings"] = [*structural_warnings, *nonstructural]
        break
    try:
        sections = sections_from_boundaries(source, atoms, boundaries)
    except (ValueError, TypeError):
        return None
    usage = row.get("usage") if isinstance(row.get("usage"), Mapping) else {}
    prior_normalizer = str(row.get("normalizer_version") or "unversioned")
    normalizer_replayed = (
        boundaries != stored_boundaries or prior_normalizer != NORMALIZER_VERSION
    )
    return SplitResult(
        row_id=str(row.get("row_id") or row.get("ID") or ""),
        status="ok",
        source_sha256=_sha256(source.strip()),
        cache_key=str(row.get("cache_key") or ""),
        prompt_content_sha256=_prompt_content_sha256(source, atoms),
        boundaries=boundaries,
        atoms=atoms,
        sections=sections,
        attempts=replayed_attempts,
        usage={str(key): int(value) for key, value in usage.items()},
        latency_s=float(row.get("latency_s") or 0.0),
        cached=True,
        normalizer_replayed=normalizer_replayed,
        replayed_from_normalizer_version=(
            prior_normalizer if normalizer_replayed else ""
        ),
    )


async def _split_one(
    *,
    client: Any,
    semaphore: asyncio.Semaphore,
    row_id: str,
    source: str,
    atoms: list[dict[str, Any]],
    config: LLMCotSplitterConfig,
    cache_key: str,
) -> SplitResult:
    messages = build_split_messages(source, atoms)
    prompt_content_sha256 = _sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True)
    )
    attempts: list[dict[str, Any]] = []
    start = time.monotonic()
    final_status = "format_error"
    final_error = "splitter returned no response"
    final_error_type = "SplitFormatError"
    async with semaphore:
        request_messages = messages
        for attempt_number in range(1, config.max_format_attempts + 1):
            attempt_start = time.monotonic()
            attempt: dict[str, Any] = {
                "attempt": attempt_number,
                "started_at": _utc_now(),
                "status": "request_started",
                "latency_s": None,
                "raw_content": None,
                "reasoning_content": None,
                "finish_reason": None,
                "response_id": None,
                "request_id": None,
                "usage": {},
                "validation_error": None,
                "outside_content": None,
                "format_warnings": [],
                "error": None,
                "error_type": None,
            }
            try:
                kwargs = {
                    "model": config.model,
                    "messages": request_messages,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "timeout": config.timeout_s,
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": config.enable_thinking,
                        },
                    },
                }
                response = await client.chat.completions.create(**kwargs)
                content, reasoning, finish_reason = _message_parts(response)
                attempt.update({
                    "raw_content": content,
                    "reasoning_content": reasoning,
                    "finish_reason": finish_reason,
                    "response_id": str(getattr(response, "id", "") or "") or None,
                    "request_id": _request_id(response),
                    "usage": _usage(response),
                })
                if finish_reason == "length":
                    raise SplitFormatError("finish_reason=length", content)
                boundaries, structural_warnings = _parse_split_boundaries_with_warnings(
                    content, atoms,
                )
                attempt["format_warnings"].extend(structural_warnings)
                _body, outside_prefix, outside_suffix = _split_block_parts(content)
                if outside_prefix.strip() or outside_suffix.strip():
                    attempt["outside_content"] = {
                        "prefix": outside_prefix,
                        "suffix": outside_suffix,
                    }
                    attempt["format_warnings"].append("content_outside_marker_block")
                sections = sections_from_boundaries(source, atoms, boundaries)
                attempt["status"] = "ok"
                attempt["latency_s"] = time.monotonic() - attempt_start
                attempt["finished_at"] = _utc_now()
                attempts.append(attempt)
                return SplitResult(
                    row_id=row_id,
                    status="ok",
                    source_sha256=_sha256(source.strip()),
                    cache_key=cache_key,
                    prompt_content_sha256=prompt_content_sha256,
                    boundaries=boundaries,
                    atoms=atoms,
                    sections=sections,
                    attempts=attempts,
                    usage=_aggregate_usage(attempts),
                    latency_s=time.monotonic() - start,
                )
            except SplitFormatError as exc:
                final_status = "format_error"
                final_error = exc.reason
                final_error_type = type(exc).__name__
                attempt.update({
                    "status": "format_error",
                    "validation_error": exc.reason,
                    "error": exc.reason,
                    "error_type": type(exc).__name__,
                })
                if attempt_number < config.max_format_attempts:
                    request_messages = _correction_messages(
                        messages,
                        str(attempt.get("raw_content") or exc.raw_content),
                        exc.reason,
                    )
            except Exception as exc:  # noqa: BLE001
                final_status = "api_error"
                final_error = f"{type(exc).__name__}: {exc}"
                final_error_type = type(exc).__name__
                attempt.update({
                    "status": "api_error",
                    "error": final_error,
                    "error_type": final_error_type,
                })
            finally:
                if attempt.get("latency_s") is None:
                    attempt["latency_s"] = time.monotonic() - attempt_start
                attempt.setdefault("finished_at", _utc_now())
                if not attempts or attempts[-1] is not attempt:
                    attempts.append(attempt)

    return SplitResult(
        row_id=row_id,
        status=final_status,
        source_sha256=_sha256(source.strip()),
        cache_key=cache_key,
        prompt_content_sha256=prompt_content_sha256,
        atoms=atoms,
        attempts=attempts,
        usage=_aggregate_usage(attempts),
        latency_s=time.monotonic() - start,
        error=final_error,
        error_type=final_error_type,
    )


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[rank]


def _write_summary(
    path: Path,
    results: Sequence[SplitResult],
    config: LLMCotSplitterConfig,
) -> None:
    status_counts: dict[str, int] = {}
    usage: dict[str, int] = {}
    latencies: list[float] = []
    artifact_usage: dict[str, int] = {}
    artifact_latencies: list[float] = []
    section_counts: list[int] = []
    attempt_count = 0
    artifact_attempt_count = 0
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        artifact_attempt_count += len(result.attempts)
        if result.attempts:
            artifact_latencies.append(result.latency_s)
        for key, value in result.usage.items():
            artifact_usage[key] = artifact_usage.get(key, 0) + int(value)
        if not result.cached:
            latencies.append(result.latency_s)
            attempt_count += len(result.attempts)
            for key, value in result.usage.items():
                usage[key] = usage.get(key, 0) + int(value)
        if result.ok:
            section_counts.append(len(result.sections))
    payload = {
        "splitter_version": SPLITTER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": _utc_now(),
        "config": {
            **asdict(config),
            "api_key": "***",
        },
        "rows": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "cached_rows": sum(result.cached for result in results),
        "request_attempts": attempt_count,
        "usage": usage,
        "latency_s": {
            "sum": sum(latencies),
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "max": max(latencies) if latencies else None,
        },
        # A cache-only replay must not erase the cost/latency evidence stored in
        # the selected per-row artifacts.  The legacy top-level fields above
        # continue to describe only this invocation; this nested block describes
        # the original requests represented by the current selected results.
        "selected_artifact_stats": {
            "request_attempts": artifact_attempt_count,
            "usage": artifact_usage,
            "latency_s": {
                "sum": sum(artifact_latencies),
                "p50": _percentile(artifact_latencies, 0.50),
                "p90": _percentile(artifact_latencies, 0.90),
                "max": max(artifact_latencies) if artifact_latencies else None,
            },
        },
        "sections_per_row": {
            "mean": (sum(section_counts) / len(section_counts)) if section_counts else None,
            "min": min(section_counts) if section_counts else None,
            "max": max(section_counts) if section_counts else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def split_cot_rows(
    rows: Iterable[Mapping[str, Any]],
    config: LLMCotSplitterConfig | Mapping[str, Any] | Any,
    artifact_root: str | Path,
    *,
    client: Any | None = None,
    row_id_field: str = "name",
    source_field: str = "post_think_cot",
) -> dict[str, SplitResult]:
    """Split many rows concurrently with strict, resumable LLM annotation.

    Successful rows are cached by source, model and prompt version. Failed rows
    remain explicit terminal results and are retried by a later invocation. No
    deterministic splitter is used as a fallback.
    """
    cfg = LLMCotSplitterConfig.from_value(config)
    row_list = list(rows)
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(row_list):
        row_id = str(row.get(row_id_field) or row.get("ID") or "")
        if not row_id:
            raise ValueError(f"splitter row {index} has no {row_id_field!r} or 'ID'")
        if row_id in seen:
            raise ValueError(f"duplicate splitter row ID: {row_id}")
        seen.add(row_id)
        source = str(row.get(source_field) or "")
        if not source.strip():
            raise ValueError(f"splitter row {row_id!r} has empty {source_field!r}")
        normalized.append((row_id, source))

    root = Path(artifact_root)
    results_path = root / RESULTS_FILENAME
    summary_path = root / SUMMARY_FILENAME
    cached_rows = _read_latest(results_path)
    results: dict[str, SplitResult] = {}
    pending: list[tuple[str, str, list[dict[str, Any]], str]] = []
    for row_id, source in normalized:
        atoms = atomize_cot(source)
        cache_key = split_cache_key(source, cfg, atoms)
        cached_row = cached_rows.get(row_id)
        if cached_row and str(cached_row.get("cache_key") or "") == cache_key:
            cached_result = _result_from_cache(cached_row, source, atoms)
            if cached_result is not None:
                results[row_id] = cached_result
                if cached_result.normalizer_replayed:
                    _append_jsonl(results_path, cached_result.to_artifact_row())
                continue
        pending.append((row_id, source, atoms, cache_key))

    owned_client = client is None
    if client is None and pending:
        client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.openai_base_url,
            max_retries=0,
        )
    semaphore = asyncio.Semaphore(cfg.concurrency)
    artifact_lock = asyncio.Lock()

    async def run_and_persist(
        row_id: str,
        source: str,
        atoms: list[dict[str, Any]],
        cache_key: str,
    ) -> tuple[str, SplitResult]:
        result = await _split_one(
            client=client,
            semaphore=semaphore,
            row_id=row_id,
            source=source,
            atoms=atoms,
            config=cfg,
            cache_key=cache_key,
        )
        async with artifact_lock:
            _append_jsonl(results_path, result.to_artifact_row())
        return row_id, result

    try:
        if pending:
            completed = await asyncio.gather(*(
                run_and_persist(row_id, source, atoms, cache_key)
                for row_id, source, atoms, cache_key in pending
            ))
            results.update(completed)
    finally:
        if owned_client and client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                maybe_awaitable = close()
                if hasattr(maybe_awaitable, "__await__"):
                    await maybe_awaitable

    ordered = {row_id: results[row_id] for row_id, _source in normalized}
    _write_summary(summary_path, list(ordered.values()), cfg)
    return ordered


__all__ = [
    "ATOMIZER_VERSION",
    "LLMCotSplitterConfig",
    "NORMALIZER_VERSION",
    "PROMPT_VERSION",
    "RESULTS_FILENAME",
    "SPLITTER_VERSION",
    "SUMMARY_FILENAME",
    "SplitFormatError",
    "SplitResult",
    "atomize_cot",
    "build_split_messages",
    "parse_split_boundaries",
    "sections_from_boundaries",
    "split_cache_key",
    "split_cot_rows",
]
