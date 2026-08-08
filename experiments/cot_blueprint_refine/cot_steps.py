from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from cot_blueprint_refine.cot_manifest_validation import (
    OFFSET_SPACE,
    validate_split_manifest,
)


_SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"#{1,6}\s+.+|"
    r"\*\*(?:step|case|part|final\s+answer|answer|verification|check|solution|conclusion)\b.*?\*\*\s*:?|"
    r"(?:step|case|part)\s+(?:\d+|[A-Za-z])\b.*?:?"
    r")\s*$",
    re.IGNORECASE,
)
_GENERIC_HEADING_RE = re.compile(
    r"^\s*#{1,6}\s*(?:solution|answer|detailed solution|reasoning)\s*$",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
_STEP_ORDINAL_PREFIX_RE = re.compile(
    r"^\s*(?:step|case|part)\s+(?:\d+|[A-Za-z])\b\s*(?:[.):\-]\s*)?",
    re.IGNORECASE,
)
_BARE_ORDINAL_PREFIX_RE = re.compile(r"^\s*\d{1,3}\s*[.):]\s+")
_PURE_NUMBERED_BOLD_TITLE_RE = re.compile(
    r"^\s*(?:"
    r"\*\*\s*(?:\(?\d{1,3}\)?[.):]|[IVXLC]+[.):])\s+[^\n]+?\*\*"
    r"|(?:\(?\d{1,3}\)?[.)]|[IVXLC]+[.)])\s+\*\*[^\n]+?\*\*"
    r")\s*:?\s*$",
    re.IGNORECASE,
)
_TABLE_DELIMITER_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-+*]|\d{1,3}[.)])\s+")
_PURE_BOLD_LABEL_RE = re.compile(r"^\s*\*\*([^*\n]{1,80})\*\*\s*:\s*$")
_SHORT_PRESENTATION_INTRO_RE = re.compile(
    r"^\s*(?:"
    r"here(?:['’]s)\s+why|"
    r"key\s+observations(?:\b[^:\n]*)?|"
    r"let(?:['’]s)\s+(?:compute|summarize|check|analy[sz]e|consider)\b[^:\n]*"
    r")\s*:\s*$",
    re.IGNORECASE,
)
_SUBSTANTIVE_ASSERTION_RE = re.compile(
    r"\b(?:"
    r"is|are|was|were|has|have|had|"
    r"assume|suppose|define|given|"
    r"equal(?:s)?|satisf(?:y|ies)|impl(?:y|ies)|"
    r"divid(?:e|es)|bisect(?:s)?|intersect(?:s)?|"
    r"lie(?:s)?\s+on|"
    r"even|odd|prime|positive|negative|"
    r"greater|less|tangent|parallel|perpendicular|congruent|similar"
    r")\b",
    re.IGNORECASE,
)
_COLON_COMPLETE_ASSERTION_RE = re.compile(
    r"\b(?:"
    r"equal(?:s)?|satisf(?:y|ies)|impl(?:y|ies)|divid(?:e|es)|bisect(?:s)?|"
    r"intersect(?:s)?|lie(?:s)?\s+on|even|odd|prime|positive|negative|"
    r"greater|less|tangent|parallel|perpendicular|congruent|similar|"
    r"exactly|only|unique|uniform(?:ly)?|distributed|coprime|"
    r"mutually\s+exclusive|at\s+most|at\s+least|cannot|must|"
    r"largest|smallest|maximum|minimum"
    r")\b",
    re.IGNORECASE,
)
_CASE_SCOPE_CUE_RE = re.compile(
    r"\b(?:case|subcase|overlap|when|if|assume)\b", re.IGNORECASE,
)
_CASE_SCOPE_PAYLOAD_RE = re.compile(
    r"\b(?:all|any|subset|rows?|columns?|real|complex|even|odd|positive|negative)\b",
    re.IGNORECASE,
)
MANIFEST_SCHEMA_VERSION = 3
SUBCLAIM_BUILDER_VERSION = "structured-subclaims-v5.2"
_NARRATIVE_OPENER_RE = re.compile(
    r"^\s*To solve (?:this |the )?.+?,\s*we\s+"
    r"(?:analy[sz]e|will analy[sz]e|consider)\s+"
    r"(?:the\s+)?(?:diagram|geometric configuration|configuration|problem)"
    r"(?:\s+and\s+the\s+given\s+conditions)?(?:\s+involving\s+.+?)?"
    r"(?:\s+step\s+by\s+step)?\s*[.:]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_TASK_NARRATION_RE = re.compile(
    r"^\s*(?:(?:So|Now),?\s+)?(?:the\s+)?(?:key|goal|task)\s+is\s+to\s+"
    r"(?:determine|find|compute|solve|check)\b.*?[.!:]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_RELATION_PATTERNS = (
    (r"\\(?:leq?|geq?|neq|approx|equiv|in|notin|mid)\b", "latex_relation"),
    (r"↔|⇒|→", "implication"),
    (r"≤", "le"),
    (r"≥", "ge"),
    (r"≠", "ne"),
    (r"(?<![<>=!])=(?!=)", "eq"),
    (r"(?<![<])<(?![<=])", "lt"),
    (r"(?<![>])>(?![>=])", "gt"),
)


def _role(text: str) -> str:
    lowered = text.lower()
    if "final answer" in lowered or re.search(r"\bconclusion\b", lowered):
        return "conclusion"
    if re.search(r"\b(?:verify|verification|check)\b", lowered):
        return "verification"
    if re.search(r"\b(?:given|setup|define|let us|we know)\b", lowered):
        return "setup"
    if re.search(r"\bcase\b", lowered):
        return "case"
    if re.search(r"\b(?:calculate|compute|simplify|evaluate)\b", lowered):
        return "computation"
    return "derived_claim"


def _requires_formalization(text: str) -> bool:
    """Return false only for a high-confidence narration-only source step.

    These openers announce that an analysis will follow but assert no
    mathematical condition of their own. Keeping them as provenance context
    without forcing a Lean lemma prevents the repair model from manufacturing
    ``True`` nodes solely to satisfy root-reachability.
    """
    without_separators = re.sub(
        r"(?m)^\s*(?:-{3,}|_{3,}|\*{3,})\s*$", "", text,
    )
    compact = re.sub(r"\s+", " ", without_separators).strip()
    lowered = compact.lower()
    numbers, relations = _semantic_atoms(compact)
    if numbers or relations or any(
        phrase in lowered
        for phrase in ("we are given", "are asked", "is given", "find the length", "find the value")
    ):
        return True
    if re.search(
        r"\b(?:tangent|parallel|perpendicular|equal|congruent|similar|divisible|"
        r"intersect|lies on|satisf(?:y|ies)|greater than|less than)\b",
        lowered,
    ):
        return True
    return _NARRATIVE_OPENER_RE.fullmatch(compact) is None


def _claim_units(text: str, step_id: str, *, enabled: bool) -> list[dict[str, str]]:
    if not enabled:
        return []
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph.strip()
    ] or [text.strip()]
    claims: list[dict[str, str]] = []
    for paragraph in paragraphs:
        if _HEADING_RE.fullmatch(paragraph) or _TASK_NARRATION_RE.fullmatch(paragraph):
            continue
        claim_id = f"{step_id}.C{len(claims) + 1:03d}"
        claims.append({
            "claim_id": claim_id,
            "source_text": paragraph,
            "source_sha256": hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
        })
    return claims


def _semantic_atoms(text: str) -> tuple[list[str], list[str]]:
    numbers = sorted(set(_NUMBER_RE.findall(text)), key=lambda value: (len(value), value))
    relations: set[str] = set()
    for pattern, label in _RELATION_PATTERNS:
        if re.search(pattern, text):
            relations.add(label)
    return numbers, sorted(relations)


def _is_short_presentation_context(text: str) -> bool:
    """Recognize only high-confidence, proposition-free presentation text.

    A colon-ended label or introduction may explain the structure of the COT,
    but it should not become a Lean claim by itself.  Numeric literals,
    symbolic relations, and common assertion predicates are explicit guards:
    when any of them occur, the text remains a claim.
    """
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact.endswith(":") or len(compact) > 160:
        return False

    numbers, relations = _semantic_atoms(compact)
    if numbers or relations or _SUBSTANTIVE_ASSERTION_RE.search(compact):
        return False

    return (
        _PURE_BOLD_LABEL_RE.fullmatch(compact) is not None
        or _SHORT_PRESENTATION_INTRO_RE.fullmatch(compact) is not None
    )


def _is_semantic_case_scope(text: str) -> bool:
    """Return true for a heading that carries an actual branch condition.

    Plain labels such as ``Case 1`` remain layout.  A heading such as
    ``Case 1: k = 4`` or ``Overlap: all rows and all columns`` is an assumption
    governing the claims below it and must survive as machine-visible scope.
    """
    compact = re.sub(r"^\s*#{1,6}\s*", "", text).strip().strip("*").strip()
    if not _CASE_SCOPE_CUE_RE.search(compact):
        return False
    _numbers, relations = _semantic_atoms(compact)
    return bool(relations or _CASE_SCOPE_PAYLOAD_RE.search(compact))


def _is_colon_prefix(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    return bool(compact and compact.endswith(":"))


def _semantic_text(section: str) -> str:
    """Remove document numbering while retaining mathematical heading text.

    A heading such as ``### Step 12: Substitute x = 7`` contributes the
    mathematical literal ``7``, but its presentational ordinal ``12`` is not a
    claim made by the COT and must not enter the semantic fingerprint.
    """
    lines = section.splitlines()
    if not lines or not _HEADING_RE.match(lines[0]):
        return section
    heading = re.sub(r"^\s*#{1,6}\s*", "", lines[0]).strip()
    if heading.startswith("**") and heading.endswith("**"):
        heading = heading[2:-2].strip()
    heading = _STEP_ORDINAL_PREFIX_RE.sub("", heading)
    heading = _BARE_ORDINAL_PREFIX_RE.sub("", heading)
    return "\n".join([heading, *lines[1:]])


def _clean_block(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _structured_sections(text: str) -> list[str]:
    sections: list[list[str]] = []
    current: list[str] = []
    saw_heading = False
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _SEPARATOR_RE.match(raw_line):
            continue
        if _HEADING_RE.match(raw_line):
            saw_heading = True
            cleaned = _clean_block(current)
            if cleaned:
                sections.append(cleaned.splitlines())
            current = [] if _GENERIC_HEADING_RE.match(raw_line) else [raw_line]
            continue
        current.append(raw_line)
    cleaned = _clean_block(current)
    if cleaned:
        sections.append(cleaned.splitlines())
    if not saw_heading:
        return []
    return [block for lines in sections if (block := _clean_block(lines))]


def _paragraph_sections(text: str) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph.strip() and not _SEPARATOR_RE.match(paragraph)
    ]
    sections: list[str] = []
    for paragraph in paragraphs:
        if sections and len(paragraph) < 80 and not re.search(r"[=<>]|\\boxed|\$", paragraph):
            sections[-1] = f"{sections[-1]}\n\n{paragraph}"
        else:
            sections.append(paragraph)
    return sections


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_pure_numbered_bold_title(text: str) -> bool:
    return _PURE_NUMBERED_BOLD_TITLE_RE.fullmatch(text.strip()) is not None


def _is_table_delimiter(text: str) -> bool:
    return _TABLE_DELIMITER_RE.fullmatch(text.strip()) is not None


def _section_atom_units(
    section: str,
    section_info: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_atoms = section_info.get("atoms")
    if not isinstance(raw_atoms, list) or not raw_atoms:
        raise ValueError("structured subclaims require a non-empty ordered atom inventory")
    units: list[dict[str, Any]] = []
    cursor = 0
    seen: set[str] = set()
    for index, raw_atom in enumerate(raw_atoms, start=1):
        if not isinstance(raw_atom, dict):
            raise ValueError(f"atom {index} is not an object")
        atom_id = str(raw_atom.get("atom_id") or "")
        if not atom_id or atom_id in seen:
            raise ValueError(f"invalid or duplicate atom id: {atom_id!r}")
        seen.add(atom_id)
        start = int(raw_atom.get("source_start", -1))
        end = int(raw_atom.get("source_end", -1))
        if start != cursor or end <= start or end > len(section):
            raise ValueError(
                f"atoms do not exactly partition section at {atom_id}: "
                f"expected_start={cursor} start={start} end={end}"
            )
        text = section[start:end]
        units.append({
            "atom_ids": [atom_id],
            "kind": str(raw_atom.get("kind") or "text"),
            "source_start": start,
            "source_end": end,
            "source_text": text,
        })
        cursor = end
    if cursor != len(section):
        raise ValueError("atoms leave the end of the section uncovered")

    # The generic sentence atomizer can split ``**1. Setup**`` at the ordinal
    # full stop. Rejoin only an exact, pure title line; mathematical prose is
    # never coalesced by this presentation-only rule.
    coalesced: list[dict[str, Any]] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if (
            unit["source_text"].lstrip().startswith("**")
            or re.match(r"^\s*(?:\(?\d{1,3}\)?[.)]|[IVXLC]+[.)])\s*", unit["source_text"], re.I)
        ):
            combined = unit["source_text"]
            final = index
            while final + 1 < len(units) and "\n" not in combined and "\r" not in combined:
                final += 1
                combined += units[final]["source_text"]
            if _is_pure_numbered_bold_title(combined):
                group = units[index:final + 1]
                coalesced.append({
                    "atom_ids": [value for item in group for value in item["atom_ids"]],
                    "kind": "heading",
                    "source_start": group[0]["source_start"],
                    "source_end": group[-1]["source_end"],
                    "source_text": combined,
                })
                index = final + 1
                continue
        coalesced.append(unit)
        index += 1
    return coalesced


def _display_intro(text: str) -> bool:
    compact = text.rstrip()
    if compact.endswith(":"):
        return True
    return re.search(
        r"(?:as follows|we (?:have|obtain|get|find|write|see)|this gives)\s*[.]?$",
        compact,
        re.IGNORECASE,
    ) is not None


def _structured_claims_and_segments(
    section: str,
    section_info: dict[str, Any],
    step_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    units = _section_atom_units(section, section_info)

    # A Markdown table's header row and mandatory delimiter carry layout, not
    # a source proposition. Every remaining data row stays independently
    # checkable even when the LLM left a coarse step boundary around the table.
    table_delimiters = {
        index for index, unit in enumerate(units)
        if unit["kind"] == "table_row" and _is_table_delimiter(unit["source_text"])
    }
    table_headers = {
        index - 1 for index in table_delimiters
        if index > 0 and units[index - 1]["kind"] == "table_row"
    }

    typed: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        text = str(unit["source_text"])
        kind = str(unit["kind"])
        scope_type = ""
        pure_numbered_title = _is_pure_numbered_bold_title(text)
        if kind == "heading" or pure_numbered_title:
            segment_kind, subtype = "context", "heading"
            if _is_semantic_case_scope(text):
                scope_type = "case_condition"
            elif pure_numbered_title:
                scope_type = "section_label"
        elif index in table_delimiters or index in table_headers:
            segment_kind, subtype = "context", "table_layout"
        elif kind == "table_row":
            segment_kind, subtype = "claim", "table_data_row"
        elif kind == "list_item":
            item_text = _LIST_PREFIX_RE.sub("", text).strip()
            if _is_short_presentation_context(item_text):
                segment_kind, subtype = "context", "list_layout"
            elif item_text and _requires_formalization(item_text):
                segment_kind, subtype = "claim", "list_item"
            else:
                segment_kind, subtype = "context", "list_layout"
        elif kind == "prose" and (
            _is_short_presentation_context(text) or not _requires_formalization(text)
        ):
            segment_kind, subtype = "context", "narration"
        else:
            segment_kind = "claim"
            subtype = "prose" if kind in {"prose", "text"} else kind
        typed.append({
            **unit,
            "segment_kind": segment_kind,
            "subtype": subtype,
            "scope_type": scope_type,
        })

    # A colon-ended prefix is not an independently provable proposition.  It
    # supplies notation, a quantifier, a method, or a label for the claims that
    # follow.  Keep its exact bytes as a shared scope instead of assigning a
    # misleading standalone COT_CLAIM ID.  This pass runs after initial typing
    # so it can require an actual later claim and never drops a terminal colon
    # assertion with no body.
    for index, unit in enumerate(typed):
        later_claim = any(
            candidate["segment_kind"] == "claim" for candidate in typed[index + 1:]
        )
        if not later_claim:
            continue
        kind = str(unit["kind"])
        if kind in {"prose", "text"} and _is_colon_prefix(str(unit["source_text"])):
            numbers, relations = _semantic_atoms(str(unit["source_text"]))
            # A colon does not erase a mathematical assertion.  Numeric or
            # relational payload and strong mathematical predicates stay a
            # claim (and can merge with an immediately following display).
            # Bare lead-ins such as "We are told:" or "The solutions are:"
            # become shared scope instead.
            if (
                numbers
                or relations
                or _COLON_COMPLETE_ASSERTION_RE.search(str(unit["source_text"]))
            ):
                continue
            unit.update({
                "segment_kind": "context",
                "subtype": "narration",
                "scope_type": "claim_prefix",
            })
        elif unit["subtype"] == "list_layout":
            unit["scope_type"] = "list_prefix"
        elif unit["subtype"] == "table_layout" and any(
            candidate["subtype"] == "table_data_row"
            for candidate in typed[index + 1:]
        ):
            unit["scope_type"] = "table_schema"

    grouped: list[dict[str, Any]] = []
    for unit in typed:
        previous = grouped[-1] if grouped else None
        merge = False
        if previous is not None and previous["segment_kind"] == unit["segment_kind"]:
            if unit["segment_kind"] == "context":
                merge = (
                    previous["subtype"] == unit["subtype"]
                    and previous.get("scope_type") == unit.get("scope_type")
                    and unit.get("scope_type") != "case_condition"
                )
            elif unit["subtype"] == "display_math" and previous["subtype"] == "prose":
                merge = _display_intro(str(previous["source_text"]))
        if merge:
            previous["source_end"] = unit["source_end"]
            previous["source_text"] += unit["source_text"]
            previous["atom_ids"].extend(unit["atom_ids"])
            if unit["subtype"] == "display_math":
                previous["subtype"] = "prose_display_math"
            continue
        grouped.append(dict(unit))

    claims: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for group in grouped:
        start = int(group["source_start"])
        end = int(group["source_end"])
        if group["segment_kind"] == "claim":
            claim_id = f"{step_id}.C{len(claims) + 1:03d}"
            claim = {
                "claim_id": claim_id,
                "source_text": section[start:end],
                "source_sha256": _hash_text(section[start:end]),
                "source_start": start,
                "source_end": end,
                "atom_ids": list(group["atom_ids"]),
                "claim_kind": group["subtype"],
            }
            claims.append(claim)
            segments.append({
                "kind": "claim",
                "claim_id": claim_id,
                "source_start": start,
                "source_end": end,
            })
        else:
            segment = {
                "kind": "context",
                "context_type": group["subtype"],
                "source_start": start,
                "source_end": end,
            }
            if group.get("scope_type"):
                segment["scope_type"] = str(group["scope_type"])
            segments.append(segment)

    # Bind every scope to the exact following claims it governs.  Case
    # conditions remain active for the rest of their LLM macro-step; ordinary
    # prefixes stop at the next scope.  Table schemas bind only table rows.
    claims_by_id = {str(claim["claim_id"]): claim for claim in claims}
    scope_index = 0
    for segment_index, segment in enumerate(segments):
        scope_type = str(segment.get("scope_type") or "")
        if not scope_type:
            continue
        first_semantic = next((
            following
            for following in segments[segment_index + 1:]
            if following.get("kind") == "claim" or following.get("scope_type")
        ), None)
        nested_scope_body = bool(
            first_semantic is not None
            and first_semantic.get("kind") == "context"
            and first_semantic.get("scope_type")
        )
        targets: list[str] = []
        for following in segments[segment_index + 1:]:
            following_scope = str(following.get("scope_type") or "")
            if scope_type == "case_condition" and following_scope == "case_condition":
                break
            if (
                scope_type not in {"case_condition", "table_schema"}
                and not nested_scope_body
                and following_scope
            ):
                break
            if following.get("kind") != "claim":
                continue
            claim_id = str(following["claim_id"])
            claim_kind = str(claims_by_id[claim_id].get("claim_kind") or "")
            if scope_type == "table_schema" and claim_kind != "table_data_row":
                break
            targets.append(claim_id)
        if not targets:
            # Retain lossless context while avoiding an empty, misleading scope.
            segment.pop("scope_type", None)
            continue
        scope_index += 1
        scope_id = f"{step_id}.G{scope_index:03d}"
        segment["scope_id"] = scope_id
        segment["applies_to_claim_ids"] = targets
        for claim_id in targets:
            claims_by_id[claim_id].setdefault("scope_ids", []).append(scope_id)

    ordered_atoms = []
    for raw_atom in section_info["atoms"]:
        start = int(raw_atom["source_start"])
        end = int(raw_atom["source_end"])
        atom_text = section[start:end]
        ordered_atoms.append({
            "atom_id": str(raw_atom["atom_id"]),
            "kind": str(raw_atom.get("kind") or "text"),
            "source_start": start,
            "source_end": end,
            "source_text": atom_text,
            "source_sha256": _hash_text(atom_text),
        })
    return claims, segments, ordered_atoms


def build_cot_steps_from_sections(
    sections: Iterable[str | dict[str, Any]],
    *,
    one_claim_per_step: bool = False,
    structured_subclaims: bool = False,
    splitter_mode: str = "deterministic_v1",
) -> list[dict[str, Any]]:
    """Build the semantic manifest from already chosen source boundaries.

    ``sections`` may contain plain strings (the legacy deterministic path) or
    dictionaries produced by a boundary annotator.  The latter can carry exact
    document offsets and immutable atom IDs.  In ``one_claim_per_step`` mode the
    complete, unmodified source slice is the sole claim for that step.  This is
    intentionally different from the legacy blank-line claim splitter: the LLM
    decides only boundaries, while the host creates IDs, hashes and dependencies.
    """
    steps: list[dict[str, Any]] = []
    for index, section_value in enumerate(sections, start=1):
        if isinstance(section_value, dict):
            section_info = dict(section_value)
            section = str(section_info.get("source_text") or "")
        else:
            section_info = {}
            section = str(section_value)
        if not section:
            raise ValueError(f"empty COT source section at index {index}")
        step_id = f"S{index:03d}"
        numbers, relations = _semantic_atoms(_semantic_text(section))
        requires_formalization = _requires_formalization(section)
        segments: list[dict[str, Any]] | None = None
        ordered_atoms: list[dict[str, Any]] | None = None
        if structured_subclaims:
            if not isinstance(section_value, dict):
                raise ValueError("structured subclaims require section dictionaries")
            claims, segments, ordered_atoms = _structured_claims_and_segments(
                section, section_info, step_id,
            )
        elif one_claim_per_step and requires_formalization:
            claims = [{
                "claim_id": f"{step_id}.C001",
                "source_text": section,
                "source_sha256": _hash_text(section),
                "source_start": 0,
                "source_end": len(section),
            }]
        else:
            claims = _claim_units(section, step_id, enabled=requires_formalization)
        requires_formalization = bool(claims)
        step: dict[str, Any] = {
            "step_id": step_id,
            "source_text": section,
            "source_sha256": _hash_text(section),
            "role": _role(section) if requires_formalization else "context",
            "requires_formalization": requires_formalization,
            "claims": claims,
            "depends_on": [],
            "numbers": numbers,
            "relations": relations,
            "splitter_mode": splitter_mode,
        }
        for key in ("source_start", "source_end", "atom_ids"):
            if key in section_info:
                step[key] = section_info[key]
        if structured_subclaims:
            step.update({
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "subclaim_builder_version": SUBCLAIM_BUILDER_VERSION,
                "offset_space": dict(OFFSET_SPACE),
                "atoms": ordered_atoms,
                "segments": segments,
                "atom_count": len(ordered_atoms or []),
                "layout_context_segment_count": sum(
                    1 for segment in (segments or [])
                    if segment.get("kind") == "context"
                    and segment.get("context_type") in {"heading", "table_layout", "list_layout"}
                ),
                "scope_count": sum(
                    1 for segment in (segments or []) if segment.get("scope_id")
                ),
            })
        elif one_claim_per_step:
            step["segments"] = ([{
                "kind": "claim",
                "claim_id": claims[0]["claim_id"],
                "source_start": 0,
                "source_end": len(section),
            }] if claims else [{
                "kind": "context",
                "source_start": 0,
                "source_end": len(section),
            }])
        steps.append(step)
    previous_substantive = ""
    for step in steps:
        if not step["requires_formalization"]:
            continue
        step["depends_on"] = [previous_substantive] if previous_substantive else []
        previous_substantive = step["step_id"]
    if steps and not any(step["role"] == "conclusion" for step in steps):
        for step in reversed(steps):
            if step["requires_formalization"]:
                step["role"] = "conclusion"
                break
    return steps


def split_cot_steps(text: str) -> list[dict[str, Any]]:
    """Deterministically split a polished post-think solution into source steps.

    The splitter intentionally uses only document structure. It does not repair,
    summarize, or reinterpret mathematical content, so it adds no model call and
    cannot silently change an erroneous source claim.
    """
    source = text.strip()
    if not source:
        return []
    sections = _structured_sections(source) or _paragraph_sections(source)
    if not sections:
        sections = [source]
    return build_cot_steps_from_sections(sections)


def render_numbered_cot(steps: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for step in steps:
        dependency_text = ",".join(step.get("depends_on") or []) or "none"
        blocks.append(
            f"[COT_STEP {step['step_id']} role={step['role']} depends_on={dependency_text} "
            f"requires_formalization={str(step.get('requires_formalization', True)).lower()}]\n"
            f"{step['source_text']}\n"
            f"[/COT_STEP {step['step_id']}]"
        )
    return "\n\n".join(blocks)


def encode_steps(steps: list[dict[str, Any]]) -> str:
    return json.dumps(steps, ensure_ascii=False, sort_keys=True)


def decode_steps(value: Any, *, source: str | None = None) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = [dict(step) for step in value]
    elif not value:
        rows = []
    else:
        decoded = json.loads(str(value))
        if not isinstance(decoded, list):
            raise ValueError("cot_manifest_json must decode to a list")
        rows = [dict(step) for step in decoded]
    if source is not None and any(
        int(step.get("manifest_schema_version") or 0) >= 2 for step in rows
    ):
        validate_split_manifest(source, rows)
    return rows
