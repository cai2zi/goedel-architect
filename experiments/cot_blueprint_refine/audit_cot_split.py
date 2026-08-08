#!/usr/bin/env python3
"""Read-only quality audit for an installed COT split manifest.

The auditor never calls a model and never rewrites experiment artifacts.  It
compares ``post_think_cot`` with the installed steps/claims and, when available,
joins those steps to the immutable atom inventory in ``llm_cot_splits.jsonl``.
An output file is written only when the caller explicitly passes ``--json-out``
or ``--table-out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_COMPOUND_THRESHOLD = 4

_SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S.*$", re.DOTALL)
_NAMED_HEADING_RE = re.compile(
    r"^\s*(?:\*\*)?(?:step|case|part|answer|final\s+answer|conclusion|"
    r"verification|check)\b[^\n]*(?:\*\*)?\s*:?\s*$",
    re.IGNORECASE,
)
_LIST_TITLE_RE = re.compile(
    r"^\s*(?:[-+*]|\d{1,3}[.)])\s+\*\*[^*\n]+\*\*\s*:?\s*$"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _preview(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _latest_rows(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        row_id = str(row.get(key) or "")
        if not row_id:
            continue
        if row_id not in latest:
            order.append(row_id)
        latest[row_id] = row
    return [latest[row_id] for row_id in order]


def _decode_steps(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if value is None or value == "":
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [dict(item) for item in decoded if isinstance(item, Mapping)]


def _resolve_prepared(target: Path) -> tuple[Path, Path]:
    target = target.resolve()
    if target.is_file():
        return target.parent, target
    direct = target / "generation_inputs.jsonl"
    if direct.is_file():
        return target, direct
    nested = target / "prepared" / "generation_inputs.jsonl"
    if nested.is_file():
        return target / "prepared", nested
    raise FileNotFoundError(
        f"could not find generation_inputs.jsonl under {target}"
    )


def _is_pure_title(text: str) -> bool:
    lines = [
        line.strip()
        for line in str(text or "").splitlines()
        if line.strip() and not _SEPARATOR_RE.fullmatch(line.strip())
    ]
    if len(lines) != 1:
        return False
    line = lines[0]
    list_title = _LIST_TITLE_RE.fullmatch(line)
    if list_title and re.search(r"[$\\=<>≤≥≠]|(?:⇒|↔|→)", line):
        # A bold list item can still be a mathematical assertion, e.g.
        # ``- **AB = BC**``.  Presentation styling alone must not hide it.
        list_title = None
    return bool(
        _MARKDOWN_HEADING_RE.fullmatch(line)
        or _NAMED_HEADING_RE.fullmatch(line)
        or list_title
    )


def _is_layout_atom(atom: Mapping[str, Any]) -> bool:
    kind = str(atom.get("kind") or "")
    text = str(atom.get("source_text") or "")
    compact = text.strip()
    if not compact or kind == "heading" or _SEPARATOR_RE.fullmatch(compact):
        return True
    if kind == "table_row" and _TABLE_SEPARATOR_RE.fullmatch(compact):
        return True
    return _is_pure_title(text)


def _structural_family(atom: Mapping[str, Any]) -> str | None:
    if _is_layout_atom(atom):
        return None
    kind = str(atom.get("kind") or "")
    if kind == "list_item":
        return "list_item"
    if kind == "table_row":
        return "table_row"
    if kind == "display_math":
        return "display_math"
    return None


def _load_split_indexes(
    path: Path | None,
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    exact: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_version: dict[tuple[str, str], dict[str, Any]] = {}
    latest_ok: dict[str, dict[str, Any]] = {}
    if path is None or not path.is_file():
        return exact, by_version, latest_ok
    for result in _read_jsonl(path):
        if str(result.get("status") or "") != "ok":
            continue
        row_id = str(result.get("row_id") or result.get("ID") or "")
        version = str(result.get("splitter_version") or "")
        source_sha = str(result.get("source_sha256") or "")
        if not row_id:
            continue
        exact[(row_id, version, source_sha)] = result
        by_version[(row_id, version)] = result
        latest_ok[row_id] = result
    return exact, by_version, latest_ok


def _recompute_atoms(source: str) -> list[dict[str, Any]]:
    """Use the current immutable atomizer only when no recorded inventory exists."""
    experiments_root = Path(__file__).resolve().parents[1]
    if str(experiments_root) not in sys.path:
        sys.path.insert(0, str(experiments_root))
    try:
        from cot_blueprint_refine.llm_cot_splitter import atomize_cot

        return [dict(atom) for atom in atomize_cot(source)]
    except (ImportError, RuntimeError, TypeError, ValueError):
        return []


def _atom_inventory(
    row: Mapping[str, Any],
    source: str,
    indexes: tuple[
        dict[tuple[str, str, str], dict[str, Any]],
        dict[tuple[str, str], dict[str, Any]],
        dict[str, dict[str, Any]],
    ],
) -> tuple[list[dict[str, Any]], str]:
    exact, by_version, latest_ok = indexes
    row_id = str(row.get("name") or row.get("row_id") or row.get("ID") or "")
    version = str(row.get("cot_splitter_version") or "")
    source_sha = _sha256(source.strip())
    result = exact.get((row_id, version, source_sha))
    provenance = "recorded_exact"
    if result is None:
        candidate = by_version.get((row_id, version))
        if candidate and str(candidate.get("source_sha256") or "") in {"", source_sha}:
            result = candidate
            provenance = "recorded_version"
    if result is None:
        candidate = latest_ok.get(row_id)
        if candidate and str(candidate.get("source_sha256") or "") in {"", source_sha}:
            result = candidate
            provenance = "recorded_latest"
    if result is not None and isinstance(result.get("atoms"), list):
        atoms = [dict(atom) for atom in result["atoms"] if isinstance(atom, Mapping)]
        if atoms:
            return atoms, provenance
    return _recompute_atoms(source), "recomputed_current_atomizer"


def _step_spans(
    source: str,
    steps: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, int]], bool]:
    core = source.strip()
    core_start = source.find(core) if core else 0
    spans: list[tuple[int, int]] = []
    cursor = core_start
    valid = True
    for step in steps:
        text = str(step.get("source_text") or "")
        declared_start = step.get("source_start")
        declared_end = step.get("source_end")
        try:
            start = int(declared_start)
            end = int(declared_end)
        except (TypeError, ValueError):
            start, end = cursor, cursor + len(text)
        if start != cursor or end < start or source[start:end] != text:
            valid = False
            start, end = cursor, cursor + len(text)
        spans.append((start, end))
        cursor = end
    if cursor != core_start + len(core):
        valid = False
    return spans, valid


def _atoms_for_step(
    step: Mapping[str, Any],
    span: tuple[int, int],
    atoms: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {
        str(atom.get("atom_id") or ""): dict(atom)
        for atom in atoms
        if str(atom.get("atom_id") or "")
    }
    atom_ids = [str(value) for value in (step.get("atom_ids") or [])]
    if atom_ids and all(atom_id in by_id for atom_id in atom_ids):
        return [by_id[atom_id] for atom_id in atom_ids]
    start, end = span
    selected: list[dict[str, Any]] = []
    for atom in atoms:
        try:
            atom_start = int(atom.get("source_start"))
            atom_end = int(atom.get("source_end"))
        except (TypeError, ValueError):
            continue
        if start <= atom_start and atom_end <= end:
            selected.append(dict(atom))
    return selected


def _relative_atom_spans(
    step_text: str,
    step_span: tuple[int, int],
    atoms: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int] | None]:
    start, _end = step_span
    relative: list[tuple[int, int] | None] = []
    cursor = 0
    for atom in atoms:
        text = str(atom.get("source_text") or "")
        try:
            atom_start = int(atom.get("source_start")) - start
            atom_end = int(atom.get("source_end")) - start
        except (TypeError, ValueError):
            atom_start = step_text.find(text, cursor)
            atom_end = atom_start + len(text) if atom_start >= 0 else -1
        if atom_start < 0 or atom_end < atom_start or step_text[atom_start:atom_end] != text:
            atom_start = step_text.find(text, cursor)
            atom_end = atom_start + len(text) if atom_start >= 0 else -1
        if atom_start < 0:
            relative.append(None)
            continue
        relative.append((atom_start, atom_end))
        cursor = atom_end
    return relative


def _claim_spans(
    step_text: str,
    claims: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, int] | None], int, int]:
    spans: list[tuple[int, int] | None] = []
    not_found = 0
    hash_mismatch = 0
    cursor = 0
    for claim in claims:
        text = str(claim.get("source_text") or "")
        if claim.get("source_sha256") and str(claim["source_sha256"]) != _sha256(text):
            hash_mismatch += 1
        start = end = -1
        try:
            declared_start = int(claim.get("source_start"))
            declared_end = int(claim.get("source_end"))
            if step_text[declared_start:declared_end] == text:
                start, end = declared_start, declared_end
        except (TypeError, ValueError):
            pass
        if start < 0:
            start = step_text.find(text, cursor)
            if start < 0:
                start = step_text.find(text)
            end = start + len(text) if start >= 0 else -1
        if start < 0:
            spans.append(None)
            not_found += 1
        else:
            spans.append((start, end))
            cursor = end
    return spans, not_found, hash_mismatch


def _core_span(text: str, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    body = text[start:end]
    left = len(body) - len(body.lstrip())
    right = len(body.rstrip())
    return start + left, start + right


def _contains(container: tuple[int, int], contained: tuple[int, int]) -> bool:
    return container[0] <= contained[0] and contained[1] <= container[1]


def _audit_step(
    row_id: str,
    step: Mapping[str, Any],
    step_index: int,
    step_span: tuple[int, int],
    atoms: Sequence[Mapping[str, Any]],
    *,
    compound_threshold: int,
    include_text: bool,
) -> dict[str, Any]:
    text = str(step.get("source_text") or "")
    claims = [dict(value) for value in (step.get("claims") or []) if isinstance(value, Mapping)]
    context = not bool(step.get("requires_formalization", bool(claims))) or str(
        step.get("role") or ""
    ) == "context"
    pure_title = _is_pure_title(text)
    atom_spans = _relative_atom_spans(text, step_span, atoms)
    claim_spans, claims_not_found, claim_hash_mismatches = _claim_spans(text, claims)
    scope_spans = [
        (int(segment["source_start"]), int(segment["source_end"]))
        for segment in (step.get("segments") or [])
        if isinstance(segment, Mapping)
        and segment.get("scope_id")
        and isinstance(segment.get("source_start"), int)
        and isinstance(segment.get("source_end"), int)
    ]
    unscoped_context_spans = [
        (int(segment["source_start"]), int(segment["source_end"]))
        for segment in (step.get("segments") or [])
        if isinstance(segment, Mapping)
        and segment.get("kind") == "context"
        and not segment.get("scope_id")
        and isinstance(segment.get("source_start"), int)
        and isinstance(segment.get("source_end"), int)
    ]
    layout_flags = [_is_layout_atom(atom) for atom in atoms]
    substantive_indices = [index for index, layout in enumerate(layout_flags) if not layout]
    substantive_claim_indices = [
        index
        for index, claim in enumerate(claims)
        if str(claim.get("source_text") or "").strip()
        and not _is_pure_title(str(claim.get("source_text") or ""))
    ]

    atom_core_spans: list[tuple[int, int] | None] = []
    for atom, span in zip(atoms, atom_spans):
        atom_core_spans.append(
            _core_span(text, span) if span is not None else None
        )
    claims_covering_atom: dict[int, list[int]] = {}
    claim_substantive_counts: dict[int, int] = {}
    claim_family_counts: dict[tuple[int, str], int] = {}
    for claim_index, claim_span in enumerate(claim_spans):
        if claim_span is None:
            continue
        covered = [
            atom_index
            for atom_index, atom_span in enumerate(atom_core_spans)
            if atom_span is not None and _contains(claim_span, atom_span)
        ]
        for atom_index in covered:
            claims_covering_atom.setdefault(atom_index, []).append(claim_index)
        claim_substantive_counts[claim_index] = sum(
            atom_index in substantive_indices for atom_index in covered
        )
        family_counter = Counter(
            family
            for atom_index in covered
            if (family := _structural_family(atoms[atom_index])) is not None
        )
        for family, count in family_counter.items():
            claim_family_counts[(claim_index, family)] = count

    substantive_covered = 0
    substantive_isolated = 0
    substantive_scope_covered = 0
    substantive_claim_or_scope_covered = 0
    substantive_intentional_context = 0
    for atom_index in substantive_indices:
        covering = claims_covering_atom.get(atom_index, [])
        atom_span = atom_core_spans[atom_index]
        scope_covered = bool(
            atom_span is not None
            and any(_contains(scope_span, atom_span) for scope_span in scope_spans)
        )
        if covering:
            substantive_covered += 1
        if scope_covered:
            substantive_scope_covered += 1
        if covering or scope_covered:
            substantive_claim_or_scope_covered += 1
        if (
            atom_span is not None
            and any(_contains(span, atom_span) for span in unscoped_context_spans)
        ):
            substantive_intentional_context += 1
        if any(claim_substantive_counts.get(index, 0) <= 1 for index in covering):
            substantive_isolated += 1

    family_metrics: dict[str, dict[str, Any]] = {}
    for family in ("list_item", "table_row", "display_math"):
        target_indices = [
            index
            for index, atom in enumerate(atoms)
            if _structural_family(atom) == family
        ]
        covered = 0
        dedicated = 0
        strict_isolated = 0
        for atom_index in target_indices:
            covering = claims_covering_atom.get(atom_index, [])
            if covering:
                covered += 1
            if any(claim_family_counts.get((index, family), 0) <= 1 for index in covering):
                dedicated += 1
            if any(claim_substantive_counts.get(index, 0) <= 1 for index in covering):
                strict_isolated += 1
        family_metrics[family] = {
            "total": len(target_indices),
            "covered_by_claim": covered,
            "covered_rate": _ratio(covered, len(target_indices)),
            "dedicated_claim": dedicated,
            "dedicated_rate": _ratio(dedicated, len(target_indices)),
            "strict_isolated_claim": strict_isolated,
            "strict_isolated_rate": _ratio(strict_isolated, len(target_indices)),
        }

    substantive_count = len(substantive_indices)
    substantive_claim_count = len(substantive_claim_indices)
    # Scope prefixes are intentionally not standalone propositions.  They are
    # semantically represented by their applies_to bindings and therefore do
    # not count as unresolved bundled claims.
    unresolved_count = sum(
        1
        for atom_index in substantive_indices
        if not any(
            claim_substantive_counts.get(index, 0) <= 1
            for index in claims_covering_atom.get(atom_index, [])
        )
        and not (
            atom_core_spans[atom_index] is not None
            and any(
                _contains(scope_span, atom_core_spans[atom_index])
                for scope_span in scope_spans
            )
        )
        and not (
            atom_core_spans[atom_index] is not None
            and any(
                _contains(context_span, atom_core_spans[atom_index])
                for context_span in unscoped_context_spans
            )
        )
    )
    formalizable_atom_count = substantive_count - substantive_intentional_context
    compound = substantive_count > compound_threshold or unresolved_count > compound_threshold
    result: dict[str, Any] = {
        "row_id": row_id,
        "step_id": str(step.get("step_id") or f"S{step_index:03d}"),
        "step_index": step_index,
        "role": str(step.get("role") or ""),
        "requires_formalization": not context,
        "context": context,
        "pure_title": pure_title,
        "source_start": step_span[0],
        "source_end": step_span[1],
        "source_char_count": len(text),
        "source_sha256": _sha256(text),
        "source_preview": _preview(text),
        "atom_count": len(atoms),
        "layout_atom_count": sum(layout_flags),
        "substantive_atom_count": substantive_count,
        "claim_count": len(claims),
        "scope_count": sum(
            bool(segment.get("scope_id")) for segment in (step.get("segments") or [])
        ),
        "scope_type_counts": dict(sorted(Counter(
            str(segment.get("scope_type") or "unknown")
            for segment in (step.get("segments") or [])
            if segment.get("scope_id")
        ).items())),
        "scoped_claim_count": sum(bool(claim.get("scope_ids")) for claim in claims),
        "substantive_claim_count": substantive_claim_count,
        "claims_not_found_in_step": claims_not_found,
        "claim_hash_mismatches": claim_hash_mismatches,
        "subclaim_coverage": {
            "substantive_atoms": substantive_count,
            "covered_by_claim": substantive_covered,
            "covered_rate": _ratio(substantive_covered, substantive_count),
            "covered_by_scope": substantive_scope_covered,
            "covered_by_claim_or_scope": substantive_claim_or_scope_covered,
            "claim_or_scope_rate": _ratio(
                substantive_claim_or_scope_covered, substantive_count
            ),
            "isolated_by_claim": substantive_isolated,
            "isolated_rate": _ratio(substantive_isolated, substantive_count),
            "unresolved_after_scope": unresolved_count,
            "intentional_unscoped_context": substantive_intentional_context,
            "formalizable_atoms": formalizable_atom_count,
            "formalizable_covered": substantive_claim_or_scope_covered,
            "formalizable_coverage_rate": _ratio(
                substantive_claim_or_scope_covered, formalizable_atom_count
            ),
        },
        "structural_claim_coverage": family_metrics,
        "compound": compound,
        "compound_score": {
            "unisolated_substantive_atoms": unresolved_count,
            "substantive_atoms": substantive_count,
            "substantive_claims": substantive_claim_count,
            "atoms_per_substantive_claim": (
                substantive_count / substantive_claim_count
                if substantive_claim_count
                else None
            ),
        },
    }
    if include_text:
        result["source_text"] = text
        result["claims"] = claims
        result["atoms"] = [dict(atom) for atom in atoms]
    return result


def _max_compound(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not steps:
        return None
    value = max(
        steps,
        key=lambda step: (
            int(step["compound_score"]["unisolated_substantive_atoms"]),
            int(step["substantive_atom_count"]),
            int(step["source_char_count"]),
        ),
    )
    return {
        "step_id": value["step_id"],
        "substantive_atom_count": value["substantive_atom_count"],
        "substantive_claim_count": value["substantive_claim_count"],
        "unisolated_substantive_atoms": value["compound_score"][
            "unisolated_substantive_atoms"
        ],
        "source_char_count": value["source_char_count"],
        "source_preview": value["source_preview"],
    }


def _audit_row(
    row: Mapping[str, Any],
    indexes: tuple[
        dict[tuple[str, str, str], dict[str, Any]],
        dict[tuple[str, str], dict[str, Any]],
        dict[str, dict[str, Any]],
    ],
    *,
    compound_threshold: int,
    include_text: bool,
) -> dict[str, Any]:
    row_id = str(row.get("name") or row.get("row_id") or row.get("ID") or "")
    source = str(row.get("post_think_cot") or row.get("informal_proof") or "")
    source_core = source.strip()
    steps = _decode_steps(
        row.get("cot_manifest_json") or row.get("cot_manifest") or row.get("cot_steps")
    )
    reconstructed = "".join(str(step.get("source_text") or "") for step in steps)
    spans, offsets_valid = _step_spans(source, steps)
    atoms, atom_provenance = _atom_inventory(row, source, indexes)
    atom_text = "".join(str(atom.get("source_text") or "") for atom in atoms)
    atom_lossless = atom_text == source_core if atoms else None
    audited_steps = [
        _audit_step(
            row_id,
            step,
            index,
            span,
            _atoms_for_step(step, span, atoms),
            compound_threshold=compound_threshold,
            include_text=include_text,
        )
        for index, (step, span) in enumerate(zip(steps, spans), start=1)
    ]
    claims = sum(int(step["claim_count"]) for step in audited_steps)
    scope_types = Counter(
        scope_type
        for step in audited_steps
        for scope_type, count in step["scope_type_counts"].items()
        for _ in range(int(count))
    )
    substantive_atoms = sum(int(step["substantive_atom_count"]) for step in audited_steps)
    covered_atoms = sum(
        int(step["subclaim_coverage"]["covered_by_claim"]) for step in audited_steps
    )
    isolated_atoms = sum(
        int(step["subclaim_coverage"]["isolated_by_claim"]) for step in audited_steps
    )
    scope_covered_atoms = sum(
        int(step["subclaim_coverage"]["covered_by_scope"]) for step in audited_steps
    )
    semantic_covered_atoms = sum(
        int(step["subclaim_coverage"]["covered_by_claim_or_scope"])
        for step in audited_steps
    )
    unresolved_atoms = sum(
        int(step["subclaim_coverage"]["unresolved_after_scope"])
        for step in audited_steps
    )
    intentional_context_atoms = sum(
        int(step["subclaim_coverage"]["intentional_unscoped_context"])
        for step in audited_steps
    )
    formalizable_atoms = sum(
        int(step["subclaim_coverage"]["formalizable_atoms"])
        for step in audited_steps
    )
    families: dict[str, dict[str, Any]] = {}
    for family in ("list_item", "table_row", "display_math"):
        total = sum(
            int(step["structural_claim_coverage"][family]["total"])
            for step in audited_steps
        )
        covered = sum(
            int(step["structural_claim_coverage"][family]["covered_by_claim"])
            for step in audited_steps
        )
        dedicated = sum(
            int(step["structural_claim_coverage"][family]["dedicated_claim"])
            for step in audited_steps
        )
        strict = sum(
            int(step["structural_claim_coverage"][family]["strict_isolated_claim"])
            for step in audited_steps
        )
        families[family] = {
            "total": total,
            "covered_by_claim": covered,
            "covered_rate": _ratio(covered, total),
            "dedicated_claim": dedicated,
            "dedicated_rate": _ratio(dedicated, total),
            "strict_isolated_claim": strict,
            "strict_isolated_rate": _ratio(strict, total),
        }
    lossless = reconstructed == source_core
    pure_title_steps = sum(bool(step["pure_title"]) for step in audited_steps)
    context_pure_titles = sum(
        bool(step["pure_title"] and step["context"]) for step in audited_steps
    )
    required_pure_titles = pure_title_steps - context_pure_titles
    issues: list[str] = []
    if not steps:
        issues.append("empty_manifest")
    if not lossless:
        issues.append("step_reconstruction_not_lossless")
    if atom_lossless is False:
        issues.append("atom_reconstruction_not_lossless")
    if not offsets_valid:
        issues.append("step_offsets_invalid")
    if required_pure_titles:
        issues.append("required_pure_title_step")
    if any(int(step["claims_not_found_in_step"]) for step in audited_steps):
        issues.append("claim_text_not_in_step")
    if any(int(step["claim_hash_mismatches"]) for step in audited_steps):
        issues.append("claim_hash_mismatch")
    if any(bool(step["compound"]) for step in audited_steps):
        issues.append("compound_step")
    result: dict[str, Any] = {
        "row_id": row_id,
        "source": str(row.get("source") or ""),
        "row_index": row.get("row_index"),
        "splitter_version": str(row.get("cot_splitter_version") or ""),
        "source_char_count": len(source_core),
        "source_sha256": _sha256(source_core),
        "step_reconstruction_lossless": lossless,
        "step_offsets_valid": offsets_valid,
        "atom_reconstruction_lossless": atom_lossless,
        "atom_inventory_provenance": atom_provenance,
        "step_count": len(audited_steps),
        "claim_count": claims,
        "scope_count": sum(int(step["scope_count"]) for step in audited_steps),
        "scope_type_counts": dict(sorted(scope_types.items())),
        "scoped_claim_count": sum(
            int(step["scoped_claim_count"]) for step in audited_steps
        ),
        "context_step_count": sum(bool(step["context"]) for step in audited_steps),
        "pure_title_step_count": pure_title_steps,
        "context_only_pure_title_step_count": context_pure_titles,
        "required_pure_title_step_count": required_pure_titles,
        "substantive_atom_count": substantive_atoms,
        "subclaim_coverage": {
            "substantive_atoms": substantive_atoms,
            "covered_by_claim": covered_atoms,
            "covered_rate": _ratio(covered_atoms, substantive_atoms),
            "covered_by_scope": scope_covered_atoms,
            "covered_by_claim_or_scope": semantic_covered_atoms,
            "claim_or_scope_rate": _ratio(semantic_covered_atoms, substantive_atoms),
            "isolated_by_claim": isolated_atoms,
            "isolated_rate": _ratio(isolated_atoms, substantive_atoms),
            "unresolved_after_scope": unresolved_atoms,
            "intentional_unscoped_context": intentional_context_atoms,
            "formalizable_atoms": formalizable_atoms,
            "formalizable_covered": semantic_covered_atoms,
            "formalizable_coverage_rate": _ratio(
                semantic_covered_atoms, formalizable_atoms
            ),
        },
        "structural_claim_coverage": families,
        "compound_step_count": sum(bool(step["compound"]) for step in audited_steps),
        "max_compound_step": _max_compound(audited_steps),
        "issues": issues,
        "steps": audited_steps,
    }
    if include_text:
        result["source_text"] = source_core
    return result


def _sum_family(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    total = sum(int(row["structural_claim_coverage"][family]["total"]) for row in rows)
    covered = sum(
        int(row["structural_claim_coverage"][family]["covered_by_claim"])
        for row in rows
    )
    dedicated = sum(
        int(row["structural_claim_coverage"][family]["dedicated_claim"])
        for row in rows
    )
    strict = sum(
        int(row["structural_claim_coverage"][family]["strict_isolated_claim"])
        for row in rows
    )
    return {
        "total": total,
        "covered_by_claim": covered,
        "covered_rate": _ratio(covered, total),
        "dedicated_claim": dedicated,
        "dedicated_rate": _ratio(dedicated, total),
        "strict_isolated_claim": strict,
        "strict_isolated_rate": _ratio(strict, total),
    }


def audit_dataset(
    target: str | Path,
    *,
    splits_path: str | Path | None = None,
    compound_threshold: int = DEFAULT_COMPOUND_THRESHOLD,
    include_text: bool = False,
    row_ids: set[str] | None = None,
) -> dict[str, Any]:
    prepared, inputs_path = _resolve_prepared(Path(target))
    if splits_path is None:
        candidate = prepared / "cot_splitter" / "llm_cot_splits.jsonl"
        split_file = candidate if candidate.is_file() else None
    else:
        split_file = Path(splits_path).resolve()
    indexes = _load_split_indexes(split_file)
    input_rows = _latest_rows(_read_jsonl(inputs_path), "name")
    if row_ids:
        input_rows = [row for row in input_rows if str(row.get("name") or "") in row_ids]
    input_rows.sort(
        key=lambda row: (
            str(row.get("source") or ""),
            int(row.get("row_index") or -1),
            str(row.get("name") or ""),
        )
    )
    rows = [
        _audit_row(
            row,
            indexes,
            compound_threshold=compound_threshold,
            include_text=include_text,
        )
        for row in input_rows
    ]
    substantive_atoms = sum(int(row["substantive_atom_count"]) for row in rows)
    covered_atoms = sum(int(row["subclaim_coverage"]["covered_by_claim"]) for row in rows)
    isolated_atoms = sum(int(row["subclaim_coverage"]["isolated_by_claim"]) for row in rows)
    scope_covered_atoms = sum(
        int(row["subclaim_coverage"]["covered_by_scope"]) for row in rows
    )
    semantic_covered_atoms = sum(
        int(row["subclaim_coverage"]["covered_by_claim_or_scope"]) for row in rows
    )
    unresolved_atoms = sum(
        int(row["subclaim_coverage"]["unresolved_after_scope"]) for row in rows
    )
    intentional_context_atoms = sum(
        int(row["subclaim_coverage"]["intentional_unscoped_context"])
        for row in rows
    )
    formalizable_atoms = sum(
        int(row["subclaim_coverage"]["formalizable_atoms"])
        for row in rows
    )
    top_compound = sorted(
        [
            {
                "row_id": row["row_id"],
                **step,
            }
            for row in rows
            for step in row["steps"]
        ],
        key=lambda step: (
            int(step["compound_score"]["unisolated_substantive_atoms"]),
            int(step["substantive_atom_count"]),
            int(step["source_char_count"]),
        ),
        reverse=True,
    )
    issue_counts = Counter(issue for row in rows for issue in row["issues"])
    scope_types = Counter(
        scope_type
        for row in rows
        for scope_type, count in row["scope_type_counts"].items()
        for _ in range(int(count))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "generation_inputs": str(inputs_path),
            "split_results": str(split_file) if split_file else None,
        },
        "definitions": {
            "substantive_atom": (
                "an immutable atom that is not a heading, separator, table separator, "
                "or title-only list item"
            ),
            "covered_by_claim": "the atom's non-whitespace span is contained in a claim",
            "covered_by_scope": (
                "the atom is an exact context prefix/case segment with applies_to claim bindings"
            ),
            "isolated_by_claim": (
                "a covering claim contains at most one substantive atom"
            ),
            "dedicated_structural_claim": (
                "a covering claim contains at most one atom of the same structural family"
            ),
            "compound_step": (
                f"more than {compound_threshold} substantive atoms or more than "
                f"{compound_threshold} substantive atoms without an isolated claim"
            ),
        },
        "summary": {
            "row_count": len(rows),
            "lossless_row_count": sum(bool(row["step_reconstruction_lossless"]) for row in rows),
            "valid_offset_row_count": sum(bool(row["step_offsets_valid"]) for row in rows),
            "atom_lossless_row_count": sum(row["atom_reconstruction_lossless"] is True for row in rows),
            "step_count": sum(int(row["step_count"]) for row in rows),
            "claim_count": sum(int(row["claim_count"]) for row in rows),
            "scope_count": sum(int(row["scope_count"]) for row in rows),
            "scope_type_counts": dict(sorted(scope_types.items())),
            "scoped_claim_count": sum(int(row["scoped_claim_count"]) for row in rows),
            "context_step_count": sum(int(row["context_step_count"]) for row in rows),
            "pure_title_step_count": sum(int(row["pure_title_step_count"]) for row in rows),
            "context_only_pure_title_step_count": sum(
                int(row["context_only_pure_title_step_count"]) for row in rows
            ),
            "required_pure_title_step_count": sum(
                int(row["required_pure_title_step_count"]) for row in rows
            ),
            "substantive_atom_count": substantive_atoms,
            "subclaim_coverage": {
                "substantive_atoms": substantive_atoms,
                "covered_by_claim": covered_atoms,
                "covered_rate": _ratio(covered_atoms, substantive_atoms),
                "covered_by_scope": scope_covered_atoms,
                "covered_by_claim_or_scope": semantic_covered_atoms,
                "claim_or_scope_rate": _ratio(semantic_covered_atoms, substantive_atoms),
                "isolated_by_claim": isolated_atoms,
                "isolated_rate": _ratio(isolated_atoms, substantive_atoms),
                "unresolved_after_scope": unresolved_atoms,
                "intentional_unscoped_context": intentional_context_atoms,
                "formalizable_atoms": formalizable_atoms,
                "formalizable_covered": semantic_covered_atoms,
                "formalizable_coverage_rate": _ratio(
                    semantic_covered_atoms, formalizable_atoms
                ),
            },
            "structural_claim_coverage": {
                family: _sum_family(rows, family)
                for family in ("list_item", "table_row", "display_math")
            },
            "compound_step_count": sum(int(row["compound_step_count"]) for row in rows),
            "rows_with_compound_step": sum(bool(row["compound_step_count"]) for row in rows),
            "issue_counts": dict(sorted(issue_counts.items())),
        },
        "top_compound_steps": top_compound,
        "rows": rows,
    }


def _coverage_cell(metric: Mapping[str, Any], key: str = "dedicated_claim") -> str:
    return f"{int(metric.get(key) or 0)}/{int(metric.get('total') or 0)}"


def render_table(report: Mapping[str, Any]) -> str:
    headers = [
        "row_id",
        "lossless",
        "steps",
        "claims",
        "ctx",
        "ctx_title",
        "req_title",
        "sub_atoms",
        "isolated",
        "compound",
        "max_sub",
        "list_ded",
        "table_ded",
        "math_ded",
        "max_step",
    ]
    lines = ["\t".join(headers)]
    for row in report.get("rows") or []:
        maximum = row.get("max_compound_step") or {}
        structural = row["structural_claim_coverage"]
        lines.append(
            "\t".join(
                [
                    str(row["row_id"]),
                    "yes" if row["step_reconstruction_lossless"] else "NO",
                    str(row["step_count"]),
                    str(row["claim_count"]),
                    str(row["context_step_count"]),
                    str(row["context_only_pure_title_step_count"]),
                    str(row["required_pure_title_step_count"]),
                    str(row["substantive_atom_count"]),
                    (
                        f"{row['subclaim_coverage']['isolated_by_claim']}/"
                        f"{row['subclaim_coverage']['substantive_atoms']}"
                    ),
                    str(row["compound_step_count"]),
                    str(maximum.get("substantive_atom_count") or 0),
                    _coverage_cell(structural["list_item"]),
                    _coverage_cell(structural["table_row"]),
                    _coverage_cell(structural["display_math"]),
                    str(maximum.get("step_id") or ""),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        help="prepared directory, experiment directory, or generation_inputs.jsonl",
    )
    parser.add_argument(
        "--splits",
        help="optional llm_cot_splits.jsonl (auto-detected under prepared/cot_splitter)",
    )
    parser.add_argument("--json-out", help="write the full JSON report to this path")
    parser.add_argument("--table-out", help="also write the TSV summary to this path")
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="print JSON instead of the per-row TSV table",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="include complete source/step/claim/atom text in JSON",
    )
    parser.add_argument(
        "--compound-threshold",
        type=int,
        default=DEFAULT_COMPOUND_THRESHOLD,
        help=f"substantive-atom threshold (default: {DEFAULT_COMPOUND_THRESHOLD})",
    )
    parser.add_argument(
        "--row-id",
        action="append",
        default=[],
        help="audit only this row ID; repeat for more rows",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="number of compound steps retained in JSON (default: 20)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 for a non-lossless row, bad offset/hash, or required pure-title step",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.compound_threshold < 1:
        raise ValueError("--compound-threshold must be positive")
    report = audit_dataset(
        args.target,
        splits_path=args.splits,
        compound_threshold=args.compound_threshold,
        include_text=args.include_text,
        row_ids=set(args.row_id) or None,
    )
    report["top_compound_steps"] = report["top_compound_steps"][: max(0, args.top)]
    table = render_table(report)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.table_out:
        output = Path(args.table_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table, encoding="utf-8")
    if args.json_stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(table, end="")
        summary = report["summary"]
        print(
            "# summary "
            f"rows={summary['row_count']} "
            f"lossless={summary['lossless_row_count']}/{summary['row_count']} "
            f"steps={summary['step_count']} claims={summary['claim_count']} "
            f"isolated={summary['subclaim_coverage']['isolated_by_claim']}/"
            f"{summary['subclaim_coverage']['substantive_atoms']} "
            f"compound={summary['compound_step_count']} "
            f"required_titles={summary['required_pure_title_step_count']}"
        )
    if args.strict:
        fatal = {
            "empty_manifest",
            "step_reconstruction_not_lossless",
            "atom_reconstruction_not_lossless",
            "step_offsets_invalid",
            "required_pure_title_step",
            "claim_text_not_in_step",
            "claim_hash_mismatch",
        }
        if any(fatal.intersection(row["issues"]) for row in report["rows"]):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
