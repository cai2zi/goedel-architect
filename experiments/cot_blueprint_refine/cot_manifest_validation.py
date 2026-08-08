"""Strict integrity checks for lossless LLM-split COT manifests.

The coordinate contract used here is deliberately small and explicit:

* ``step.source_start/source_end`` are offsets into the complete source;
* atom, claim, and segment offsets are relative to their containing step;
* steps cover ``source.strip()`` without gaps or overlaps;
* atoms and segments each cover their complete step without gaps or overlaps;
* every claim is represented by exactly one claim segment.

This module does not classify prose or judge mathematical correctness.  It only
checks that a persisted structural annotation is lossless and unambiguous.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any


OFFSET_SPACE = {
    "step": "global",
    "atoms": "step_relative",
    "claims": "step_relative",
    "segments": "step_relative",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STEP_ID_RE = re.compile(r"^S[0-9]{3}$")
_CLAIM_ID_RE = re.compile(r"^S[0-9]{3}\.C[0-9]{3,}$")
_SCOPE_ID_RE = re.compile(r"^S[0-9]{3}\.G[0-9]{3,}$")
_ATOM_ID_RE = re.compile(r"^A[0-9]{4,}$")


class SplitManifestValidationError(ValueError):
    """A split manifest violates the lossless offset/identity contract."""


def _fail(path: str, message: str) -> None:
    raise SplitManifestValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _list(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(path, "must be a list")
    return value


def _text(record: Mapping[str, Any], key: str, path: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        _fail(f"{path}.{key}", "must be a string")
    return value


def _offset(record: Mapping[str, Any], key: str, path: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{path}.{key}", "must be an integer")
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_required_hash(record: Mapping[str, Any], text: str, path: str) -> None:
    digest = record.get("source_sha256")
    hash_path = f"{path}.source_sha256"
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        _fail(hash_path, "must be a lowercase 64-character SHA-256 digest")
    actual = _sha256(text)
    if digest != actual:
        _fail(hash_path, f"hash mismatch (expected {actual}, got {digest})")


def _check_optional_slice_fields(
    record: Mapping[str, Any],
    actual_text: str,
    path: str,
) -> None:
    if "source_text" in record:
        declared = _text(record, "source_text", path)
        if declared != actual_text:
            _fail(f"{path}.source_text", "does not equal the declared source slice")
    if "source_sha256" in record:
        _check_required_hash(record, actual_text, path)


def _check_offset_space(step: Mapping[str, Any], path: str) -> None:
    if "offset_space" not in step:
        return
    declared = _mapping(step.get("offset_space"), f"{path}.offset_space")
    if dict(declared) != OFFSET_SPACE:
        _fail(
            f"{path}.offset_space",
            f"must equal {OFFSET_SPACE!r}",
        )


def _validate_atoms(
    step: Mapping[str, Any],
    step_text: str,
    path: str,
    all_atom_ids: list[tuple[str, str]],
) -> dict[str, tuple[int, int]]:
    atoms = _list(step.get("atoms"), f"{path}.atoms")
    if not atoms:
        _fail(f"{path}.atoms", "must contain at least one atom")

    expected_start = 0
    atom_spans: dict[str, tuple[int, int]] = {}
    ordered_ids: list[str] = []
    for index, raw_atom in enumerate(atoms):
        atom_path = f"{path}.atoms[{index}]"
        atom = _mapping(raw_atom, atom_path)
        atom_id = _text(atom, "atom_id", atom_path)
        if not _ATOM_ID_RE.fullmatch(atom_id):
            _fail(f"{atom_path}.atom_id", "must match A followed by at least four digits")
        if atom_id in atom_spans:
            _fail(f"{atom_path}.atom_id", f"duplicate atom ID {atom_id}")

        kind = _text(atom, "kind", atom_path)
        if not kind:
            _fail(f"{atom_path}.kind", "must be non-empty")
        start = _offset(atom, "source_start", atom_path)
        end = _offset(atom, "source_end", atom_path)
        if start != expected_start:
            relation = "gap" if start > expected_start else "overlap"
            _fail(
                f"{atom_path}.source_start",
                f"atom {relation}: expected {expected_start}, got {start}",
            )
        if end <= start or end > len(step_text):
            _fail(
                f"{atom_path}.source_end",
                f"must satisfy {start} < end <= {len(step_text)}, got {end}",
            )
        atom_text = step_text[start:end]
        _check_required_hash(atom, atom_text, atom_path)
        if "source_text" in atom:
            declared_text = _text(atom, "source_text", atom_path)
            if declared_text != atom_text:
                _fail(
                    f"{atom_path}.source_text",
                    "does not equal the atom span in its step",
                )

        atom_spans[atom_id] = (start, end)
        ordered_ids.append(atom_id)
        all_atom_ids.append((atom_id, f"{atom_path}.atom_id"))
        expected_start = end

    if expected_start != len(step_text):
        _fail(
            f"{path}.atoms",
            f"do not cover the step tail [{expected_start}, {len(step_text)})",
        )
    if "atom_ids" in step:
        declared_ids = _list(step.get("atom_ids"), f"{path}.atom_ids")
        if list(declared_ids) != ordered_ids:
            _fail(f"{path}.atom_ids", "must exactly match atoms in source order")
    return atom_spans


def _validate_claim_atom_ids(
    claim: Mapping[str, Any],
    claim_start: int,
    claim_end: int,
    atom_spans: Mapping[str, tuple[int, int]],
    path: str,
) -> None:
    if "atom_ids" not in claim:
        return
    raw_ids = _list(claim.get("atom_ids"), f"{path}.atom_ids")
    atom_ids: list[str] = []
    for index, raw_id in enumerate(raw_ids):
        if not isinstance(raw_id, str):
            _fail(f"{path}.atom_ids[{index}]", "must be a string")
        if raw_id not in atom_spans:
            _fail(f"{path}.atom_ids[{index}]", f"unknown atom ID {raw_id}")
        if raw_id in atom_ids:
            _fail(f"{path}.atom_ids[{index}]", f"duplicate atom ID {raw_id}")
        atom_ids.append(raw_id)
    if not atom_ids:
        _fail(f"{path}.atom_ids", "must not be empty when present")

    ordered_step_ids = list(atom_spans)
    positions = [ordered_step_ids.index(atom_id) for atom_id in atom_ids]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        _fail(f"{path}.atom_ids", "must be consecutive and in source order")
    covered_start = atom_spans[atom_ids[0]][0]
    covered_end = atom_spans[atom_ids[-1]][1]
    if (covered_start, covered_end) != (claim_start, claim_end):
        _fail(
            f"{path}.atom_ids",
            "atom union must exactly equal the claim span",
        )


def _validate_claims(
    step: Mapping[str, Any],
    step_id: str,
    step_text: str,
    atom_spans: Mapping[str, tuple[int, int]],
    path: str,
    seen_claim_ids: set[str],
) -> tuple[list[str], dict[str, tuple[int, int, str, tuple[str, ...]]]]:
    claims = _list(step.get("claims"), f"{path}.claims")
    ordered_ids: list[str] = []
    by_id: dict[str, tuple[int, int, str, tuple[str, ...]]] = {}
    previous_end = -1
    for index, raw_claim in enumerate(claims, start=1):
        claim_path = f"{path}.claims[{index - 1}]"
        claim = _mapping(raw_claim, claim_path)
        claim_id = _text(claim, "claim_id", claim_path)
        expected_id = f"{step_id}.C{index:03d}"
        if not _CLAIM_ID_RE.fullmatch(claim_id) or claim_id != expected_id:
            _fail(
                f"{claim_path}.claim_id",
                f"claims must have canonical source-order IDs; expected {expected_id}",
            )
        if claim_id in seen_claim_ids:
            _fail(f"{claim_path}.claim_id", f"duplicate claim ID {claim_id}")

        start = _offset(claim, "source_start", claim_path)
        end = _offset(claim, "source_end", claim_path)
        if start < 0 or end <= start or end > len(step_text):
            _fail(
                f"{claim_path}.source_end",
                f"claim span must satisfy 0 <= {start} < end <= {len(step_text)}, got {end}",
            )
        if start < previous_end:
            _fail(
                f"{claim_path}.source_start",
                "claims overlap or are not in source order",
            )
        actual_text = step_text[start:end]
        claim_text = _text(claim, "source_text", claim_path)
        if claim_text != actual_text:
            _fail(
                f"{claim_path}.source_text",
                "does not equal the claim span in its step",
            )
        _check_required_hash(claim, actual_text, claim_path)
        _validate_claim_atom_ids(claim, start, end, atom_spans, claim_path)

        raw_scope_ids = claim.get("scope_ids", [])
        scope_ids: list[str] = []
        for scope_index, raw_scope_id in enumerate(
            _list(raw_scope_ids, f"{claim_path}.scope_ids")
        ):
            if not isinstance(raw_scope_id, str) or not _SCOPE_ID_RE.fullmatch(raw_scope_id):
                _fail(
                    f"{claim_path}.scope_ids[{scope_index}]",
                    "must be a canonical COT scope ID",
                )
            if raw_scope_id in scope_ids:
                _fail(
                    f"{claim_path}.scope_ids[{scope_index}]",
                    f"duplicate scope ID {raw_scope_id}",
                )
            scope_ids.append(raw_scope_id)

        seen_claim_ids.add(claim_id)
        ordered_ids.append(claim_id)
        by_id[claim_id] = (start, end, actual_text, tuple(scope_ids))
        previous_end = end
    return ordered_ids, by_id


def _validate_segments(
    step: Mapping[str, Any],
    step_id: str,
    step_text: str,
    ordered_claim_ids: Sequence[str],
    claims_by_id: Mapping[str, tuple[int, int, str, tuple[str, ...]]],
    path: str,
) -> None:
    segments = _list(step.get("segments"), f"{path}.segments")
    if not segments:
        _fail(f"{path}.segments", "must contain at least one segment")

    expected_start = 0
    encountered_claim_ids: list[str] = []
    expected_claim_scopes: dict[str, list[str]] = {
        claim_id: [] for claim_id in ordered_claim_ids
    }
    scope_count = 0
    for index, raw_segment in enumerate(segments):
        segment_path = f"{path}.segments[{index}]"
        segment = _mapping(raw_segment, segment_path)
        kind = _text(segment, "kind", segment_path)
        if kind not in {"context", "claim"}:
            _fail(f"{segment_path}.kind", "must be 'context' or 'claim'")
        start = _offset(segment, "source_start", segment_path)
        end = _offset(segment, "source_end", segment_path)
        if start != expected_start:
            relation = "gap" if start > expected_start else "overlap"
            _fail(
                f"{segment_path}.source_start",
                f"segment {relation}: expected {expected_start}, got {start}",
            )
        if end <= start or end > len(step_text):
            _fail(
                f"{segment_path}.source_end",
                f"must satisfy {start} < end <= {len(step_text)}, got {end}",
            )
        segment_text = step_text[start:end]
        _check_optional_slice_fields(segment, segment_text, segment_path)

        if kind == "context":
            if segment.get("claim_id") not in (None, ""):
                _fail(f"{segment_path}.claim_id", "context segment must not bind a claim")
            scope_id = segment.get("scope_id")
            applies_to = segment.get("applies_to_claim_ids")
            if scope_id in (None, ""):
                if applies_to not in (None, []):
                    _fail(
                        f"{segment_path}.applies_to_claim_ids",
                        "non-scope context must not bind claims",
                    )
            else:
                scope_count += 1
                expected_scope_id = f"{step_id}.G{scope_count:03d}"
                if not isinstance(scope_id, str) or scope_id != expected_scope_id:
                    _fail(
                        f"{segment_path}.scope_id",
                        f"scopes must have canonical source-order IDs; expected {expected_scope_id}",
                    )
                scope_type = segment.get("scope_type")
                if not isinstance(scope_type, str) or not scope_type:
                    _fail(f"{segment_path}.scope_type", "must be a non-empty string")
                target_ids = _list(applies_to, f"{segment_path}.applies_to_claim_ids")
                if not target_ids:
                    _fail(
                        f"{segment_path}.applies_to_claim_ids",
                        "scope must bind at least one following claim",
                    )
                for target_index, raw_target in enumerate(target_ids):
                    target_path = f"{segment_path}.applies_to_claim_ids[{target_index}]"
                    if not isinstance(raw_target, str) or raw_target not in claims_by_id:
                        _fail(target_path, f"unknown claim ID {raw_target!r}")
                    claim_start = claims_by_id[raw_target][0]
                    if claim_start < end:
                        _fail(target_path, "scope may bind only following claims")
                    if scope_id in expected_claim_scopes[raw_target]:
                        _fail(target_path, f"duplicate scope binding {scope_id}")
                    expected_claim_scopes[raw_target].append(scope_id)
        else:
            claim_id = _text(segment, "claim_id", segment_path)
            if claim_id not in claims_by_id:
                _fail(f"{segment_path}.claim_id", f"unknown claim ID {claim_id}")
            if claim_id in encountered_claim_ids:
                _fail(
                    f"{segment_path}.claim_id",
                    f"claim {claim_id} is represented by more than one segment",
                )
            claim_start, claim_end, claim_text, _scope_ids = claims_by_id[claim_id]
            if (start, end) != (claim_start, claim_end) or segment_text != claim_text:
                _fail(
                    segment_path,
                    f"span does not exactly match claim {claim_id}",
                )
            encountered_claim_ids.append(claim_id)
        expected_start = end

    if expected_start != len(step_text):
        _fail(
            f"{path}.segments",
            f"do not cover the step tail [{expected_start}, {len(step_text)})",
        )
    if encountered_claim_ids != list(ordered_claim_ids):
        missing = [
            claim_id for claim_id in ordered_claim_ids
            if claim_id not in encountered_claim_ids
        ]
        if missing:
            _fail(
                f"{path}.segments",
                f"claims missing a unique segment: {', '.join(missing)}",
            )
        _fail(f"{path}.segments", "claim segments are not in claim/source order")
    for claim_id in ordered_claim_ids:
        declared = list(claims_by_id[claim_id][3])
        expected = expected_claim_scopes[claim_id]
        if declared != expected:
            _fail(
                f"{path}.claims[{ordered_claim_ids.index(claim_id)}].scope_ids",
                f"must equal segment scope bindings {expected!r}",
            )
    declared_scope_count = step.get("scope_count")
    if declared_scope_count is not None and declared_scope_count != scope_count:
        _fail(
            f"{path}.scope_count",
            f"must equal the number of scoped context segments ({scope_count})",
        )


def validate_split_manifest(
    source: str,
    steps: Sequence[Mapping[str, Any]],
) -> None:
    """Validate a lossless structured split manifest.

    Success returns ``None``.  The first violation raises
    :class:`SplitManifestValidationError` with a stable field path suitable for
    logs and tests.  Equal claim texts and equal hashes are intentionally valid:
    identity is determined exclusively by canonical claim IDs and exact spans.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    manifest_steps = _list(steps, "steps")
    left = len(source) - len(source.lstrip())
    right = len(source.rstrip())
    stripped = source[left:right]
    if not stripped:
        if manifest_steps:
            _fail("steps", "empty source must have no steps")
        return
    if not manifest_steps:
        _fail("steps", "non-empty source must have at least one step")

    expected_global_start = left
    reconstructed: list[str] = []
    seen_step_ids: set[str] = set()
    seen_claim_ids: set[str] = set()
    all_atom_ids: list[tuple[str, str]] = []
    for index, raw_step in enumerate(manifest_steps, start=1):
        path = f"steps[{index - 1}]"
        step = _mapping(raw_step, path)
        step_id = _text(step, "step_id", path)
        expected_step_id = f"S{index:03d}"
        if not _STEP_ID_RE.fullmatch(step_id) or step_id != expected_step_id:
            _fail(
                f"{path}.step_id",
                f"steps must have canonical source-order IDs; expected {expected_step_id}",
            )
        if step_id in seen_step_ids:
            _fail(f"{path}.step_id", f"duplicate step ID {step_id}")
        seen_step_ids.add(step_id)
        _check_offset_space(step, path)

        start = _offset(step, "source_start", path)
        end = _offset(step, "source_end", path)
        if start != expected_global_start:
            relation = "gap" if start > expected_global_start else "overlap"
            _fail(
                f"{path}.source_start",
                f"step {relation}: expected global offset {expected_global_start}, got {start}",
            )
        if end <= start or end > right:
            _fail(
                f"{path}.source_end",
                f"must satisfy {start} < end <= stripped source end {right}, got {end}",
            )
        actual_step_text = source[start:end]
        step_text = _text(step, "source_text", path)
        if step_text != actual_step_text:
            _fail(
                f"{path}.source_text",
                "does not equal its global source span",
            )
        _check_required_hash(step, step_text, path)

        atom_spans = _validate_atoms(step, step_text, path, all_atom_ids)
        ordered_claim_ids, claims_by_id = _validate_claims(
            step,
            step_id,
            step_text,
            atom_spans,
            path,
            seen_claim_ids,
        )
        _validate_segments(
            step,
            step_id,
            step_text,
            ordered_claim_ids,
            claims_by_id,
            path,
        )

        requires = step.get("requires_formalization")
        if not isinstance(requires, bool):
            _fail(f"{path}.requires_formalization", "must be a boolean")
        if requires != bool(ordered_claim_ids):
            _fail(
                f"{path}.requires_formalization",
                "must be true exactly when the step has one or more claims",
            )

        reconstructed.append(step_text)
        expected_global_start = end

    if expected_global_start != right:
        _fail(
            "steps",
            f"do not cover the source tail [{expected_global_start}, {right})",
        )
    if "".join(reconstructed) != stripped:
        _fail("steps", "concatenated step text does not equal source.strip()")

    atom_count = len(all_atom_ids)
    width = max(4, len(str(atom_count)))
    for index, (atom_id, atom_path) in enumerate(all_atom_ids, start=1):
        expected_atom_id = f"A{index:0{width}d}"
        if atom_id != expected_atom_id:
            _fail(
                atom_path,
                f"atoms must have unique canonical global source-order IDs; "
                f"expected {expected_atom_id}",
            )


__all__ = [
    "OFFSET_SPACE",
    "SplitManifestValidationError",
    "validate_split_manifest",
]
