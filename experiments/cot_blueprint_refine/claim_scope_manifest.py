"""Minimal, lossless Claim/Scope manifest used by the fidelity experiment.

The persisted representation deliberately has no Step or Atom layer.  Atoms
may be used by an annotator as temporary coordinates, but only exact source
spans for mathematical claims and genuinely shared scopes survive here.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_NAME = "cot_claim_scope"
SCHEMA_VERSION = 1
_CLAIM_ID_RE = re.compile(r"C[0-9]{3,}")
_SCOPE_ID_RE = re.compile(r"G[0-9]{3,}")
_SCOPE_TYPE_RE = re.compile(r"[a-z][a-z0-9_]{1,63}")


class ClaimScopeValidationError(ValueError):
    pass


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fail(path: str, message: str) -> None:
    raise ClaimScopeValidationError(f"{path}: {message}")


def _records(value: Any, path: str) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(path, "must be a list")
    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            _fail(f"{path}[{index}]", "must be an object")
        rows.append(item)
    return rows


def _span(
    record: Mapping[str, Any],
    *,
    path: str,
    source: str,
) -> tuple[int, int, str]:
    start = record.get("source_start")
    end = record.get("source_end")
    if isinstance(start, bool) or not isinstance(start, int):
        _fail(f"{path}.source_start", "must be an integer")
    if isinstance(end, bool) or not isinstance(end, int):
        _fail(f"{path}.source_end", "must be an integer")
    if start < 0 or end <= start or end > len(source):
        _fail(path, f"invalid source span [{start}, {end})")
    actual = source[start:end]
    if record.get("source_text") != actual:
        _fail(f"{path}.source_text", "does not equal the declared source span")
    if record.get("source_sha256") != _sha256(actual):
        _fail(f"{path}.source_sha256", "hash mismatch")
    return start, end, actual


def validate_claim_scope_manifest(
    manifest: Any,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Validate and return a plain dict without changing any source bytes."""
    if not isinstance(manifest, Mapping):
        _fail("manifest", "must be an object")
    result = dict(manifest)
    if result.get("schema") != SCHEMA_NAME:
        _fail("manifest.schema", f"must equal {SCHEMA_NAME!r}")
    if result.get("schema_version") != SCHEMA_VERSION:
        _fail("manifest.schema_version", f"must equal {SCHEMA_VERSION}")
    stored_source = result.get("source_text")
    if not isinstance(stored_source, str) or not stored_source:
        _fail("manifest.source_text", "must be a non-empty string")
    if source is not None and stored_source != source.strip():
        _fail("manifest.source_text", "does not equal post_think_cot.strip()")
    if result.get("source_sha256") != _sha256(stored_source):
        _fail("manifest.source_sha256", "hash mismatch")

    claims = _records(result.get("claims"), "manifest.claims")
    scopes = _records(result.get("scopes"), "manifest.scopes")
    if not claims:
        _fail("manifest.claims", "must contain at least one mathematical claim")

    claim_spans: dict[str, tuple[int, int]] = {}
    claim_scopes: dict[str, tuple[str, ...]] = {}
    previous_start = -1
    for index, claim in enumerate(claims, start=1):
        path = f"manifest.claims[{index - 1}]"
        claim_id = claim.get("claim_id")
        expected_id = f"C{index:03d}"
        if claim_id != expected_id or not _CLAIM_ID_RE.fullmatch(str(claim_id or "")):
            _fail(f"{path}.claim_id", f"must equal {expected_id}")
        start, end, _text = _span(claim, path=path, source=stored_source)
        if start < previous_start:
            _fail(path, "claims must be in source order")
        previous_start = start
        raw_scope_ids = claim.get("scope_ids", [])
        if isinstance(raw_scope_ids, (str, bytes)) or not isinstance(raw_scope_ids, Sequence):
            _fail(f"{path}.scope_ids", "must be a list")
        scope_ids = tuple(str(value) for value in raw_scope_ids)
        if len(set(scope_ids)) != len(scope_ids):
            _fail(f"{path}.scope_ids", "contains duplicates")
        claim_spans[str(claim_id)] = (start, end)
        claim_scopes[str(claim_id)] = scope_ids

    scope_targets: dict[str, tuple[str, ...]] = {}
    scope_spans: dict[str, tuple[int, int]] = {}
    previous_start = -1
    for index, scope in enumerate(scopes, start=1):
        path = f"manifest.scopes[{index - 1}]"
        scope_id = scope.get("scope_id")
        expected_id = f"G{index:03d}"
        if scope_id != expected_id or not _SCOPE_ID_RE.fullmatch(str(scope_id or "")):
            _fail(f"{path}.scope_id", f"must equal {expected_id}")
        scope_type = str(scope.get("scope_type") or "")
        if not _SCOPE_TYPE_RE.fullmatch(scope_type):
            _fail(f"{path}.scope_type", "must be a stable lower_snake_case label")
        start, end, _text = _span(scope, path=path, source=stored_source)
        if start < previous_start:
            _fail(path, "scopes must be in source order")
        previous_start = start
        raw_targets = scope.get("applies_to_claim_ids")
        if isinstance(raw_targets, (str, bytes)) or not isinstance(raw_targets, Sequence):
            _fail(f"{path}.applies_to_claim_ids", "must be a list")
        targets = tuple(str(value) for value in raw_targets)
        if len(targets) < 2:
            _fail(
                f"{path}.applies_to_claim_ids",
                "a persisted scope must modify at least two claims; merge a single-use prefix into its claim",
            )
        if len(set(targets)) != len(targets):
            _fail(f"{path}.applies_to_claim_ids", "contains duplicates")
        unknown = [target for target in targets if target not in claim_spans]
        if unknown:
            _fail(f"{path}.applies_to_claim_ids", f"unknown claim IDs: {unknown}")
        if any(claim_spans[target][0] < end for target in targets):
            _fail(path, "a scope must precede every claim it modifies")
        scope_targets[str(scope_id)] = targets
        scope_spans[str(scope_id)] = (start, end)

    for claim_id, linked_scopes in claim_scopes.items():
        expected = tuple(
            scope_id for scope_id, targets in scope_targets.items() if claim_id in targets
        )
        if linked_scopes != expected:
            _fail(
                f"manifest.claims[{int(claim_id[1:]) - 1}].scope_ids",
                f"must equal the reverse scope links {list(expected)}",
            )

    labelled = [
        (start, end, claim_id) for claim_id, (start, end) in claim_spans.items()
    ] + [
        (start, end, scope_id) for scope_id, (start, end) in scope_spans.items()
    ]
    labelled.sort()
    for left, right in zip(labelled, labelled[1:]):
        if right[0] < left[1]:
            _fail("manifest", f"overlapping semantic spans: {left[2]} and {right[2]}")
    return result


def make_claim_scope_manifest(
    source: str,
    *,
    claims: Sequence[Mapping[str, Any]],
    scopes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    core = source.strip()
    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source_text": core,
        "source_sha256": _sha256(core),
        "claims": [dict(item) for item in claims],
        "scopes": [dict(item) for item in scopes],
    }
    return validate_claim_scope_manifest(manifest, source=core)


def decode_claim_scope_manifest(value: Any, *, source: str | None = None) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ClaimScopeValidationError(f"invalid manifest JSON: {exc}") from exc
    return validate_claim_scope_manifest(value, source=source)


def encode_claim_scope_manifest(manifest: Mapping[str, Any]) -> str:
    checked = validate_claim_scope_manifest(manifest)
    return json.dumps(checked, ensure_ascii=False, sort_keys=True)


def unassigned_spans(manifest: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    checked = validate_claim_scope_manifest(manifest)
    source = str(checked["source_text"])
    labelled = sorted(
        (int(item["source_start"]), int(item["source_end"]))
        for key in ("claims", "scopes")
        for item in checked[key]
    )
    gaps: list[tuple[int, int, str]] = []
    cursor = 0
    for start, end in labelled:
        if cursor < start:
            gaps.append((cursor, start, source[cursor:start]))
        cursor = end
    if cursor < len(source):
        gaps.append((cursor, len(source), source[cursor:]))
    return gaps


__all__ = [
    "ClaimScopeValidationError", "SCHEMA_NAME", "SCHEMA_VERSION",
    "decode_claim_scope_manifest", "encode_claim_scope_manifest",
    "make_claim_scope_manifest", "unassigned_spans",
    "validate_claim_scope_manifest",
]
