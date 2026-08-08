"""The sole persisted source contract for COT-grounded Blueprints."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_NAME = "cot_formal_steps"
SCHEMA_VERSION = 1
_STEP_ID_RE = re.compile(r"S[0-9]{3,}")


class FormalStepValidationError(ValueError):
    pass


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_formal_step_manifest(
    source: str,
    spans: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    if not isinstance(source, str) or not source:
        raise FormalStepValidationError("source must be a non-empty string")
    steps = []
    for index, (start, end) in enumerate(spans, start=1):
        text = source[start:end]
        steps.append({
            "step_id": f"S{index:03d}",
            "source_start": start,
            "source_end": end,
            "source_text": text,
            "source_sha256": _sha256(text),
        })
    return validate_formal_step_manifest({
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source_text": source,
        "source_sha256": _sha256(source),
        "steps": steps,
    }, source=source)


def validate_formal_step_manifest(
    value: Mapping[str, Any],
    *,
    source: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalStepValidationError("manifest must be an object")
    if value.get("schema") != SCHEMA_NAME:
        raise FormalStepValidationError(f"schema must be {SCHEMA_NAME!r}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise FormalStepValidationError(f"schema_version must be {SCHEMA_VERSION}")
    manifest_source = value.get("source_text")
    if not isinstance(manifest_source, str) or not manifest_source:
        raise FormalStepValidationError("source_text must be a non-empty string")
    if source is not None and source != manifest_source:
        raise FormalStepValidationError("manifest source differs from supplied COT")
    if value.get("source_sha256") != _sha256(manifest_source):
        raise FormalStepValidationError("source_sha256 mismatch")
    raw_steps = value.get("steps")
    if (
        isinstance(raw_steps, (str, bytes, bytearray))
        or not isinstance(raw_steps, Sequence)
        or not raw_steps
    ):
        raise FormalStepValidationError("steps must be a non-empty list")
    expected_start = 0
    checked_steps: list[dict[str, Any]] = []
    width = max(3, len(str(len(raw_steps))))
    for index, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, Mapping):
            raise FormalStepValidationError(f"steps[{index - 1}] must be an object")
        step_id = raw.get("step_id")
        expected_id = f"S{index:0{width}d}"
        if step_id != expected_id or not _STEP_ID_RE.fullmatch(str(step_id or "")):
            raise FormalStepValidationError(
                f"steps[{index - 1}].step_id must be {expected_id}"
            )
        start = raw.get("source_start")
        end = raw.get("source_end")
        if isinstance(start, bool) or not isinstance(start, int):
            raise FormalStepValidationError(f"{step_id}.source_start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise FormalStepValidationError(f"{step_id}.source_end must be an integer")
        if start != expected_start:
            raise FormalStepValidationError(
                f"{step_id} has a gap/overlap: expected start {expected_start}, got {start}"
            )
        if end <= start or end > len(manifest_source):
            raise FormalStepValidationError(f"invalid source span for {step_id}")
        text = manifest_source[start:end]
        if raw.get("source_text") != text:
            raise FormalStepValidationError(f"source_text mismatch for {step_id}")
        if raw.get("source_sha256") != _sha256(text):
            raise FormalStepValidationError(f"source_sha256 mismatch for {step_id}")
        checked_steps.append({
            "step_id": str(step_id),
            "source_start": start,
            "source_end": end,
            "source_text": text,
            "source_sha256": _sha256(text),
        })
        expected_start = end
    if expected_start != len(manifest_source):
        raise FormalStepValidationError(
            f"steps omit source tail [{expected_start}, {len(manifest_source)})"
        )
    if "".join(step["source_text"] for step in checked_steps) != manifest_source:
        raise FormalStepValidationError("step concatenation differs from source")
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source_text": manifest_source,
        "source_sha256": _sha256(manifest_source),
        "steps": checked_steps,
    }


def decode_formal_step_manifest(
    value: str | Mapping[str, Any],
    *,
    source: str | None = None,
) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FormalStepValidationError(f"invalid manifest JSON: {exc}") from exc
    return validate_formal_step_manifest(value, source=source)


def encode_formal_step_manifest(value: Mapping[str, Any]) -> str:
    return json.dumps(
        validate_formal_step_manifest(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "SCHEMA_NAME", "SCHEMA_VERSION", "FormalStepValidationError",
    "decode_formal_step_manifest", "encode_formal_step_manifest",
    "make_formal_step_manifest", "validate_formal_step_manifest",
]
