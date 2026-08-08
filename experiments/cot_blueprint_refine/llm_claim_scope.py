"""LLM annotation of a COT directly into the minimal Claim/Scope schema."""
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

from cot_blueprint_refine.claim_scope_manifest import make_claim_scope_manifest
from cot_blueprint_refine.llm_cot_splitter import atomize_cot


ANNOTATOR_VERSION = "llm-claim-scope-v4"
PROMPT_VERSION = "claim-scope-direct-v4"
ATOMIZER_VERSION = "cot-atomizer-v1"
RESULTS_FILENAME = "llm_claim_scope.jsonl"
SUMMARY_FILENAME = "llm_claim_scope_summary.json"
_OPEN = "[[CLAIM_SCOPE_V1]]"
_CLOSE = "[[/CLAIM_SCOPE_V1]]"
_ATOM_ID_RE = re.compile(r"A[0-9]{4,}")
_HEADING_ONLY_RE = re.compile(r"^\s{0,3}#{1,6}\s+[^\n]+\s*$")
_LIST_LABEL_ONLY_RE = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:step|case|part|term|row|column|answer|solution)"
    r"(?:\s+[A-Za-z0-9ivxIVX]+)?\s*:\s*$",
    re.IGNORECASE,
)
_METHOD_PREFIX_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:to solve|we (?:now )?(?:analy[sz]e|compute|consider|begin)|"
    r"let['’]?s (?:reconsider|compute|simplify|analy[sz]e)|"
    r"now (?:compute|expand|combine)|using [^:]+|the derivation involves|"
    r"understanding the problem|modeling the problem|strategy)\b",
    re.IGNORECASE,
)
_ASSERTION_SIGNAL_RE = re.compile(
    r"(?:\$|\\\[|\\\(|\\boxed|(?<![A-Za-z])[=<>≤≥≠](?![A-Za-z])|"
    r"\b(?:is|are|has|have|given|define[sd]?|denote[sd]?|equals?|therefore|hence|"
    r"implies?|exactly|only|must|cannot|satisf(?:y|ies)|divisible|prime|maximum|"
    r"minimum|largest|smallest)\b)",
    re.IGNORECASE,
)


class ClaimScopeFormatError(ValueError):
    def __init__(self, reason: str, raw_content: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_content = raw_content


@dataclass(frozen=True)
class ClaimScopeConfig:
    model: str
    openai_base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "EMPTY"
    concurrency: int = 24
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_s: float = 600.0
    max_format_attempts: int = 2
    enable_thinking: bool = False
    atomizer_version: str = ATOMIZER_VERSION
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("claim/scope model must be non-empty")
        if self.concurrency < 1 or self.max_tokens < 1 or self.timeout_s <= 0:
            raise ValueError("invalid claim/scope request limits")
        if not 1 <= self.max_format_attempts <= 2:
            raise ValueError("max_format_attempts must be 1 or 2")
        if self.atomizer_version != ATOMIZER_VERSION:
            raise ValueError(f"atomizer_version must equal {ATOMIZER_VERSION}")
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(f"prompt_version must equal {PROMPT_VERSION}")

    @classmethod
    def from_value(cls, value: Any) -> "ClaimScopeConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            data = dict(value)
        else:
            names = (
                "model", "openai_base_url", "base_url", "api_key", "concurrency",
                "temperature", "max_tokens", "timeout_s", "max_format_attempts",
                "max_attempts", "enable_thinking", "atomizer_version", "prompt_version",
            )
            data = {name: getattr(value, name) for name in names if hasattr(value, name)}
        if "base_url" in data and "openai_base_url" not in data:
            data["openai_base_url"] = data.pop("base_url")
        if "max_attempts" in data and "max_format_attempts" not in data:
            data["max_format_attempts"] = data.pop("max_attempts")
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: data[key] for key in data if key in allowed})


@dataclass
class ClaimScopeResult:
    row_id: str
    status: str
    source_sha256: str
    cache_key: str
    prompt_content_sha256: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    annotation_segments: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None
    error_type: str | None = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_artifact_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({
            "ID": self.row_id,
            "annotator_version": ANNOTATOR_VERSION,
            "prompt_version": PROMPT_VERSION,
            "atomizer_version": ATOMIZER_VERSION,
            "attempt_count": len(self.attempts),
            "finished_at": _utc_now(),
        })
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_claim_scope_messages(
    source: str,
    atoms: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    inventory = "\n".join(json.dumps({
        "atom_id": atom["atom_id"],
        "kind": atom.get("kind", "text"),
        "text": atom["source_text"],
    }, ensure_ascii=False) for atom in atoms)
    final_atom = str(atoms[-1]["atom_id"])
    system = f"""You annotate a mathematical chain-of-thought (COT) that may be wrong.
Preserve every assertion exactly as written. Never solve, repair, weaken, strengthen,
summarize, or discard a false, contradictory, unsupported, or answer-bearing claim.

Produce only the semantic organization needed for later formalization:

- claim: the smallest contiguous source span expressing one independently checkable,
  truth-evaluable mathematical proposition or transformation. A claim may be false. Equations,
  computed values, exhaustiveness statements, counts, existence/uniqueness claims,
  case conclusions, and the final answer are always claims. An introductory phrase
  used only by one claim must be INCLUDED in that claim, not made a scope.
- scope: a condition, definition, quantifier/domain restriction, notation binding,
  or case assumption that changes the interpretation of AT LEAST TWO later claims.
  It is never a complete mathematical conclusion. Use a concise lower_snake_case
  scope_type such as case_condition, shared_assumption, definition, domain_condition,
  quantifier, notation, or table_schema.
- context: headings, separators, transition prose, and other text with no mathematical
  assertion. Context is retained through the original source but is not formalized.

Classification constraints:
- Do NOT label a span as claim merely because it is an ATOM. Most headings, method
  announcements (`To solve...`, `we analyze...`, `we begin by...`), task restatements,
  bare lead-ins (`We are told:`, `Let us define:`, `The expression evaluates as:`),
  and list labels (`First term:`) are not independently truth-evaluable claims.
- A bare lead-in followed by one equation/item must share that claim segment when the
  lead-in is needed to read it; otherwise the lead-in is context. It must never become
  a standalone claim. Headings and separators normally remain context.
- Formula/display atoms ARE claims when they state a given, equality, transformation,
  computed value, or conclusion. Merge an immediately adjacent explanation only when
  it merely explains that same assertion; split it when it makes a new proposition.
- One segment may and often should contain several adjacent atoms that jointly express
  one proposition. Prefer context over manufacturing a Claim ID for non-propositional
  prose, but never use context to hide a mathematical assertion.

ATOMs are immutable temporary coordinates. Partition all atoms, in order, into adjacent
segments. Each segment object has `kind` and the ID of its final atom as `end`.
For a scope also emit `scope_type` and `through`, the atom ID ending the LAST later
claim modified by that scope. The host binds the scope to every intervening claim.
Every scope must cover at least two later claims. Do not emit IDs, source text, offsets,
explanations, Markdown fences, macro steps, dependencies, roles, or semantic summaries.

Boundary examples (the output still uses only atom endpoints):

1. `### Compute the value` is context. `We obtain:` followed by `$x=5$` is
   ONE claim ending at the formula atom, never two claims.
2. `We are given:` followed by `$a+b=7$` is ONE claim because the equation is
   a complete source assertion. A false equation is handled identically.
3. `We use induction.` is context; `Therefore n is even.` is a claim.
4. For a table, make the caption/column meaning a `table_schema` scope through
   all relevant rows. Each row may be a claim only when the schema plus row
   expresses a complete assertion. Never make the heading or colon lead-in a claim.

Before returning, silently verify that no claim is only a heading, method
announcement, colon-ending lead-in, or list label. Also verify that every
equation, computed value, false assertion, and final answer remains in a claim.

Return exactly one marker block containing one JSON array, for example:
{_OPEN}
[{{"kind":"scope","scope_type":"case_condition","through":"A0003","end":"A0001"}},{{"kind":"claim","end":"A0002"}},{{"kind":"claim","end":"A0003"}}]
{_CLOSE}
The example is format only. The final segment must end at {final_atom}."""
    user = (
        f"Source SHA-256: {_sha256(source)}\n"
        f"Atom count: {len(atoms)}\n"
        "Annotate the following immutable JSON-line source inventory.\n\n"
        + inventory
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _claim_boundary_issue(text: str) -> tuple[str, str]:
    """Classify only source spans that are certainly not propositions."""
    stripped = text.strip()
    compact = re.sub(r"\s+", " ", stripped)
    if _HEADING_ONLY_RE.fullmatch(stripped):
        return "HEADING_ONLY_CLAIM", "Markdown heading must be context."
    if _LIST_LABEL_ONLY_RE.fullmatch(stripped):
        return "LIST_LABEL_CLAIM", "Bare list/step label must be context."
    if compact.endswith(":") and not _ASSERTION_SIGNAL_RE.search(compact):
        return (
            "DANGLING_LEAD_IN_CLAIM",
            "Colon-ending lead-in must be context or merged with its following assertion.",
        )
    if _METHOD_PREFIX_RE.match(compact) and not _ASSERTION_SIGNAL_RE.search(compact):
        return (
            "METHOD_NARRATION_CLAIM",
            "Method announcement without a mathematical assertion must be context.",
        )
    return "", ""


def claim_scope_quality_issues(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return high-precision structural Claim defects suitable for one retry.

    These checks deliberately avoid judging mathematical correctness or whether
    a source assertion is useful.  They only reject spans that cannot stand as
    the persisted proposition because they are layout or an unfinished lead-in.
    """
    issues: list[dict[str, str]] = []
    for claim in manifest.get("claims") or []:
        claim_id = str(claim.get("claim_id") or "")
        text = str(claim.get("source_text") or "")
        code, detail = _claim_boundary_issue(text)
        if code:
            issues.append({"code": code, "claim_id": claim_id, "detail": detail})
    return issues


def _normalize_nonclaim_boundaries(
    segments: list[dict[str, Any]], source: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Repair high-confidence Claim boundaries without another LLM call."""
    normalized = [dict(segment) for segment in segments]
    warnings: list[str] = []
    remove: set[int] = set()
    for index, segment in enumerate(normalized):
        if segment.get("kind") != "claim":
            continue
        start = int(segment["source_start"])
        end = int(segment["source_end"])
        code, _detail = _claim_boundary_issue(source[start:end])
        if not code:
            continue
        if (
            code == "DANGLING_LEAD_IN_CLAIM"
            and index + 1 < len(normalized)
            and normalized[index + 1].get("kind") == "claim"
        ):
            normalized[index + 1]["source_start"] = start
            remove.add(index)
            warnings.append("merged_dangling_lead_in_into_next_claim")
            continue
        segment["kind"] = "context"
        segment.pop("claim_ordinal", None)
        warnings.append(f"demoted_{code.lower()}_to_context")
    return [segment for index, segment in enumerate(normalized) if index not in remove], warnings


def _extract_array(content: str) -> tuple[list[Any], str, str]:
    raw = str(content or "")
    if raw.count(_OPEN) != 1 or raw.count(_CLOSE) != 1:
        raise ClaimScopeFormatError("response must contain exactly one marker block", raw)
    start = raw.find(_OPEN) + len(_OPEN)
    end = raw.find(_CLOSE)
    if end < start:
        raise ClaimScopeFormatError("marker block is misordered", raw)
    try:
        value = json.loads(raw[start:end].strip())
    except json.JSONDecodeError as exc:
        raise ClaimScopeFormatError(f"invalid JSON array: {exc}", raw) from exc
    if not isinstance(value, list) or not value:
        raise ClaimScopeFormatError("annotation must be a non-empty JSON array", raw)
    return value, raw[:raw.find(_OPEN)], raw[end + len(_CLOSE):]


def parse_claim_scope_annotation(
    content: str,
    source: str,
    atoms: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    raw_segments, prefix, suffix = _extract_array(content)
    atom_ids = [str(atom["atom_id"]) for atom in atoms]
    positions = {atom_id: index for index, atom_id in enumerate(atom_ids)}
    cursor = 0
    segments: list[dict[str, Any]] = []
    claim_count = 0
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise ClaimScopeFormatError(f"segment {index + 1} must be an object", content)
        allowed = {"kind", "end", "scope_type", "through"}
        extra = sorted(set(raw) - allowed)
        if extra:
            raise ClaimScopeFormatError(f"segment {index + 1} has unknown keys: {extra}", content)
        kind = str(raw.get("kind") or "")
        if kind not in {"claim", "scope", "context"}:
            raise ClaimScopeFormatError(f"segment {index + 1} has invalid kind {kind!r}", content)
        end_id = str(raw.get("end") or "")
        if end_id not in positions:
            raise ClaimScopeFormatError(f"segment {index + 1} has unknown end atom {end_id!r}", content)
        final_atom = positions[end_id]
        if final_atom < cursor:
            raise ClaimScopeFormatError("segment endpoints must be strictly increasing", content)
        group = atoms[cursor:final_atom + 1]
        start = int(group[0]["source_start"])
        end = int(group[-1]["source_end"])
        segment: dict[str, Any] = {
            "kind": kind,
            "source_start": start,
            "source_end": end,
        }
        if kind == "claim":
            claim_count += 1
            segment["claim_ordinal"] = claim_count
            if "scope_type" in raw or "through" in raw:
                raise ClaimScopeFormatError("claim segments cannot carry scope fields", content)
        elif kind == "scope":
            scope_type = str(raw.get("scope_type") or "")
            through = str(raw.get("through") or "")
            if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", scope_type):
                raise ClaimScopeFormatError("scope_type must be lower_snake_case", content)
            if through not in positions or positions[through] <= final_atom:
                raise ClaimScopeFormatError("scope through must be a later atom ID", content)
            segment.update({
                "scope_type": scope_type,
                "through": through,
                "through_source_end": int(atoms[positions[through]]["source_end"]),
            })
        elif "scope_type" in raw or "through" in raw:
            raise ClaimScopeFormatError("context segments cannot carry scope fields", content)
        segments.append(segment)
        cursor = final_atom + 1
    if cursor != len(atoms):
        raise ClaimScopeFormatError(f"final segment must end at {atom_ids[-1]}", content)

    segments, boundary_warnings = _normalize_nonclaim_boundaries(segments, source)
    claims: list[dict[str, Any]] = []
    for segment in segments:
        if segment["kind"] != "claim":
            continue
        claim_id = f"C{len(claims) + 1:03d}"
        text = source[int(segment["source_start"]):int(segment["source_end"])]
        claims.append({
            "claim_id": claim_id,
            "source_start": int(segment["source_start"]),
            "source_end": int(segment["source_end"]),
            "source_text": text,
            "source_sha256": _sha256(text),
            "scope_ids": [],
        })
    if not claims:
        raise ClaimScopeFormatError("annotation contains no claims", content)

    scopes: list[dict[str, Any]] = []
    normalization_warnings: list[str] = list(boundary_warnings)
    for segment in segments:
        if segment["kind"] != "scope":
            continue
        scope_end = int(segment["source_end"])
        through_end = int(segment["through_source_end"])
        targets = [
            claim for claim in claims
            if int(claim["source_start"]) >= scope_end
            and int(claim["source_end"]) <= through_end
        ]
        if not targets:
            raise ClaimScopeFormatError("scope through covers no later claim", content)
        if len(targets) == 1:
            # The minimal persisted schema intentionally has no single-use
            # Scope.  The model has nevertheless identified the prefix as a
            # modifier, so losslessly fold its bytes into that sole Claim.
            # This is a structural normalization, not semantic inference.
            target = targets[0]
            target_start = int(target["source_start"])
            intervening_scope = any(
                other is not segment
                and other["kind"] == "scope"
                and int(other["source_start"]) >= scope_end
                and int(other["source_end"]) <= target_start
                for other in segments
            )
            if intervening_scope:
                raise ClaimScopeFormatError(
                    "single-use scope cannot be merged across another scope", content,
                )
            target["source_start"] = int(segment["source_start"])
            target_text = source[int(target["source_start"]):int(target["source_end"])]
            target["source_text"] = target_text
            target["source_sha256"] = _sha256(target_text)
            normalization_warnings.append(
                f"merged_single_use_scope_into_{target['claim_id']}"
            )
            continue
        scope_id = f"G{len(scopes) + 1:03d}"
        text = source[int(segment["source_start"]):scope_end]
        target_ids = [str(claim["claim_id"]) for claim in targets]
        scopes.append({
            "scope_id": scope_id,
            "scope_type": str(segment["scope_type"]),
            "applies_to_claim_ids": target_ids,
            "source_start": int(segment["source_start"]),
            "source_end": scope_end,
            "source_text": text,
            "source_sha256": _sha256(text),
        })
        for claim in targets:
            claim["scope_ids"].append(scope_id)
    manifest = make_claim_scope_manifest(source, claims=claims, scopes=scopes)
    warnings = list(normalization_warnings)
    if prefix.strip() or suffix.strip():
        warnings.append("content_outside_marker_block")
    return manifest, segments, warnings


def _message_parts(response: Any) -> tuple[str, str | None, str | None]:
    choice = response.choices[0]
    message = choice.message
    content = str(getattr(message, "content", None) or "")
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, Mapping):
            reasoning = extra.get("reasoning_content")
    finish = getattr(choice, "finish_reason", None)
    return content, None if reasoning is None else str(reasoning), None if finish is None else str(finish)


def _usage(response: Any) -> dict[str, int]:
    raw = getattr(response, "usage", None)
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(raw, key, None)
        if value is not None:
            result[key] = int(value)
    return result


def _aggregate_usage(attempts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for attempt in attempts:
        for key, value in dict(attempt.get("usage") or {}).items():
            total[str(key)] = total.get(str(key), 0) + int(value)
    return total


def _messages_hash(source: str, atoms: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(json.dumps(build_claim_scope_messages(source, atoms), ensure_ascii=False, sort_keys=True))


def claim_scope_cache_key(source: str, config: ClaimScopeConfig, atoms: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(json.dumps({
        "annotator_version": ANNOTATOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "atomizer_version": ATOMIZER_VERSION,
        "prompt_content_sha256": _messages_hash(source, atoms),
        "source_sha256": _sha256(source),
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "enable_thinking": config.enable_thinking,
    }, sort_keys=True))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _latest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row.get("row_id") or row.get("ID") or "")] = row
    return rows


def _cached_result(row: Mapping[str, Any], source: str, cache_key: str) -> ClaimScopeResult | None:
    if str(row.get("status") or "") != "ok" or str(row.get("cache_key") or "") != cache_key:
        return None
    try:
        manifest = make_claim_scope_manifest(
            source,
            claims=list(dict(row.get("manifest") or {}).get("claims") or []),
            scopes=list(dict(row.get("manifest") or {}).get("scopes") or []),
        )
    except (TypeError, ValueError):
        return None
    return ClaimScopeResult(
        row_id=str(row.get("row_id") or row.get("ID") or ""), status="ok",
        source_sha256=_sha256(source), cache_key=cache_key,
        prompt_content_sha256=str(row.get("prompt_content_sha256") or ""),
        manifest=manifest,
        annotation_segments=[dict(value) for value in (row.get("annotation_segments") or [])],
        attempts=[dict(value) for value in (row.get("attempts") or [])],
        usage={str(key): int(value) for key, value in dict(row.get("usage") or {}).items()},
        latency_s=float(row.get("latency_s") or 0.0), cached=True,
    )


async def _annotate_one(
    *, client: Any, semaphore: asyncio.Semaphore, row_id: str, source: str,
    atoms: list[dict[str, Any]], config: ClaimScopeConfig, cache_key: str,
) -> ClaimScopeResult:
    messages = build_claim_scope_messages(source, atoms)
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()
    final_error = "no response"
    final_type = "ClaimScopeFormatError"
    async with semaphore:
        request_messages = messages
        for attempt_number in range(1, config.max_format_attempts + 1):
            attempt_started = time.monotonic()
            attempt: dict[str, Any] = {"attempt": attempt_number, "started_at": _utc_now()}
            try:
                response = await client.chat.completions.create(
                    model=config.model, messages=request_messages,
                    temperature=config.temperature, max_tokens=config.max_tokens,
                    timeout=config.timeout_s,
                    extra_body={"chat_template_kwargs": {"enable_thinking": config.enable_thinking}},
                )
                content, reasoning, finish = _message_parts(response)
                attempt.update({
                    "raw_content": content, "reasoning_content": reasoning,
                    "finish_reason": finish, "response_id": str(getattr(response, "id", "") or ""),
                    "usage": _usage(response),
                })
                if finish == "length":
                    raise ClaimScopeFormatError("finish_reason=length", content)
                manifest, segments, warnings = parse_claim_scope_annotation(content, source, atoms)
                quality_issues = claim_scope_quality_issues(manifest)
                attempt["quality_issues"] = quality_issues
                if quality_issues:
                    rendered = "; ".join(
                        f"{issue['code']}[{issue['claim_id']}]: {issue['detail']}"
                        for issue in quality_issues[:20]
                    )
                    if len(quality_issues) > 20:
                        rendered += f"; and {len(quality_issues) - 20} more"
                    raise ClaimScopeFormatError(
                        "deterministic Claim quality gate rejected the annotation: " + rendered,
                        content,
                    )
                attempt.update({"status": "ok", "format_warnings": warnings})
                attempts.append(attempt)
                return ClaimScopeResult(
                    row_id=row_id, status="ok", source_sha256=_sha256(source),
                    cache_key=cache_key, prompt_content_sha256=_messages_hash(source, atoms),
                    manifest=manifest, annotation_segments=segments, attempts=attempts,
                    usage=_aggregate_usage(attempts), latency_s=time.monotonic() - started,
                )
            except ClaimScopeFormatError as exc:
                final_error, final_type = exc.reason, type(exc).__name__
                attempt.update({"status": "format_error", "validation_error": exc.reason, "error": exc.reason})
                if attempt_number < config.max_format_attempts:
                    request_messages = [*messages, {"role": "assistant", "content": str(attempt.get("raw_content") or exc.raw_content)}, {
                        "role": "user",
                        "content": (
                            f"The annotation was rejected: {exc.reason}\n"
                            "Regroup the complete source inventory, not just the named segment. "
                            "A heading or method announcement is context; a colon-ending lead-in "
                            "must be context or share one claim with its following equation/value. "
                            "Never discard a mathematical assertion, including a false one. "
                            f"Return only a corrected {_OPEN} JSON-array {_CLOSE} block."
                        ),
                    }]
            except Exception as exc:  # noqa: BLE001
                final_error, final_type = f"{type(exc).__name__}: {exc}", type(exc).__name__
                attempt.update({"status": "api_error", "error": final_error})
            finally:
                attempt["latency_s"] = time.monotonic() - attempt_started
                attempt["finished_at"] = _utc_now()
                if not attempts or attempts[-1] is not attempt:
                    attempts.append(attempt)
    return ClaimScopeResult(
        row_id=row_id, status="format_error", source_sha256=_sha256(source),
        cache_key=cache_key, prompt_content_sha256=_messages_hash(source, atoms),
        attempts=attempts, usage=_aggregate_usage(attempts),
        latency_s=time.monotonic() - started, error=final_error, error_type=final_type,
    )


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _write_summary(path: Path, results: Sequence[ClaimScopeResult], config: ClaimScopeConfig) -> None:
    live = [result for result in results if not result.cached]
    selected_attempts = [attempt for result in results for attempt in result.attempts]
    usage = _aggregate_usage(selected_attempts)
    latencies = [result.latency_s for result in live]
    payload = {
        "annotator_version": ANNOTATOR_VERSION, "prompt_version": PROMPT_VERSION,
        "generated_at": _utc_now(), "rows": len(results),
        "status_counts": {status: sum(result.status == status for result in results) for status in sorted({r.status for r in results})},
        "cached_rows": sum(result.cached for result in results),
        "request_attempts": sum(len(result.attempts) for result in live),
        "selected_artifact_usage": usage,
        "latency_s": {"sum": sum(latencies), "p50": _percentile(latencies, .5), "p90": _percentile(latencies, .9), "max": max(latencies) if latencies else None},
        "claims": {"total": sum(len(result.manifest.get("claims", [])) for result in results if result.ok)},
        "scopes": {"total": sum(len(result.manifest.get("scopes", [])) for result in results if result.ok)},
        "config": {**asdict(config), "api_key": "***"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


async def annotate_claim_scope_rows(
    rows: Iterable[Mapping[str, Any]], config: Any, artifact_root: str | Path,
    *, client: Any | None = None,
) -> dict[str, ClaimScopeResult]:
    cfg = ClaimScopeConfig.from_value(config)
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        row_id = str(row.get("name") or row.get("ID") or "")
        source = str(row.get("post_think_cot") or "").strip()
        if not row_id or row_id in seen or not source:
            raise ValueError(f"invalid or duplicate claim/scope row at index {index}: {row_id!r}")
        seen.add(row_id)
        normalized.append((row_id, source))
    root = Path(artifact_root)
    path = root / RESULTS_FILENAME
    cached = _latest(path)
    results: dict[str, ClaimScopeResult] = {}
    pending: list[tuple[str, str, list[dict[str, Any]], str]] = []
    for row_id, source in normalized:
        atoms = atomize_cot(source)
        cache_key = claim_scope_cache_key(source, cfg, atoms)
        hit = _cached_result(cached.get(row_id, {}), source, cache_key)
        if hit is not None:
            results[row_id] = hit
        else:
            pending.append((row_id, source, atoms, cache_key))
    owned_client = client is None
    if client is None and pending:
        client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.openai_base_url, max_retries=0)
    semaphore = asyncio.Semaphore(cfg.concurrency)
    artifact_lock = asyncio.Lock()

    async def run(item: tuple[str, str, list[dict[str, Any]], str]) -> tuple[str, ClaimScopeResult]:
        row_id, source, atoms, cache_key = item
        result = await _annotate_one(client=client, semaphore=semaphore, row_id=row_id, source=source, atoms=atoms, config=cfg, cache_key=cache_key)
        async with artifact_lock:
            _append_jsonl(path, result.to_artifact_row())
        return row_id, result

    try:
        if pending:
            results.update(await asyncio.gather(*(run(item) for item in pending)))
    finally:
        if owned_client and client is not None:
            await client.close()
    ordered = [results[row_id] for row_id, _source in normalized]
    _write_summary(root / SUMMARY_FILENAME, ordered, cfg)
    return results


__all__ = [
    "ANNOTATOR_VERSION", "ATOMIZER_VERSION", "PROMPT_VERSION", "RESULTS_FILENAME",
    "SUMMARY_FILENAME", "ClaimScopeConfig", "ClaimScopeFormatError",
    "ClaimScopeResult", "annotate_claim_scope_rows", "build_claim_scope_messages",
    "claim_scope_cache_key", "claim_scope_quality_issues", "parse_claim_scope_annotation",
]
