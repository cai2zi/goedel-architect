"""397B boundary-only splitter for formalization-aware COT Steps."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI


SPLITTER_VERSION = "formal-step-boundary-v1"
PROMPT_VERSION = "formal-step-boundary-v1"
ANCHOR_VERSION = "mechanical-boundary-anchor-v1"
RESULTS_FILENAME = "formal_step_splits.jsonl"
SUMMARY_FILENAME = "formal_step_split_summary.json"
_OPEN = "[[FORMAL_STEPS_V1]]"
_CLOSE = "[[/FORMAL_STEPS_V1]]"
_CLOSE_TOLERATED = "[/FORMAL_STEPS_V1]"
_ANCHOR_RE = re.compile(r"B[0-9]{4,}")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+|\*\*(?:step|case|part|answer|conclusion)\b)", re.I)
_LIST_RE = re.compile(r"^\s*(?:[-+*]|\d{1,3}[.)])\s+")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")


class FormalStepSplitError(ValueError):
    def __init__(self, reason: str, raw_content: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_content = raw_content


@dataclass(frozen=True)
class FormalStepSplitterConfig:
    model: str
    openai_base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "dummy"
    tokenizer_path: str = ""
    concurrency: int = 24
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 600.0
    max_attempts: int = 2
    enable_thinking: bool = False
    target_tokens_per_step: int = 100
    min_target_steps: int = 5
    max_target_steps: int = 16
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("step splitter model must be non-empty")
        if self.concurrency < 1 or self.max_tokens < 1 or self.timeout_s <= 0:
            raise ValueError("invalid step splitter request limits")
        if self.max_attempts not in {1, 2}:
            raise ValueError("max_attempts must be 1 or 2")
        if self.target_tokens_per_step < 1:
            raise ValueError("target_tokens_per_step must be positive")
        if not 1 <= self.min_target_steps <= self.max_target_steps:
            raise ValueError("invalid target Step range")
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError("configured prompt_version does not match implementation")

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | Any) -> "FormalStepSplitterConfig":
        data = dict(value) if isinstance(value, Mapping) else {
            key: getattr(value, key) for key in cls.__dataclass_fields__ if hasattr(value, key)
        }
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: data[key] for key in data if key in allowed})


@dataclass
class FormalStepSplitResult:
    row_id: str
    status: str
    source_sha256: str
    cache_key: str
    target_steps: int
    recommended_min: int
    recommended_max: int
    cot_tokens: int
    boundaries: list[str] = field(default_factory=list)
    anchors: list[dict[str, Any]] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None
    error_type: str | None = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def artifact(self) -> dict[str, Any]:
        result = asdict(self)
        result.update({
            "ID": self.row_id,
            "splitter_version": SPLITTER_VERSION,
            "prompt_version": PROMPT_VERSION,
            "anchor_version": ANCHOR_VERSION,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        return result


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _protected_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    patterns = [
        re.compile(r"`+[^`]*`+"),
        re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.S),
        re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]", re.S),
    ]
    for pattern in patterns:
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
        if char == "." and index > 0 and index + 1 < len(line):
            if line[index - 1].isdigit() and line[index + 1].isdigit():
                continue
        next_index = index + 1
        while next_index < len(line) and line[next_index].isspace():
            next_index += 1
        if next_index < len(line) and next_index > index + 1:
            starts.append(next_index)
    return sorted(set(starts))


def make_boundary_anchors(source: str) -> list[dict[str, Any]]:
    """Create dense mechanical boundary coordinates covering the exact COT."""
    if not isinstance(source, str) or not source:
        raise ValueError("source must be non-empty")
    lines = source.splitlines(keepends=True)
    if not lines:
        lines = [source]
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
    anchors = []
    for number, (start, kind) in enumerate(seeds, start=1):
        end = seeds[number][0] if number < len(seeds) else len(source)
        text = source[start:end]
        anchors.append({
            "anchor_id": f"B{number:0{width}d}",
            "source_start": start,
            "source_end": end,
            "source_text": text,
            "source_sha256": _sha256(text),
            "kind": kind,
        })
    if "".join(anchor["source_text"] for anchor in anchors) != source:
        raise AssertionError("boundary anchors do not reconstruct the exact COT")
    return anchors


@lru_cache(maxsize=2)
def _tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def step_count_prior(source: str, config: FormalStepSplitterConfig) -> tuple[int, int, int, int]:
    cot_tokens = (
        len(_tokenizer(config.tokenizer_path).encode(source, add_special_tokens=False))
        if config.tokenizer_path else max(1, math.ceil(len(source) / 4))
    )
    target = max(
        config.min_target_steps,
        min(config.max_target_steps, round(cot_tokens / config.target_tokens_per_step)),
    )
    return cot_tokens, target, max(4, target - 2), min(18, target + 2)


def build_split_messages(
    source: str,
    anchors: Sequence[Mapping[str, Any]],
    config: FormalStepSplitterConfig,
) -> list[dict[str, str]]:
    cot_tokens, target, lower, upper = step_count_prior(source, config)
    inventory = "\n".join(json.dumps({
        "boundary": anchor["anchor_id"],
        "kind": anchor["kind"],
        "text": anchor["source_text"],
    }, ensure_ascii=False) for anchor in anchors)
    final = anchors[-1]["anchor_id"]
    system = f"""You split a possibly wrong mathematical chain-of-thought into formalization Steps.
Do not solve, repair, summarize, reorder, omit, or rewrite any source content. Boundary labels are mechanical coordinates, not semantic atoms.

A Step is the smallest semantically coherent reasoning unit that can be represented by a small connected Lean Blueprint subgraph with one or more nodes. It normally contains a local setup, its needed conditions, one main inference/calculation, and its tightly connected result.

Boundary rules:
- Merge headings, method narration, colon-ended lead-ins, displayed formulas, and their immediate explanation into one Step. None may form a Step alone.
- Split independent conclusions, independent transformations, new case branches, or changes of mathematical object.
- Never split a formula from the prose that introduces or interprets it.
- Do not follow source labels like `Step 1` mechanically.
- Do not make one Step per sentence/list item. Several adjacent assertions that require the same local object model and form one inference belong together.
- Do not hide multiple independently falsifiable reasoning jumps inside one large Step.
- Preserve wrong and contradictory reasoning exactly; boundaries must not make it easier to prove.
- Every boundary coordinate must be used in order through adjacent groups; the final boundary must be {final}.

The COT has approximately {cot_tokens} tokens. A soft prior is {lower}–{upper} Steps, target {target}. Semantic coherence overrides this range; it is not a quota.

Return exactly one {_OPEN} ... {_CLOSE} block. Inside, write only the boundary ID ending each Step, one per line, strictly increasing. The final line must be {final}."""
    user = (
        f"Source SHA-256: {_sha256(source)}\n"
        f"Boundary coordinate count: {len(anchors)}\n\n{inventory}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_boundaries(content: str, anchors: Sequence[Mapping[str, Any]]) -> list[str]:
    raw = str(content or "")
    full_close_count = raw.count(_CLOSE)
    close_count = full_close_count or raw.count(_CLOSE_TOLERATED)
    if raw.count(_OPEN) != 1 or close_count != 1:
        raise FormalStepSplitError("response must contain exactly one marker block", raw)
    start = raw.find(_OPEN) + len(_OPEN)
    end = raw.find(_CLOSE)
    if end < 0:
        end = raw.find(_CLOSE_TOLERATED)
    if end < start:
        raise FormalStepSplitError("marker block is misordered", raw)
    values = [line.strip() for line in raw[start:end].strip().splitlines() if line.strip()]
    if not values or any(not _ANCHOR_RE.fullmatch(value) for value in values):
        raise FormalStepSplitError("marker block contains an invalid boundary", raw)
    ids = [str(anchor["anchor_id"]) for anchor in anchors]
    positions = {value: index for index, value in enumerate(ids)}
    if any(value not in positions for value in values):
        raise FormalStepSplitError("response contains an unknown boundary", raw)
    indices = [positions[value] for value in values]
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise FormalStepSplitError("boundaries must be unique and strictly increasing", raw)
    if values[-1] != ids[-1]:
        raise FormalStepSplitError(f"final boundary must be {ids[-1]}", raw)
    for value in values[:-1]:
        anchor = anchors[positions[value]]
        if str(anchor.get("kind")) == "heading":
            raise FormalStepSplitError(f"heading-only boundary is forbidden: {value}", raw)
        if str(anchor.get("source_text") or "").rstrip().endswith(":"):
            raise FormalStepSplitError(f"colon lead-in boundary is forbidden: {value}", raw)
    return values


def spans_from_boundaries(
    source: str,
    anchors: Sequence[Mapping[str, Any]],
    boundaries: Sequence[str],
) -> list[tuple[int, int]]:
    ids = {str(anchor["anchor_id"]): index for index, anchor in enumerate(anchors)}
    spans: list[tuple[int, int]] = []
    first = 0
    for boundary in boundaries:
        final = ids[boundary]
        spans.append((
            int(anchors[first]["source_start"]),
            int(anchors[final]["source_end"]),
        ))
        first = final + 1
    if first != len(anchors) or "".join(source[a:b] for a, b in spans) != source:
        raise FormalStepSplitError("boundaries do not exactly partition the COT")
    return spans


def _cache_key(source: str, config: FormalStepSplitterConfig, anchors: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "source_sha256": _sha256(source), "splitter": SPLITTER_VERSION,
        "prompt": PROMPT_VERSION, "model": config.model,
        "temperature": config.temperature, "max_tokens": config.max_tokens,
        "target_tokens_per_step": config.target_tokens_per_step,
        "min_target_steps": config.min_target_steps,
        "max_target_steps": config.max_target_steps,
        "anchors": [(a["anchor_id"], a["source_sha256"]) for a in anchors],
    }
    return _sha256(json.dumps(payload, sort_keys=True))


def _usage(response: Any) -> dict[str, int]:
    raw = getattr(response, "usage", None)
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(raw, key, None) if raw is not None else None
        if value is not None:
            result[key] = int(value)
    return result


def _latest(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[str(row.get("ID") or "")] = row
    return latest


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


async def _split_one(
    client: AsyncOpenAI,
    row: Mapping[str, Any],
    config: FormalStepSplitterConfig,
    cached_rows: Mapping[str, Mapping[str, Any]],
) -> FormalStepSplitResult:
    row_id = str(row.get("name") or row.get("ID") or "")
    source = str(row.get("post_think_cot") or "")
    anchors = make_boundary_anchors(source)
    cot_tokens, target, lower, upper = step_count_prior(source, config)
    cache_key = _cache_key(source, config, anchors)
    cached = cached_rows.get(row_id)
    if cached and cached.get("status") == "ok" and cached.get("cache_key") == cache_key:
        boundaries = parse_boundaries(
            f"{_OPEN}\n" + "\n".join(cached.get("boundaries") or []) + f"\n{_CLOSE}",
            anchors,
        )
        return FormalStepSplitResult(
            row_id, "ok", _sha256(source), cache_key, target, lower, upper, cot_tokens,
            boundaries=boundaries, anchors=anchors,
            spans=spans_from_boundaries(source, anchors, boundaries), cached=True,
        )
    base_messages = build_split_messages(source, anchors, config)
    messages = list(base_messages)
    attempts = []
    started = time.monotonic()
    final_error = ""
    final_type = "FormalStepSplitError"
    for attempt in range(1, config.max_attempts + 1):
        try:
            extra_body = {"chat_template_kwargs": {"enable_thinking": config.enable_thinking}}
            response = await client.chat.completions.create(
                model=config.model, messages=messages, temperature=config.temperature,
                max_completion_tokens=config.max_tokens, extra_body=extra_body,
            )
            choice = response.choices[0]
            content = str(choice.message.content or "")
            attempts.append({
                "attempt": attempt, "raw_content": content,
                "reasoning_content": getattr(choice.message, "reasoning_content", None),
                "finish_reason": choice.finish_reason, "response_id": response.id,
                "usage": _usage(response),
            })
            if str(choice.finish_reason or "") == "length":
                raise FormalStepSplitError("finish_reason=length", content)
            boundaries = parse_boundaries(content, anchors)
            usage = {}
            for item in attempts:
                for key, value in item["usage"].items():
                    usage[key] = usage.get(key, 0) + value
            return FormalStepSplitResult(
                row_id, "ok", _sha256(source), cache_key, target, lower, upper, cot_tokens,
                boundaries=boundaries, anchors=anchors,
                spans=spans_from_boundaries(source, anchors, boundaries), attempts=attempts,
                usage=usage, latency_s=time.monotonic() - started,
            )
        except FormalStepSplitError as exc:
            final_error, final_type = exc.reason, type(exc).__name__
            if attempt < config.max_attempts:
                correction = f"Rejected: {exc.reason}."
                if exc.reason.startswith("colon lead-in boundary is forbidden:"):
                    boundary = exc.reason.rsplit(":", 1)[-1].strip()
                    correction += (
                        f" Remove {boundary}; it introduces the material immediately after it. "
                        "Place that Step boundary only after the attached formula and its "
                        "immediate interpretation."
                    )
                elif exc.reason.startswith("heading-only boundary is forbidden:"):
                    boundary = exc.reason.rsplit(":", 1)[-1].strip()
                    correction += (
                        f" Remove {boundary}; merge that heading with the reasoning below it."
                    )
                messages = [*base_messages, {"role": "assistant", "content": exc.raw_content}, {
                    "role": "user",
                    "content": f"{correction} Return only one corrected {_OPEN} ... {_CLOSE} block.",
                }]
        except Exception as exc:  # API errors are retained as explicit failures.
            final_error, final_type = str(exc), type(exc).__name__
            break
    return FormalStepSplitResult(
        row_id, "error", _sha256(source), cache_key, target, lower, upper, cot_tokens,
        anchors=anchors, attempts=attempts, latency_s=time.monotonic() - started,
        error=final_error, error_type=final_type,
    )


async def split_formal_step_rows(
    rows: Sequence[Mapping[str, Any]],
    config_value: Mapping[str, Any] | Any,
    artifact_root: Path,
) -> dict[str, FormalStepSplitResult]:
    config = FormalStepSplitterConfig.from_value(config_value)
    result_path = artifact_root / RESULTS_FILENAME
    cached = _latest(result_path)
    client = AsyncOpenAI(
        api_key=config.api_key, base_url=config.openai_base_url.rstrip("/"),
        timeout=config.timeout_s,
    )
    semaphore = asyncio.Semaphore(config.concurrency)

    async def run(row: Mapping[str, Any]) -> FormalStepSplitResult:
        async with semaphore:
            result = await _split_one(client, row, config, cached)
            if not result.cached:
                _append(result_path, result.artifact())
            return result

    values = await asyncio.gather(*(run(row) for row in rows))
    await client.close()
    results = {result.row_id: result for result in values}
    status = {}
    for result in values:
        status[result.status] = status.get(result.status, 0) + 1
    summary = {
        "splitter_version": SPLITTER_VERSION, "prompt_version": PROMPT_VERSION,
        "anchor_version": ANCHOR_VERSION, "rows": len(values), "status_counts": status,
        "cached_rows": sum(result.cached for result in values),
        "steps": [len(result.spans) for result in values if result.ok],
        "target_steps": [result.target_steps for result in values],
        "usage": {
            key: sum(result.usage.get(key, 0) for result in values)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return results


__all__ = [
    "ANCHOR_VERSION", "PROMPT_VERSION", "SPLITTER_VERSION",
    "FormalStepSplitError", "FormalStepSplitterConfig", "FormalStepSplitResult",
    "build_split_messages", "make_boundary_anchors", "parse_boundaries",
    "spans_from_boundaries", "split_formal_step_rows", "step_count_prior",
]
