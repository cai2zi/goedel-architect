from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from omegaconf import DictConfig
from tqdm import tqdm

from cot_blueprint_refine.common import (
    EXPERIMENT_DIR,
    THINK_CLOSE_RE,
    THINK_OPEN_RE,
    extract_boxed_contents,
    extract_boxed_spans,
    extract_post_think,
    latest_rows,
    output_root,
    write_json,
)


PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
FINAL_OPEN_RE = re.compile(r"<final_refined_solution\s*>", re.IGNORECASE)
FINAL_CLOSE_RE = re.compile(r"</final_refined_solution\s*>", re.IGNORECASE)


def _render(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    system = (PROMPTS_DIR / "cot_refine_system.md").read_text(encoding="utf-8").strip()
    user_template = (PROMPTS_DIR / "cot_refine_user.md").read_text(encoding="utf-8")
    user = _render(
        user_template,
        problem=str(row.get("problem") or ""),
        claimed_answer=str(row.get("claimed_answer") or ""),
        original_cot=str(row.get("original_cot") or ""),
        lean_context=str(row.get("lean_context") or ""),
    ).strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _response_json(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "dict"):
        return response.dict()
    return json.loads(json.dumps(response, default=str))


def _message_parts(response: Any) -> tuple[str, str | None, str | None]:
    choice = response.choices[0]
    message = choice.message
    content = str(message.content or "")
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None and getattr(message, "model_extra", None):
        reasoning = message.model_extra.get("reasoning_content")
    return content, None if reasoning is None else str(reasoning), getattr(choice, "finish_reason", None)


def _canonicalize_boxed_answers(content: str) -> tuple[str, str, int]:
    spans = extract_boxed_spans(content)
    placeholders = {"...", r"\ldots", "…"}
    meaningful = [span for span in spans if span[2].strip() not in placeholders]
    if not meaningful:
        return "", "expected_one_boxed_answer:found_0", len(spans)
    distinct = {span[2].strip() for span in meaningful}
    if len(distinct) != 1:
        return "", f"conflicting_boxed_answers:found_{len(distinct)}", len(spans)
    keep = meaningful[-1]
    normalized = content
    for start, end, value in reversed(spans):
        if (start, end, value) == keep:
            continue
        normalized = normalized[:start] + value + normalized[end:]
    return normalized, "", len(spans)


def normalize_refined_output(content: str, finish_reason: str | None) -> tuple[str, str, bool, int]:
    if finish_reason == "length":
        return "", "finish_reason_length", False, 0
    final_opens = list(FINAL_OPEN_RE.finditer(content))
    if final_opens:
        final_open = final_opens[-1]
        final_close = FINAL_CLOSE_RE.search(content, final_open.end())
        if final_close is None:
            return "", "unclosed_final_refined_solution", False, 0
        content = content[final_open.end():final_close.start()]
    think_stripped = False
    leading = content.lstrip()
    leading_open = THINK_OPEN_RE.match(leading)
    if leading_open:
        cleaned, reason = extract_post_think(leading)
        if reason == "unclosed_think":
            content = leading[leading_open.end():].strip()
        elif reason:
            return "", f"invalid_think_output:{reason}", False, 0
        else:
            content = cleaned
        think_stripped = True
    content = content.strip()
    if not content:
        return "", "empty_response", think_stripped, 0
    content, box_error, raw_box_count = _canonicalize_boxed_answers(content)
    if box_error:
        return "", box_error, think_stripped, raw_box_count
    if len(extract_boxed_contents(content)) != 1:
        return "", "boxed_answer_normalization_failed", think_stripped, raw_box_count
    return content, "", think_stripped, raw_box_count


async def _call_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    config: DictConfig,
) -> dict[str, Any]:
    base = {
        "ID": str(row.get("ID") or ""),
        "source": str(row.get("source") or ""),
        "problem": str(row.get("problem") or ""),
        "claimed_answer": str(row.get("claimed_answer") or ""),
        "root_proved": bool(row.get("root_proved")),
        "blueprint_context_status": str(row.get("status") or ""),
    }
    if row.get("status") != "ready" or not row.get("lean_validated"):
        return {
            **base,
            "status": "pipeline_error",
            "error": str(row.get("error") or "blueprint context is not ready"),
            "attempts": 0,
        }

    messages = build_messages(row)
    timeout_s = config.refine.timeout_s
    start = time.time()
    last_error = ""
    async with semaphore:
        for attempt in range(1, int(config.refine.max_retries) + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": str(config.refine.model),
                    "messages": messages,
                    "temperature": float(config.refine.temperature),
                    "max_tokens": int(config.refine.max_tokens),
                }
                if timeout_s is not None:
                    kwargs["timeout"] = float(timeout_s)
                response = await client.chat.completions.create(**kwargs)
                content, reasoning, finish_reason = _message_parts(response)
                refined_cot, output_error, think_stripped, box_count = normalize_refined_output(
                    content, finish_reason,
                )
                status = "ok" if not output_error else "invalid_output"
                return {
                    **base,
                    "status": status,
                    "error": output_error or None,
                    "refined_cot": refined_cot,
                    "raw_content": content,
                    "reasoning_content": reasoning,
                    "raw_response": _response_json(response),
                    "finish_reason": finish_reason,
                    "think_stripped": think_stripped,
                    "boxed_answer_count": box_count,
                    "prompt": messages,
                    "attempts": attempt,
                    "latency_s": time.time() - start,
                    "model": str(config.refine.model),
                    "openai_base_url": str(config.refine.openai_base_url),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                if attempt >= int(config.refine.max_retries):
                    break
                delay = min(
                    float(config.refine.retry_max_delay_s),
                    float(config.refine.retry_base_delay_s) * (2 ** (attempt - 1)),
                )
                await asyncio.sleep(delay)
    return {
        **base,
        "status": "error",
        "error": last_error,
        "refined_cot": "",
        "attempts": int(config.refine.max_retries),
        "latency_s": time.time() - start,
        "model": str(config.refine.model),
        "openai_base_url": str(config.refine.openai_base_url),
    }


async def refine(config: DictConfig) -> dict[str, Any]:
    root = output_root(config)
    contexts_path = root / "blueprint_contexts" / "blueprint_contexts.jsonl"
    contexts = latest_rows(contexts_path, "ID")
    output_path = root / "refinement" / "refined_predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = latest_rows(output_path, "ID") if bool(config.resume) and output_path.exists() else []
    existing_by_id = {str(row.get("ID") or ""): row for row in existing}
    pending = [
        row for row in contexts
        if str(row.get("ID") or "") not in existing_by_id
        or existing_by_id[str(row.get("ID") or "")].get("status") != "ok"
    ]
    if not bool(config.resume):
        existing_by_id = {}
    client = AsyncOpenAI(
        api_key=str(config.refine.api_key),
        base_url=str(config.refine.openai_base_url).rstrip("/"),
    )
    semaphore = asyncio.Semaphore(int(config.refine.concurrency))
    tasks = [asyncio.create_task(_call_one(client, semaphore, row, config)) for row in pending]
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for row in existing_by_id.values():
                if row.get("status") == "ok":
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="cot-refine", unit="row"):
                result = await task
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        await client.close()
    final_rows = latest_rows(output_path, "ID")
    counts: dict[str, int] = {}
    for row in final_rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    metrics = {"rows": len(final_rows), "pending": len(pending), "counts": counts, "output": str(output_path)}
    write_json(root / "refinement" / "refinement_metrics.json", metrics)
    print(f"[refine] rows={len(final_rows)} pending={len(pending)} counts={counts}", flush=True)
    return metrics
