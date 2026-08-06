from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from datetime import datetime, timezone
from functools import lru_cache
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
    stable_name,
    write_json,
)


PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
FINAL_OPEN_RE = re.compile(r"<final_refined_solution\s*>", re.IGNORECASE)
FINAL_CLOSE_RE = re.compile(r"</final_refined_solution\s*>", re.IGNORECASE)
CONTEXT_QUALITIES = {"VERIFIED", "INVALID_BLUEPRINT_CANDIDATE", "INFRA_ERROR"}
INFRA_ERROR = "INFRA_ERROR"
CONVERSATION_SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def conversation_path(root: Path, row_id: str, variant_name: str = "blueprint") -> Path:
    return root / "refinement" / variant_name / "conversations" / f"{stable_name(row_id)}.json"


def _persist_conversation(root: Path, payload: dict[str, Any]) -> str:
    path = conversation_path(
        root,
        str(payload.get("ID") or ""),
        str(payload.get("refine_variant") or "blueprint"),
    )
    payload["updated_at"] = _utc_now()
    write_json(path, payload)
    return str(path)


def synthesize_legacy_conversation(
    refinement: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backfill a conversation artifact from pre-logging refinement rows."""
    context = context or {}
    messages = refinement.get("prompt")
    response = refinement.get("raw_response")
    event: dict[str, Any] = {
        "attempt": int(refinement.get("attempts") or 0),
        "status": "legacy_row_reconstructed",
        "started_at": None,
        "finished_at": None,
        "latency_s": refinement.get("latency_s"),
        "request": {
            "base_url": str(refinement.get("openai_base_url") or ""),
            "model": str(refinement.get("model") or ""),
            "messages": messages if isinstance(messages, list) else [],
            "temperature": refinement.get("temperature"),
            "max_tokens": refinement.get("effective_max_tokens"),
            "timeout_s": refinement.get("timeout_s"),
        },
        "response": response if isinstance(response, dict) else None,
        "assistant_content": str(refinement.get("raw_content") or ""),
        "assistant_reasoning_content": refinement.get("reasoning_content"),
        "finish_reason": refinement.get("finish_reason"),
        "normalization": {
            "status": str(refinement.get("status") or ""),
            "error": refinement.get("error"),
            "refined_cot": str(refinement.get("refined_cot") or ""),
            "think_stripped": bool(refinement.get("think_stripped")),
            "boxed_answer_count": refinement.get("boxed_answer_count"),
        },
        "exception": None,
        "reconstructed_from_refined_predictions": True,
    }
    if not messages and not response and refinement.get("status") not in {"ok", "invalid_output"}:
        event["status"] = "legacy_error_row_without_request_payload"
        event["exception"] = {
            "type": "UnknownLegacyError",
            "message": str(refinement.get("error") or ""),
            "traceback": "",
        }
    return {
        "schema_version": CONVERSATION_SCHEMA_VERSION,
        "ID": str(refinement.get("ID") or context.get("ID") or ""),
        "source": str(refinement.get("source") or context.get("source") or ""),
        "refine_variant": str(refinement.get("refine_variant") or "blueprint"),
        "prompt_mode": str(refinement.get("prompt_mode") or "blueprint"),
        "blueprint_used": bool(refinement.get("blueprint_used", True)),
        "source_solution_model_label": str(
            refinement.get("source_solution_model_label") or "Qwen3-8B"
        ),
        "context_quality": str(
            refinement.get("context_quality") or context.get("context_quality") or INFRA_ERROR
        ),
        "blueprint_context_status": str(
            refinement.get("blueprint_context_status") or context.get("status") or ""
        ),
        "created_at": None,
        "updated_at": _utc_now(),
        "events": [event],
        "reconstructed": True,
    }


def _render(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def _context_guidance(quality: str) -> str:
    if quality == "INVALID_BLUEPRINT_CANDIDATE":
        return (
            "This is the last blueprint candidate produced upstream. It did not pass Lean or "
            "the blueprint contract, but its decomposition may still be useful when refining the "
            "original solution. Use it as fallible reference together with the diagnostics, not as "
            "a machine-checked proof."
        )
    if quality == "INFRA_ERROR":
        return (
            "Formal checking did not complete because of an infrastructure failure. Any blueprint "
            "text below is reference only; independently verify the original solution."
        )
    return (
        "This blueprint context passed the export validation. Interpret each node according to its "
        "COT_BLUEPRINT_NODE_STATUS comment."
    )


def build_messages(
    row: dict[str, Any],
    *,
    prompt_mode: str = "blueprint",
    source_solution_model_label: str = "Qwen3-8B",
    lean_context: str | None = None,
) -> list[dict[str, str]]:
    if prompt_mode not in {"blueprint", "cot_only"}:
        raise ValueError(f"unknown refinement prompt mode: {prompt_mode!r}")
    common_system = (PROMPTS_DIR / "cot_refine_system_base.md").read_text(encoding="utf-8")
    arm_system = (PROMPTS_DIR / f"cot_refine_system_{prompt_mode}.md").read_text(
        encoding="utf-8"
    )
    system = _render(
        common_system.strip() + "\n\n" + arm_system.strip(),
        source_solution_model_label=source_solution_model_label,
    )
    user_template = (PROMPTS_DIR / f"cot_refine_user_{prompt_mode}.md").read_text(
        encoding="utf-8"
    )
    values = {
        "problem": str(row.get("problem") or ""),
        "claimed_answer": str(row.get("claimed_answer") or ""),
        "original_cot": str(row.get("original_cot") or ""),
        "source_solution_model_label": source_solution_model_label,
    }
    if prompt_mode == "blueprint":
        values.update({
            "lean_context": (
                str(row.get("lean_context") or "") if lean_context is None else lean_context
            ),
            "context_quality": str(row.get("context_quality") or "INFRA_ERROR"),
            "context_guidance": _context_guidance(
                str(row.get("context_quality") or "INFRA_ERROR")
            ),
            "blueprint_diagnostics": str(row.get("error") or "(none)"),
        })
    user = _render(user_template, **values).strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@lru_cache(maxsize=4)
def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _message_token_count(messages: list[dict[str, str]], tokenizer: Any) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = encoded.get("input_ids", []) if hasattr(encoded, "get") else encoded
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def _compress_verified_proofs(context: str, nodes: list[dict[str, Any]]) -> str:
    compressed = context
    replacement = "by\n  -- Verified proof omitted from the prompt to fit the context window.\n  sorry"
    for node in nodes:
        if node.get("prompt_signal") != "PROVED":
            continue
        proof = str(node.get("proof_body") or "")
        if len(proof) < 160:
            continue
        compressed = compressed.replace(proof, replacement, 1)
    return compressed


def _truncate_tokens_head_tail(text: str, token_budget: int, tokenizer: Any) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= token_budget:
        return text
    if token_budget <= 16:
        return tokenizer.decode(tokens[:max(0, token_budget)], skip_special_tokens=True)
    marker = "\n\n-- BLUEPRINT CONTENT OMITTED TO FIT THE MODEL CONTEXT --\n\n"
    marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
    available = max(1, token_budget - len(marker_tokens))
    head_count = max(1, int(available * 0.65))
    tail_count = max(0, available - head_count)
    kept = tokens[:head_count] + marker_tokens + (tokens[-tail_count:] if tail_count else [])
    return tokenizer.decode(kept, skip_special_tokens=True)


def fit_refinement_messages(
    row: dict[str, Any],
    config: DictConfig,
    *,
    prompt_mode: str,
    source_solution_model_label: str,
    extra_safety_tokens: int = 0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    tokenizer = _load_tokenizer(str(config.refine.tokenizer_path))
    context = str(row.get("lean_context") or "")
    original_context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
    max_input_tokens = (
        int(config.refine.context_window)
        - int(config.refine.max_tokens)
        - int(config.refine.context_safety_margin)
        - int(extra_safety_tokens)
    )
    if max_input_tokens <= 0:
        raise ValueError("refine context budget is non-positive")
    if prompt_mode == "cot_only":
        context = ""
        original_context_tokens = 0
    messages = build_messages(
        row,
        prompt_mode=prompt_mode,
        source_solution_model_label=source_solution_model_label,
        lean_context=context,
    )
    input_tokens = _message_token_count(messages, tokenizer)
    truncated = False
    if prompt_mode == "cot_only":
        if input_tokens > max_input_tokens:
            raise ValueError(
                f"cot-only prompt uses {input_tokens} tokens, exceeding input budget "
                f"{max_input_tokens}"
            )
        return messages, {
            "max_input_tokens": max_input_tokens,
            "blueprint_truncated": False,
            "blueprint_tokens_original": 0,
            "blueprint_tokens_used": 0,
            "input_tokens": input_tokens,
            "effective_max_tokens": int(config.refine.max_tokens),
        }
    if input_tokens > max_input_tokens:
        context = _compress_verified_proofs(context, list(row.get("nodes") or []))
        truncated = context != str(row.get("lean_context") or "")
        messages = build_messages(
            row,
            prompt_mode=prompt_mode,
            source_solution_model_label=source_solution_model_label,
            lean_context=context,
        )
        input_tokens = _message_token_count(messages, tokenizer)
    if input_tokens > max_input_tokens:
        empty_messages = build_messages(
            row,
            prompt_mode=prompt_mode,
            source_solution_model_label=source_solution_model_label,
            lean_context="",
        )
        fixed_tokens = _message_token_count(empty_messages, tokenizer)
        if fixed_tokens >= max_input_tokens:
            raise ValueError(
                f"non-blueprint prompt uses {fixed_tokens} tokens, exceeding input budget "
                f"{max_input_tokens}"
            )
        blueprint_budget = max(1, max_input_tokens - fixed_tokens - 16)
        context = _truncate_tokens_head_tail(context, blueprint_budget, tokenizer)
        truncated = True
        messages = build_messages(
            row,
            prompt_mode=prompt_mode,
            source_solution_model_label=source_solution_model_label,
            lean_context=context,
        )
        input_tokens = _message_token_count(messages, tokenizer)
        while input_tokens > max_input_tokens and blueprint_budget > 16:
            blueprint_budget = max(16, blueprint_budget - max(16, input_tokens - max_input_tokens))
            context = _truncate_tokens_head_tail(context, blueprint_budget, tokenizer)
            messages = build_messages(
                row,
                prompt_mode=prompt_mode,
                source_solution_model_label=source_solution_model_label,
                lean_context=context,
            )
            input_tokens = _message_token_count(messages, tokenizer)
    if input_tokens > max_input_tokens:
        raise ValueError(
            f"prompt remains too long after blueprint truncation: {input_tokens}>{max_input_tokens}"
        )
    return messages, {
        "max_input_tokens": max_input_tokens,
        "blueprint_truncated": truncated,
        "blueprint_tokens_original": original_context_tokens,
        "blueprint_tokens_used": len(tokenizer.encode(context, add_special_tokens=False)),
        "input_tokens": input_tokens,
        "effective_max_tokens": int(config.refine.max_tokens),
    }


def fit_messages_to_context(
    row: dict[str, Any],
    config: DictConfig,
    *,
    extra_safety_tokens: int = 0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Backward-compatible blueprint-arm wrapper used by existing tooling/tests."""
    return fit_refinement_messages(
        row,
        config,
        prompt_mode="blueprint",
        source_solution_model_label=str(
            config.refine.get("source_solution_model_label", "Qwen3-8B")
        ),
        extra_safety_tokens=extra_safety_tokens,
    )


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
    variant_name: str = "blueprint",
    variant_config: DictConfig | None = None,
) -> dict[str, Any]:
    root = output_root(config)
    prompt_mode = str(
        variant_config.get("prompt_mode", variant_name) if variant_config is not None else variant_name
    )
    source_solution_model_label = str(
        config.refine.get("source_solution_model_label", "Qwen3-8B")
    )
    base = {
        "ID": str(row.get("ID") or ""),
        "source": str(row.get("source") or ""),
        "problem": str(row.get("problem") or ""),
        "claimed_answer": str(row.get("claimed_answer") or ""),
        "root_proved": bool(row.get("root_proved")),
        "blueprint_context_status": str(row.get("status") or ""),
        "context_quality": str(row.get("context_quality") or "INFRA_ERROR"),
        "refine_variant": variant_name,
        "prompt_mode": prompt_mode,
        "blueprint_used": prompt_mode == "blueprint",
        "source_solution_model_label": source_solution_model_label,
    }
    conversation: dict[str, Any] = {
        "schema_version": CONVERSATION_SCHEMA_VERSION,
        **base,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "events": [],
        "reconstructed": False,
    }
    if prompt_mode == "blueprint" and not bool(row.get("refine_eligible", True)):
        conversation["events"].append({
            "attempt": 0,
            "status": "skipped",
            "started_at": _utc_now(),
            "finished_at": _utc_now(),
            "request": None,
            "response": None,
            "exception": None,
            "reason": str(row.get("error") or "blueprint context is not ready"),
        })
        artifact_path = _persist_conversation(root, conversation)
        return {
            **base,
            "status": "pipeline_error",
            "error": str(row.get("error") or "blueprint context is not ready"),
            "attempts": 0,
            "conversation_path": artifact_path,
        }

    timeout_s = config.refine.timeout_s
    start = time.time()
    last_error = ""
    fit_metadata: dict[str, Any] = {}
    extra_safety_tokens = 0
    async with semaphore:
        for attempt in range(1, int(config.refine.max_retries) + 1):
            attempt_start = time.time()
            event: dict[str, Any] = {
                "attempt": attempt,
                "status": "preparing_request",
                "started_at": _utc_now(),
                "finished_at": None,
                "latency_s": None,
                "fit_metadata": None,
                "request": None,
                "response": None,
                "assistant_content": "",
                "assistant_reasoning_content": None,
                "finish_reason": None,
                "normalization": None,
                "exception": None,
                "retry_delay_s": None,
            }
            conversation["events"].append(event)
            _persist_conversation(root, conversation)
            try:
                if prompt_mode == "blueprint":
                    messages, fit_metadata = fit_messages_to_context(
                        row,
                        config,
                        extra_safety_tokens=extra_safety_tokens,
                    )
                else:
                    messages, fit_metadata = fit_refinement_messages(
                        row,
                        config,
                        prompt_mode=prompt_mode,
                        source_solution_model_label=source_solution_model_label,
                        extra_safety_tokens=extra_safety_tokens,
                    )
                kwargs: dict[str, Any] = {
                    "model": str(config.refine.model),
                    "messages": messages,
                    "temperature": float(config.refine.temperature),
                    "max_tokens": int(config.refine.max_tokens),
                }
                if timeout_s is not None:
                    kwargs["timeout"] = float(timeout_s)
                event["fit_metadata"] = dict(fit_metadata)
                event["request"] = {
                    "base_url": str(config.refine.openai_base_url),
                    "model": str(config.refine.model),
                    "messages": messages,
                    "temperature": float(config.refine.temperature),
                    "max_tokens": int(config.refine.max_tokens),
                    "timeout_s": None if timeout_s is None else float(timeout_s),
                }
                event["status"] = "request_started"
                _persist_conversation(root, conversation)
                response = await client.chat.completions.create(**kwargs)
                content, reasoning, finish_reason = _message_parts(response)
                refined_cot, output_error, think_stripped, box_count = normalize_refined_output(
                    content, finish_reason,
                )
                status = "ok" if not output_error else "invalid_output"
                event.update({
                    "status": status,
                    "finished_at": _utc_now(),
                    "latency_s": time.time() - attempt_start,
                    "response": _response_json(response),
                    "assistant_content": content,
                    "assistant_reasoning_content": reasoning,
                    "finish_reason": finish_reason,
                    "normalization": {
                        "status": status,
                        "error": output_error or None,
                        "refined_cot": refined_cot,
                        "think_stripped": think_stripped,
                        "boxed_answer_count": box_count,
                    },
                })
                artifact_path = _persist_conversation(root, conversation)
                return {
                    **base,
                    **fit_metadata,
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
                    "temperature": float(config.refine.temperature),
                    "timeout_s": None if timeout_s is None else float(timeout_s),
                    "conversation_path": artifact_path,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                event.update({
                    "status": "exception",
                    "finished_at": _utc_now(),
                    "latency_s": time.time() - attempt_start,
                    "exception": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        ),
                    },
                })
                if "maximum context length" in last_error.lower():
                    extra_safety_tokens += 512
                if attempt >= int(config.refine.max_retries):
                    _persist_conversation(root, conversation)
                    break
                delay = min(
                    float(config.refine.retry_max_delay_s),
                    float(config.refine.retry_base_delay_s) * (2 ** (attempt - 1)),
                )
                event["retry_delay_s"] = delay
                _persist_conversation(root, conversation)
                await asyncio.sleep(delay)
    artifact_path = _persist_conversation(root, conversation)
    return {
        **base,
        **fit_metadata,
        "status": "error",
        "error": last_error,
        "refined_cot": "",
        "attempts": int(config.refine.max_retries),
        "latency_s": time.time() - start,
        "model": str(config.refine.model),
        "openai_base_url": str(config.refine.openai_base_url),
        "temperature": float(config.refine.temperature),
        "timeout_s": None if timeout_s is None else float(timeout_s),
        "conversation_path": artifact_path,
    }


async def refine(
    config: DictConfig,
    variant_name: str = "blueprint",
    variant_config: DictConfig | None = None,
) -> dict[str, Any]:
    root = output_root(config)
    if variant_config is None:
        variants = config.refine.get("variants")
        variant_config = variants.get(variant_name) if variants is not None else None
    prompt_mode = str(
        variant_config.get("prompt_mode", variant_name) if variant_config is not None else variant_name
    )
    contexts_path = root / "blueprint_contexts" / "blueprint_contexts.jsonl"
    contexts = latest_rows(contexts_path, "ID")
    contexts_by_id = {str(row.get("ID") or ""): row for row in contexts}
    output_path = root / "refinement" / variant_name / "refined_predictions.jsonl"
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
    tasks = [
        asyncio.create_task(
            _call_one(client, semaphore, row, config, variant_name, variant_config)
        )
        for row in pending
    ]
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for row in existing_by_id.values():
                if row.get("status") == "ok":
                    context = contexts_by_id.get(str(row.get("ID") or ""), {})
                    artifact = conversation_path(root, str(row.get("ID") or ""), variant_name)
                    if not artifact.exists():
                        legacy = synthesize_legacy_conversation(row, context)
                        _persist_conversation(root, legacy)
                    row = {
                        **row,
                        "refine_variant": variant_name,
                        "prompt_mode": prompt_mode,
                        "blueprint_used": prompt_mode == "blueprint",
                        "source_solution_model_label": str(
                            config.refine.get("source_solution_model_label", "Qwen3-8B")
                        ),
                        "context_quality": str(context.get("context_quality") or "INFRA_ERROR"),
                        "conversation_path": str(artifact),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            for task in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"cot-refine/{variant_name}",
                unit="row",
            ):
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
    quality_counts: dict[str, int] = {}
    invalid_ids: list[str] = []
    infra_ids: list[str] = []
    truncated_ids: list[str] = []
    for row in final_rows:
        row_id = str(row.get("ID") or "")
        context = contexts_by_id.get(row_id, {})
        quality = str(context.get("context_quality") or INFRA_ERROR)
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        if quality == "INVALID_BLUEPRINT_CANDIDATE":
            invalid_ids.append(row_id)
        elif quality == "INFRA_ERROR":
            infra_ids.append(row_id)
        if bool(row.get("blueprint_truncated")):
            truncated_ids.append(row_id)
    metrics = {
        "refine_variant": variant_name,
        "prompt_mode": prompt_mode,
        "rows": len(final_rows),
        "pending": len(pending),
        "counts": counts,
        "context_quality_counts": dict(sorted(quality_counts.items())),
        "invalid_blueprint_candidate_ids": sorted(invalid_ids),
        "infra_error_ids": sorted(infra_ids),
        "blueprint_truncated_count": len(truncated_ids),
        "blueprint_truncated_ids": sorted(truncated_ids),
        "output": str(output_path),
    }
    write_json(root / "refinement" / variant_name / "refinement_metrics.json", metrics)
    print(
        f"[refine/{variant_name}] rows={len(final_rows)} pending={len(pending)} counts={counts}",
        flush=True,
    )
    print(
        f"[refine-context] quality={metrics['context_quality_counts']} "
        f"invalid_ids={metrics['invalid_blueprint_candidate_ids']} "
        f"infra_ids={metrics['infra_error_ids']}",
        flush=True,
    )
    print(
        f"[refine-context] truncated={metrics['blueprint_truncated_count']} "
        f"ids={metrics['blueprint_truncated_ids']}",
        flush=True,
    )
    return metrics
