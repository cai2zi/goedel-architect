from __future__ import annotations

import asyncio
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from math_verify import verify
from omegaconf import DictConfig

from cot_blueprint_refine.common import (
    REPO_ROOT,
    claimed_answer,
    extract_boxed_spans,
    extract_post_think,
    latest_rows,
    output_root,
    read_jsonl,
    write_json,
    write_jsonl,
)
from cot_blueprint_refine.judge import judge_equivalences
from cot_blueprint_refine.run_cot_refinement import (  # noqa: E402
    conversation_path,
    synthesize_legacy_conversation,
)


MATH_EVAL_ROOT = REPO_ROOT.parent / "math_verify_eval"
sys.path.insert(0, str(MATH_EVAL_ROOT))
from run_math_verify_eval import (  # noqa: E402
    gold_candidates,
    grade_response,
    parse_math,
    string_match_correct,
)


ANALYSIS_PARQUET_NAME = "cot_blueprint_refine_analysis.parquet"
PIPELINE_CODE_NAME = "cot_blueprint_refine_pipeline_code.md"
ANALYSIS_PROMPT_NAME = "cot_blueprint_refine_full_analysis_prompt.md"
PIPELINE_TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md"}
COMPARISON_CSV_FIELDS = [
    "ID", "source", "gold",
    "before_extracted_pred", "before_math_verify_correct",
    "before_judge_status", "before_judge_equivalent",
    "before_judge_reason", "before_judge_error", "before_judge_error_layer",
    "before_judge_request_id", "before_judge_cache_hit",
    "before_correct",
    "after_extracted_pred", "after_math_verify_correct",
    "after_judge_status", "after_judge_equivalent",
    "after_judge_reason", "after_judge_error", "after_judge_error_layer",
    "after_judge_request_id", "after_judge_cache_hit",
    "after_correct", "transition",
    "before_whole_cot_math_verify_correct", "after_whole_cot_math_verify_correct",
    "blueprint_status", "context_quality", "root_proved", "refine_status",
    "blueprint_truncated",
]


def _canonical_answer_math_text(answer: str) -> str:
    """Put one answer field in an unambiguous Math-Verify extraction boundary."""
    answer = str(answer or "").strip()
    spans = extract_boxed_spans(answer)
    if len(spans) == 1 and spans[0][0] == 0 and spans[0][1] == len(answer):
        return answer
    return f"\\boxed{{{answer}}}" if answer else ""


def grade_final_answer(gold: str, candidate: str) -> dict[str, Any]:
    """Grade only the canonical last-box/claimed-answer field.

    `grade_response` deliberately remains available for whole-COT diagnostics,
    but its `any_match` extraction must not determine the experiment score.
    Boxing both sides also prevents a leading numeric subexpression (for
    example `10` in `10 + 40\\pi/3`) from becoming the parsed answer.
    """
    canonical_candidate = _canonical_answer_math_text(candidate)
    pred_parsed = parse_math(canonical_candidate)
    gold_parsed_groups = [
        parsed
        for answer in gold_candidates(str(gold or ""))
        if (parsed := parse_math(_canonical_answer_math_text(answer)))
    ]
    try:
        math_verify_correct = any(
            verify(gold_group, pred_parsed) for gold_group in gold_parsed_groups
        )
    except Exception:  # noqa: BLE001
        math_verify_correct = False
    return {
        "is_correct": bool(math_verify_correct),
        "string_match_correct": bool(
            string_match_correct(str(gold or ""), canonical_candidate)
        ),
        "extracted_pred": [str(item) for item in pred_parsed],
        "extracted_gold": [[str(item) for item in group] for group in gold_parsed_groups],
        "math_verify_parse_ok": bool(pred_parsed and gold_parsed_groups),
        "scoring_input": canonical_candidate,
        "scoring_mode": "canonical_claimed_answer",
    }


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None or value == "" else int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _read_or_reconstruct_conversation(
    root: Path,
    row_id: str,
    refinement: dict[str, Any],
    context: dict[str, Any],
    config: DictConfig,
) -> tuple[dict[str, Any], Path, str, bool]:
    path = conversation_path(root, row_id)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "recorded_full" if not payload.get("reconstructed") else "reconstructed_sidecar"
    else:
        reconstruction_row = {**refinement, "ID": row_id}
        payload = synthesize_legacy_conversation(reconstruction_row, context)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source = "reconstructed_from_refined_predictions"
    events = list(payload.get("events") or [])
    if payload.get("reconstructed"):
        for event in events:
            request = event.get("request")
            if not isinstance(request, dict):
                continue
            defaults = {
                "base_url": str(config.refine.openai_base_url),
                "model": str(config.refine.model),
                "temperature": float(config.refine.temperature),
                "max_tokens": int(config.refine.max_tokens),
                "timeout_s": (
                    None if config.refine.timeout_s is None else float(config.refine.timeout_s)
                ),
            }
            for key, value in defaults.items():
                if request.get(key) in {None, ""}:
                    request[key] = value
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def event_is_complete(event: dict[str, Any]) -> bool:
        if event.get("status") == "skipped":
            return True
        if event.get("exception"):
            return True
        return bool(event.get("request")) and isinstance(event.get("response"), dict)

    complete = bool(events) and all(event_is_complete(event) for event in events)
    expected_attempts = _int_or_none(refinement.get("attempts")) or 0
    if payload.get("reconstructed") and expected_attempts > len(events):
        complete = False
    return payload, path, source, complete


def _response_usage(refinement: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    response = refinement.get("raw_response") or {}
    usage = response.get("usage") if isinstance(response, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    prompt = _int_or_none(usage.get("prompt_tokens"))
    completion = _int_or_none(usage.get("completion_tokens"))
    total = _int_or_none(usage.get("total_tokens"))
    return prompt, completion, total


ANALYSIS_SCHEMA = pa.schema([
    ("ID", pa.string()),
    ("source", pa.string()),
    ("row_index", pa.int64()),
    ("problem", pa.string()),
    ("gold", pa.string()),
    ("claimed_answer", pa.string()),
    ("after_claimed_answer", pa.string()),
    ("original_raw_cot", pa.string()),
    ("original_post_think_cot", pa.string()),
    ("before_math_verify_correct", pa.bool_()),
    ("after_math_verify_correct", pa.bool_()),
    ("before_correct", pa.bool_()),
    ("after_correct", pa.bool_()),
    ("transition", pa.string()),
    ("before_parse_ok", pa.bool_()),
    ("after_parse_ok", pa.bool_()),
    ("before_extracted_pred_json", pa.string()),
    ("after_extracted_pred_json", pa.string()),
    ("before_whole_cot_math_verify_correct", pa.bool_()),
    ("after_whole_cot_math_verify_correct", pa.bool_()),
    ("before_whole_cot_parse_ok", pa.bool_()),
    ("after_whole_cot_parse_ok", pa.bool_()),
    ("before_whole_cot_extracted_pred_json", pa.string()),
    ("after_whole_cot_extracted_pred_json", pa.string()),
    ("before_judge_status", pa.string()),
    ("before_judge_equivalent", pa.bool_()),
    ("before_judge_reason", pa.string()),
    ("before_judge_error", pa.string()),
    ("before_judge_error_layer", pa.string()),
    ("before_judge_request_id", pa.string()),
    ("before_judge_cache_hit", pa.bool_()),
    ("before_judge_response_json", pa.string()),
    ("before_judge_audit_json", pa.string()),
    ("after_judge_status", pa.string()),
    ("after_judge_equivalent", pa.bool_()),
    ("after_judge_reason", pa.string()),
    ("after_judge_error", pa.string()),
    ("after_judge_error_layer", pa.string()),
    ("after_judge_request_id", pa.string()),
    ("after_judge_cache_hit", pa.bool_()),
    ("after_judge_response_json", pa.string()),
    ("after_judge_audit_json", pa.string()),
    ("judge_model", pa.string()),
    ("judge_base_url", pa.string()),
    ("blueprint_status", pa.string()),
    ("context_quality", pa.string()),
    ("context_error", pa.string()),
    ("root_proved", pa.bool_()),
    ("robustpa_status", pa.string()),
    ("robustpa_error", pa.string()),
    ("checkpoint_path", pa.string()),
    ("trace_path", pa.string()),
    ("blueprint_path", pa.string()),
    ("blueprint_candidate_source", pa.string()),
    ("node_count", pa.int64()),
    ("node_status_counts_json", pa.string()),
    ("blueprint_nodes_json", pa.string()),
    ("lean_context", pa.string()),
    ("refine_status", pa.string()),
    ("refine_error", pa.string()),
    ("refine_model", pa.string()),
    ("refine_base_url", pa.string()),
    ("refine_attempts", pa.int64()),
    ("refine_latency_s", pa.float64()),
    ("finish_reason", pa.string()),
    ("think_stripped", pa.bool_()),
    ("boxed_answer_count", pa.int64()),
    ("blueprint_truncated", pa.bool_()),
    ("blueprint_tokens_original", pa.int64()),
    ("blueprint_tokens_used", pa.int64()),
    ("input_tokens", pa.int64()),
    ("effective_max_tokens", pa.int64()),
    ("usage_prompt_tokens", pa.int64()),
    ("usage_completion_tokens", pa.int64()),
    ("usage_total_tokens", pa.int64()),
    ("refined_cot", pa.string()),
    ("raw_assistant_content", pa.string()),
    ("reasoning_content", pa.string()),
    ("prompt_messages_json", pa.string()),
    ("raw_response_json", pa.string()),
    ("conversation_path", pa.string()),
    ("conversation_source", pa.string()),
    ("conversation_complete", pa.bool_()),
    ("conversation_event_count", pa.int64()),
    ("conversation_request_count", pa.int64()),
    ("conversation_exception_count", pa.int64()),
    ("conversation_json", pa.string()),
    ("original_prediction_json", pa.string()),
    ("generation_input_json", pa.string()),
    ("blueprint_context_json", pa.string()),
    ("robustpa_result_json", pa.string()),
    ("refinement_result_json", pa.string()),
])


def _global_eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "") != "ok":
            continue
        if str(row.get("finish_reason") or "") == "length":
            continue
        post_think, reason = extract_post_think(str(row.get("raw_cot") or ""))
        if reason or not claimed_answer(post_think):
            continue
        eligible.append({**row, "post_think_cot": post_think})
    return eligible


def _source_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source") or "")].append(row)
    result: dict[str, dict[str, Any]] = {}
    for source, items in sorted(grouped.items()):
        before = sum(bool(row.get("before_correct")) for row in items)
        after = sum(bool(row.get("after_correct")) for row in items)
        before_math_verify = sum(
            bool(row.get("before_math_verify_correct", row.get("before_correct")))
            for row in items
        )
        after_math_verify = sum(
            bool(row.get("after_math_verify_correct", row.get("after_correct")))
            for row in items
        )
        total = len(items)
        result[source] = {
            "total": total,
            "before_math_verify_correct": before_math_verify,
            "before_math_verify_accuracy": before_math_verify / total if total else 0.0,
            "after_math_verify_correct": after_math_verify,
            "after_math_verify_accuracy": after_math_verify / total if total else 0.0,
            "before_correct": before,
            "before_accuracy": before / total if total else 0.0,
            "after_correct": after,
            "after_accuracy": after / total if total else 0.0,
            "refined_ok": sum(row.get("refine_status") == "ok" for row in items),
        }
    return result


def write_analysis_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=ANALYSIS_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def write_pipeline_code_snapshot(root: Path) -> Path:
    path = root / PIPELINE_CODE_NAME
    sections = [
        "# COT Blueprint Refine pipeline code snapshot",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This snapshot contains every text source/config/prompt file under "
        "`experiments/cot_blueprint_refine`, its directly invoked RobustPA/Lean/Math-Verify "
        "dependencies, and the resolved configuration used by this run.",
        "",
    ]
    files = sorted(
        candidate for candidate in (REPO_ROOT / "experiments" / "cot_blueprint_refine").rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in PIPELINE_TEXT_SUFFIXES
        and "__pycache__" not in candidate.parts
    )
    direct_dependencies = [
        REPO_ROOT / "experiments" / "robustpa_refine" / "run_robustpa_refine.py",
        REPO_ROOT / "experiments" / "robustpa_refine" / "io_utils.py",
        REPO_ROOT / "experiments" / "robustpa_refine" / "runtime.py",
        *[
            REPO_ROOT / "src" / name
            for name in (
                "blueprint.py", "blueprint_text.py", "checkpoint.py", "goedel_prompts.py",
                "kimina_lean_compiler.py", "llm_client.py", "mathlib_retrieval.py",
                "orchestrator.py", "pipeline.py", "prover.py", "refinement.py", "tracer.py",
            )
        ],
        *sorted((REPO_ROOT / "prompts").glob("*.md")),
        MATH_EVAL_ROOT / "run_math_verify_eval.py",
    ]
    files.extend(candidate for candidate in direct_dependencies if candidate.exists())
    language_by_suffix = {
        ".py": "python", ".sh": "bash", ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
    }
    for candidate in files:
        try:
            rel_label = candidate.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel_label = f"external/{candidate.parent.name}/{candidate.name}"
        sections.extend([
            f"## `{rel_label}`",
            "",
            f"~~~~{language_by_suffix.get(candidate.suffix.lower(), 'text')}",
            candidate.read_text(encoding="utf-8"),
            "~~~~",
            "",
        ])
    resolved_config = root / "config_resolved.yaml"
    if resolved_config.exists():
        sections.extend([
            "## `config_resolved.yaml` (experiment artifact)",
            "",
            "~~~~yaml",
            resolved_config.read_text(encoding="utf-8"),
            "~~~~",
            "",
        ])
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def write_full_analysis_prompt(root: Path, parquet_path: Path, code_path: Path) -> Path:
    path = root / ANALYSIS_PROMPT_NAME
    prompt = f"""# COT Blueprint Refinement 全量实验分析 Prompt

你是一名擅长数学推理评测、LLM 行为分析、Lean/Mathlib 与实验审计的研究员。现提供两个附件：

1. `{parquet_path.name}`：每个 eligible 样本一行的全量实验数据；包含原始题目/COT、blueprint 节点及 Lean context、RobustPA 状态、refinement 完整请求与响应对话、前后 Math-Verify 与 LLM judge 判分，以及逐样本 `transition`。
2. `{code_path.name}`：本次 `experiments/cot_blueprint_refine` 流水线的完整代码、配置、prompt，以及实际 resolved config。

请完整读取两个附件后再分析。Parquet 中以 `_json` 结尾的列是 JSON 字符串，必须解析；尤其必须读取 `conversation_json` 中的每个 event、完整 messages、reasoning/content、finish reason、normalization 与 exception。不要只依赖汇总字段。代码文件是解释分母、状态语义、截断、判分与错误处理的权威依据。

## 分析要求

1. 先做数据完整性与实现审计：核对样本数、ID 唯一性、三类 `context_quality`、refinement/对话覆盖率、缺失字段、重建会话比例、重试事件是否完整、token/finish reason、异常、截断样本，以及 before/after 的 Math-Verify、judge fallback 和最终 OR 判分代码。明确 gold 是否只在 evaluate 阶段读取，检查是否存在数据泄漏、错误分母或状态误解。
2. 给出全量定量结果：分别报告 Math-Verify-only 与 judge-assisted 的 before/after accuracy、绝对变化、相对变化、正确数净变化，以及四类 transition（correct→correct、correct→wrong、wrong→correct、wrong→wrong）的准确数量与比例。不得把 judge error、缺失或失败样本从分母中静默删除。
3. 分层比较：至少按 source、原始正确性、`context_quality`、`root_proved`、`robustpa_status`、refine status、blueprint 是否截断、节点数，以及 PROVED/NOT_PROVED/BLOCKED_BY_DEPENDENCY/FORMALLY_NEGATED 节点数量分析。小样本分组必须报告 n，避免过度解读。
4. 对所有 correct→wrong 和 wrong→correct 样本逐条审阅完整对话；对 wrong→wrong 做系统分类并抽取有代表性的案例。判断变化机制，例如 blueprint 成功纠错、未证明节点触发复核、无效 blueprint 误导/仍有帮助、模型独立重算、答案格式变化、推理退化、context overflow/truncation、Lean 语义与原题不匹配等。每个结论都要列出明确 ID 和原文证据。
5. 单独分析 `INVALID_BLUEPRINT_CANDIDATE` 与 `INFRA_ERROR` 的全部 ID；区分数学失败、形式化/contract 失败、基础设施失败和 refinement 失败。不要把 NOT_PROVED 解释成命题为假，也不要把 BLOCKED_BY_DEPENDENCY 解释成已经检查失败。
6. 分析效率与调用行为：attempts、latency、input/output token、finish reason、异常重试、blueprint 压缩/截断与结果质量的关系。若 usage 缺失，请明确缺失，不要估算成真实 token。
7. 审阅 refinement 与 judge prompt、结构化输出和缓存键是否真正实现了实验意图；分析 Math-Verify/judge 分歧，并指出任何会影响因果解释、公平性、可复现性或后续正式实验的实现风险。
8. 基于证据给出可执行改进建议。每项建议应关联具体失败类型和 ID，并区分“修复实验实现”“改进 blueprint/formalization”“改进 COT refinement prompt/decoding”“需要新增对照实验”。

## 输出格式

- 执行摘要：核心数值、最重要的收益/退化结论、结论可信度。
- 数据与实现审计。
- 总体与分层定量表格。
- Transition 全量分析：correct→wrong 和 wrong→correct 必须逐条列出 ID、before/after 答案、关键 blueprint 信号、对话证据与原因判断。
- 失败模式 taxonomy：判定标准、数量、ID 列表、典型原文证据。
- INVALID_BLUEPRINT_CANDIDATE / INFRA_ERROR 专项分析。
- 效率、token、重试与截断分析。
- 局限性与因果解释边界。
- 按优先级排序的改进建议和下一轮实验设计。

所有数值必须能由 Parquet 复算，所有实现判断必须能在代码快照中定位。请区分“数据直接证明”“代码表明”“相关性观察”“推测”，不要编造缺失信息。
"""
    path.write_text(prompt, encoding="utf-8")
    return path


def summarize_comparisons(
    comparisons: list[dict[str, Any]],
    *,
    dataset_total: int,
    global_eligible_total: int,
    global_before_correct: int,
    historical_raw_correct: int,
) -> dict[str, Any]:
    total = len(comparisons)
    before_correct = sum(bool(row.get("before_correct")) for row in comparisons)
    after_correct = sum(bool(row.get("after_correct")) for row in comparisons)
    before_math_verify = sum(
        bool(row.get("before_math_verify_correct", row.get("before_correct")))
        for row in comparisons
    )
    after_math_verify = sum(
        bool(row.get("after_math_verify_correct", row.get("after_correct")))
        for row in comparisons
    )
    transitions = Counter(str(row.get("transition") or "unknown") for row in comparisons)
    node_status_counts: Counter[str] = Counter()
    for row in comparisons:
        node_status_counts.update(row.get("node_status_counts") or {})
    full_run = total == global_eligible_total
    context_quality_counts = Counter(
        str(row.get("context_quality") or "INFRA_ERROR") for row in comparisons
    )
    invalid_ids = sorted(
        str(row.get("ID") or "")
        for row in comparisons
        if row.get("context_quality") == "INVALID_BLUEPRINT_CANDIDATE"
    )
    infra_ids = sorted(
        str(row.get("ID") or "")
        for row in comparisons
        if row.get("context_quality") == "INFRA_ERROR"
    )
    truncated_ids = sorted(
        str(row.get("ID") or "")
        for row in comparisons
        if bool(row.get("blueprint_truncated"))
    )
    judge_statuses = Counter(
        str(row.get(f"{side}_judge_status") or "legacy")
        for row in comparisons
        for side in ("before", "after")
    )
    judge_calls = sum(
        str(row.get(f"{side}_judge_status") or "") in {"ok", "error"}
        for row in comparisons
        for side in ("before", "after")
    )
    judge_equivalent = sum(
        row.get(f"{side}_judge_equivalent") is True
        for row in comparisons
        for side in ("before", "after")
    )
    judge_errors = sum(
        str(row.get(f"{side}_judge_status") or "") == "error"
        for row in comparisons
        for side in ("before", "after")
    )
    judge_cache_hits = sum(
        bool(row.get(f"{side}_judge_cache_hit"))
        for row in comparisons
        for side in ("before", "after")
    )
    return {
        "dataset": {
            "historical_baseline_scoring": (
                "legacy_whole_cot_math_verify_any_match_diagnostic_only"
            ),
            "current_math_verify_scoring": "canonical_claimed_answer_only",
            "total": dataset_total,
            "historical_raw_correct": historical_raw_correct,
            "historical_raw_accuracy": historical_raw_correct / dataset_total if dataset_total else 0.0,
            "global_eligible_total": global_eligible_total,
            "strict_post_think_before_correct": global_before_correct,
            "strict_post_think_before_eligible_accuracy": (
                global_before_correct / global_eligible_total if global_eligible_total else 0.0
            ),
            "strict_post_think_before_full_accuracy": (
                global_before_correct / dataset_total if dataset_total else 0.0
            ),
        },
        "selected": {
            "scoring_method": "math_verify_or_llm_judge",
            "total": total,
            "math_verify_only": {
                "before_correct": before_math_verify,
                "before_accuracy": before_math_verify / total if total else 0.0,
                "after_correct": after_math_verify,
                "after_accuracy": after_math_verify / total if total else 0.0,
                "accuracy_delta": (
                    (after_math_verify - before_math_verify) / total if total else 0.0
                ),
            },
            "before_correct": before_correct,
            "before_accuracy": before_correct / total if total else 0.0,
            "after_correct": after_correct,
            "after_accuracy": after_correct / total if total else 0.0,
            "accuracy_delta": (after_correct - before_correct) / total if total else 0.0,
            "blueprint_ready": sum(row.get("blueprint_status") == "ready" for row in comparisons),
            "refined_ok": sum(row.get("refine_status") == "ok" for row in comparisons),
            "transitions": dict(sorted(transitions.items())),
            "node_status_counts": dict(sorted(node_status_counts.items())),
            "context_quality_counts": dict(sorted(context_quality_counts.items())),
            "invalid_blueprint_candidate_ids": invalid_ids,
            "infra_error_ids": infra_ids,
            "blueprint_truncated_count": len(truncated_ids),
            "blueprint_truncated_ids": truncated_ids,
            "judge": {
                "calls": judge_calls,
                "equivalent": judge_equivalent,
                "errors": judge_errors,
                "cache_hits": judge_cache_hits,
                "status_counts": dict(sorted(judge_statuses.items())),
                "before_changed_to_correct": before_correct - before_math_verify,
                "after_changed_to_correct": after_correct - after_math_verify,
            },
        },
        "full_after": {
            "scoring_method": "math_verify_or_llm_judge",
            "available": full_run,
            "correct": after_correct if full_run else None,
            "math_verify_correct": after_math_verify if full_run else None,
            "eligible_accuracy": after_correct / global_eligible_total if full_run and global_eligible_total else None,
            "full_accuracy": after_correct / dataset_total if full_run and dataset_total else None,
            "incorrect_or_missing_count": dataset_total - after_correct if full_run else None,
            "unavailable_reason": "" if full_run else "partial/smoke run does not cover all eligible rows",
        },
        "by_source": _source_metrics(comparisons),
    }


def _judge_decision(
    *,
    math_verify_correct: bool,
    candidate: str,
    enabled: bool,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    if math_verify_correct:
        status = "not_needed"
    elif not enabled:
        status = "disabled"
    elif not candidate:
        status = "unavailable"
    elif result is None:
        status = "error"
        result = {"error": "judge result missing"}
    else:
        status = str(result.get("status") or "error")
    equivalent = result.get("equivalent") if result is not None and status == "ok" else None
    return {
        "status": status,
        "equivalent": equivalent,
        "reason": str((result or {}).get("reason") or ""),
        "error": str((result or {}).get("error") or ""),
        "error_layer": str((result or {}).get("error_layer") or ""),
        "request_id": str((result or {}).get("request_id") or ""),
        "cache_hit": bool((result or {}).get("cache_hit")),
        "raw_response": (result or {}).get("raw_response"),
        "audit": result,
        "correct": bool(math_verify_correct or equivalent is True),
    }


def evaluate(config: DictConfig) -> dict[str, Any]:
    root = output_root(config)
    prediction_path = Path(str(config.input_predictions)).expanduser()
    original_rows = latest_rows(prediction_path, "ID")
    original_by_id = {str(row["ID"]): row for row in original_rows}
    eligible_rows = _global_eligible(original_rows)
    global_before_correct = sum(
        grade_final_answer(
            str(row.get("gold") or ""),
            claimed_answer(str(row["post_think_cot"])),
        )["is_correct"]
        for row in eligible_rows
    )

    generation_rows = latest_rows(root / "prepared" / "generation_inputs.jsonl", "name")
    contexts = {
        str(row.get("ID") or ""): row
        for row in latest_rows(root / "blueprint_contexts" / "blueprint_contexts.jsonl", "ID")
    }
    robustpa_results = {
        str(row.get("source_id") or ""): row
        for row in latest_rows(root / "robustpa" / "blueprint" / "results.jsonl", "source_id")
    }
    refined = {
        str(row.get("ID") or ""): row
        for row in latest_rows(root / "refinement" / "refined_predictions.jsonl", "ID")
    }
    evaluation_inputs: list[dict[str, Any]] = []
    judge_requests: list[dict[str, Any]] = []
    judge_enabled = bool(config.judge.enabled)
    for generation in generation_rows:
        row_id = str(generation.get("name") or "")
        original = original_by_id.get(row_id)
        if original is None:
            raise ValueError(f"original prediction missing during evaluation: {row_id}")
        gold = str(original.get("gold") or "")
        before_text = str(generation.get("post_think_cot") or "")
        before_candidate = str(generation.get("claimed_answer") or claimed_answer(before_text))
        before = grade_final_answer(gold, before_candidate)
        before_whole_cot = grade_response(gold, before_text)
        refinement = refined.get(row_id, {})
        after_text = str(refinement.get("refined_cot") or "") if refinement.get("status") == "ok" else ""
        after_candidate = claimed_answer(after_text) if after_text else ""
        after = (
            grade_final_answer(gold, after_candidate)
            if after_text
            else {"is_correct": False, "math_verify_parse_ok": False, "extracted_pred": []}
        )
        after_whole_cot = (
            grade_response(gold, after_text)
            if after_text
            else {"is_correct": False, "math_verify_parse_ok": False, "extracted_pred": []}
        )
        item = {
            "row_id": row_id,
            "original": original,
            "generation": generation,
            "context": contexts.get(row_id, {}),
            "robustpa": robustpa_results.get(row_id, {}),
            "refinement": refinement,
            "before": before,
            "after": after,
            "before_whole_cot": before_whole_cot,
            "after_whole_cot": after_whole_cot,
            "before_candidate": before_candidate,
            "after_candidate": after_candidate,
        }
        evaluation_inputs.append(item)
        if judge_enabled:
            for side, grading, candidate in (
                ("before", before, before_candidate),
                ("after", after, after_candidate),
            ):
                if not bool(grading.get("is_correct")) and candidate:
                    judge_requests.append({
                        "ID": row_id,
                        "side": side,
                        "problem": str(original.get("problem") or ""),
                        "gold": gold,
                        "candidate": candidate,
                    })

    evaluation_dir = root / "evaluation"
    judge_path = evaluation_dir / "judge_results.jsonl"
    judgments = (
        asyncio.run(judge_equivalences(judge_requests, config, judge_path))
        if judge_enabled
        else {}
    )

    comparisons: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    for item in evaluation_inputs:
        row_id = item["row_id"]
        original = item["original"]
        generation = item["generation"]
        context = item["context"]
        robustpa = item["robustpa"]
        refinement = item["refinement"]
        before = item["before"]
        after = item["after"]
        before_whole_cot = item["before_whole_cot"]
        after_whole_cot = item["after_whole_cot"]
        before_judge = _judge_decision(
            math_verify_correct=bool(before.get("is_correct")),
            candidate=item["before_candidate"],
            enabled=judge_enabled,
            result=judgments.get((row_id, "before")),
        )
        after_judge = _judge_decision(
            math_verify_correct=bool(after.get("is_correct")),
            candidate=item["after_candidate"],
            enabled=judge_enabled,
            result=judgments.get((row_id, "after")),
        )
        before_correct = bool(before_judge["correct"])
        after_correct = bool(after_judge["correct"])
        transition = (
            "correct_to_correct" if before_correct and after_correct
            else "correct_to_wrong" if before_correct
            else "wrong_to_correct" if after_correct
            else "wrong_to_wrong"
        )
        node_counts = Counter(
            str(node.get("prompt_signal") or "") for node in (context.get("nodes") or [])
        )
        comparison = {
            "ID": row_id,
            "source": str(original.get("source") or ""),
            "problem": str(original.get("problem") or ""),
            "gold": str(original.get("gold") or ""),
            "claimed_answer": item["before_candidate"],
            "after_claimed_answer": item["after_candidate"],
            "before_math_verify_correct": bool(before.get("is_correct")),
            "after_math_verify_correct": bool(after.get("is_correct")),
            "before_correct": before_correct,
            "after_correct": after_correct,
            "transition": transition,
            "before_parse_ok": bool(before.get("math_verify_parse_ok")),
            "after_parse_ok": bool(after.get("math_verify_parse_ok")),
            "before_extracted_pred": before.get("extracted_pred", []),
            "after_extracted_pred": after.get("extracted_pred", []),
            "before_whole_cot_math_verify_correct": bool(
                before_whole_cot.get("is_correct")
            ),
            "after_whole_cot_math_verify_correct": bool(
                after_whole_cot.get("is_correct")
            ),
            "before_judge_status": before_judge["status"],
            "before_judge_equivalent": before_judge["equivalent"],
            "before_judge_reason": before_judge["reason"],
            "before_judge_error": before_judge["error"],
            "before_judge_error_layer": before_judge["error_layer"],
            "before_judge_request_id": before_judge["request_id"],
            "before_judge_cache_hit": before_judge["cache_hit"],
            "after_judge_status": after_judge["status"],
            "after_judge_equivalent": after_judge["equivalent"],
            "after_judge_reason": after_judge["reason"],
            "after_judge_error": after_judge["error"],
            "after_judge_error_layer": after_judge["error_layer"],
            "after_judge_request_id": after_judge["request_id"],
            "after_judge_cache_hit": after_judge["cache_hit"],
            "blueprint_status": str(context.get("status") or "missing"),
            "context_quality": str(context.get("context_quality") or "INFRA_ERROR"),
            "root_proved": bool(context.get("root_proved")),
            "refine_status": str(refinement.get("status") or "missing"),
            "blueprint_truncated": bool(refinement.get("blueprint_truncated")),
            "refined_cot": str(refinement.get("refined_cot") or ""),
            "node_status_counts": dict(sorted(node_counts.items())),
        }
        comparisons.append(comparison)

        conversation, conversation_file, conversation_source, conversation_complete = (
            _read_or_reconstruct_conversation(root, row_id, refinement, context, config)
        )
        events = list(conversation.get("events") or [])
        usage_prompt, usage_completion, usage_total = _response_usage(refinement)
        analysis_rows.append({
            "ID": row_id,
            "source": str(original.get("source") or ""),
            "row_index": _int_or_none(original.get("row_index")),
            "problem": str(original.get("problem") or ""),
            "gold": str(original.get("gold") or ""),
            "claimed_answer": item["before_candidate"],
            "after_claimed_answer": item["after_candidate"],
            "original_raw_cot": str(original.get("raw_cot") or ""),
            "original_post_think_cot": str(generation.get("post_think_cot") or ""),
            "before_math_verify_correct": bool(before.get("is_correct")),
            "after_math_verify_correct": bool(after.get("is_correct")),
            "before_correct": before_correct,
            "after_correct": after_correct,
            "transition": transition,
            "before_parse_ok": bool(before.get("math_verify_parse_ok")),
            "after_parse_ok": bool(after.get("math_verify_parse_ok")),
            "before_extracted_pred_json": _json_text(before.get("extracted_pred", [])),
            "after_extracted_pred_json": _json_text(after.get("extracted_pred", [])),
            "before_whole_cot_math_verify_correct": bool(
                before_whole_cot.get("is_correct")
            ),
            "after_whole_cot_math_verify_correct": bool(
                after_whole_cot.get("is_correct")
            ),
            "before_whole_cot_parse_ok": bool(
                before_whole_cot.get("math_verify_parse_ok")
            ),
            "after_whole_cot_parse_ok": bool(
                after_whole_cot.get("math_verify_parse_ok")
            ),
            "before_whole_cot_extracted_pred_json": _json_text(
                before_whole_cot.get("extracted_pred", [])
            ),
            "after_whole_cot_extracted_pred_json": _json_text(
                after_whole_cot.get("extracted_pred", [])
            ),
            "before_judge_status": before_judge["status"],
            "before_judge_equivalent": before_judge["equivalent"],
            "before_judge_reason": before_judge["reason"],
            "before_judge_error": before_judge["error"],
            "before_judge_error_layer": before_judge["error_layer"],
            "before_judge_request_id": before_judge["request_id"],
            "before_judge_cache_hit": before_judge["cache_hit"],
            "before_judge_response_json": _json_text(before_judge["raw_response"]),
            "before_judge_audit_json": _json_text(before_judge["audit"]),
            "after_judge_status": after_judge["status"],
            "after_judge_equivalent": after_judge["equivalent"],
            "after_judge_reason": after_judge["reason"],
            "after_judge_error": after_judge["error"],
            "after_judge_error_layer": after_judge["error_layer"],
            "after_judge_request_id": after_judge["request_id"],
            "after_judge_cache_hit": after_judge["cache_hit"],
            "after_judge_response_json": _json_text(after_judge["raw_response"]),
            "after_judge_audit_json": _json_text(after_judge["audit"]),
            "judge_model": str(config.judge.model) if judge_enabled else "",
            "judge_base_url": str(config.judge.openai_base_url) if judge_enabled else "",
            "blueprint_status": str(context.get("status") or "missing"),
            "context_quality": str(context.get("context_quality") or "INFRA_ERROR"),
            "context_error": str(context.get("error") or ""),
            "root_proved": bool(context.get("root_proved")),
            "robustpa_status": str(robustpa.get("status") or context.get("robustpa_status") or ""),
            "robustpa_error": str(robustpa.get("error") or ""),
            "checkpoint_path": str(robustpa.get("checkpoint_path") or context.get("checkpoint_path") or ""),
            "trace_path": str(robustpa.get("trace_path") or ""),
            "blueprint_path": str(robustpa.get("blueprint_dir") or ""),
            "blueprint_candidate_source": str(context.get("blueprint_candidate_source") or ""),
            "node_count": len(list(context.get("nodes") or [])),
            "node_status_counts_json": _json_text(dict(sorted(node_counts.items()))),
            "blueprint_nodes_json": _json_text(context.get("nodes") or []),
            "lean_context": str(context.get("lean_context") or ""),
            "refine_status": str(refinement.get("status") or "missing"),
            "refine_error": str(refinement.get("error") or ""),
            "refine_model": str(refinement.get("model") or ""),
            "refine_base_url": str(refinement.get("openai_base_url") or ""),
            "refine_attempts": _int_or_none(refinement.get("attempts")),
            "refine_latency_s": _float_or_none(refinement.get("latency_s")),
            "finish_reason": str(refinement.get("finish_reason") or ""),
            "think_stripped": bool(refinement.get("think_stripped")),
            "boxed_answer_count": _int_or_none(refinement.get("boxed_answer_count")),
            "blueprint_truncated": bool(refinement.get("blueprint_truncated")),
            "blueprint_tokens_original": _int_or_none(refinement.get("blueprint_tokens_original")),
            "blueprint_tokens_used": _int_or_none(refinement.get("blueprint_tokens_used")),
            "input_tokens": _int_or_none(refinement.get("input_tokens")),
            "effective_max_tokens": _int_or_none(refinement.get("effective_max_tokens")),
            "usage_prompt_tokens": usage_prompt,
            "usage_completion_tokens": usage_completion,
            "usage_total_tokens": usage_total,
            "refined_cot": str(refinement.get("refined_cot") or ""),
            "raw_assistant_content": str(refinement.get("raw_content") or ""),
            "reasoning_content": str(refinement.get("reasoning_content") or ""),
            "prompt_messages_json": _json_text(refinement.get("prompt") or []),
            "raw_response_json": _json_text(refinement.get("raw_response") or {}),
            "conversation_path": str(conversation_file),
            "conversation_source": conversation_source,
            "conversation_complete": conversation_complete,
            "conversation_event_count": len(events),
            "conversation_request_count": sum(bool(event.get("request")) for event in events),
            "conversation_exception_count": sum(bool(event.get("exception")) for event in events),
            "conversation_json": _json_text(conversation),
            "original_prediction_json": _json_text(original),
            "generation_input_json": _json_text(generation),
            "blueprint_context_json": _json_text(context),
            "robustpa_result_json": _json_text(robustpa),
            "refinement_result_json": _json_text(refinement),
        })
    comparisons.sort(key=lambda row: str(row["ID"]))
    analysis_rows.sort(key=lambda row: str(row["ID"]))

    metrics = summarize_comparisons(
        comparisons,
        dataset_total=len(original_rows),
        global_eligible_total=len(eligible_rows),
        global_before_correct=global_before_correct,
        historical_raw_correct=sum(bool(row.get("is_correct")) for row in original_rows),
    )
    write_jsonl(evaluation_dir / "comparison.jsonl", comparisons)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with (evaluation_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=COMPARISON_CSV_FIELDS,
        )
        writer.writeheader()
        for row in comparisons:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    with (evaluation_dir / "metrics_by_source.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source", "total", "before_math_verify_correct", "before_math_verify_accuracy",
            "after_math_verify_correct", "after_math_verify_accuracy",
            "before_correct", "before_accuracy",
            "after_correct", "after_accuracy", "refined_ok",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source, values in metrics["by_source"].items():
            writer.writerow({"source": source, **values})

    analysis_path = root / ANALYSIS_PARQUET_NAME
    write_analysis_parquet(analysis_path, analysis_rows)
    code_path = write_pipeline_code_snapshot(root)
    prompt_path = write_full_analysis_prompt(root, analysis_path, code_path)
    incomplete_conversation_ids = sorted(
        str(row["ID"]) for row in analysis_rows if not row["conversation_complete"]
    )
    reconstructed_conversation_ids = sorted(
        str(row["ID"])
        for row in analysis_rows
        if row["conversation_source"] != "recorded_full"
    )
    metrics["analysis_artifacts"] = {
        "parquet": str(analysis_path),
        "pipeline_code_markdown": str(code_path),
        "full_analysis_prompt": str(prompt_path),
        "parquet_rows": len(analysis_rows),
        "conversation_sidecar_dir": str(root / "refinement" / "conversations"),
        "conversation_complete_count": len(analysis_rows) - len(incomplete_conversation_ids),
        "conversation_incomplete_count": len(incomplete_conversation_ids),
        "conversation_incomplete_ids": incomplete_conversation_ids,
        "conversation_reconstructed_count": len(reconstructed_conversation_ids),
        "conversation_reconstructed_ids": reconstructed_conversation_ids,
        "judge_results_jsonl": str(judge_path) if judge_enabled else None,
    }
    write_json(evaluation_dir / "metrics.json", metrics)
    print(
        f"[evaluate] selected={metrics['selected']['total']} "
        f"before={metrics['selected']['before_accuracy']:.6f} "
        f"after={metrics['selected']['after_accuracy']:.6f} "
        f"full_after_available={metrics['full_after']['available']}",
        flush=True,
    )
    print(f"[evaluate-judge] {metrics['selected']['judge']}", flush=True)
    print(
        f"[evaluate-context] quality={metrics['selected']['context_quality_counts']} "
        f"invalid_ids={metrics['selected']['invalid_blueprint_candidate_ids']} "
        f"infra_ids={metrics['selected']['infra_error_ids']} "
        f"truncated_ids={metrics['selected']['blueprint_truncated_ids']}",
        flush=True,
    )
    print(
        f"[evaluate-analysis] parquet={analysis_path} rows={len(analysis_rows)} "
        f"conversations_complete={len(analysis_rows) - len(incomplete_conversation_ids)} "
        f"conversations_reconstructed={len(reconstructed_conversation_ids)}",
        flush=True,
    )
    print(
        f"[evaluate-analysis] code={code_path} prompt={prompt_path}",
        flush=True,
    )
    return metrics
