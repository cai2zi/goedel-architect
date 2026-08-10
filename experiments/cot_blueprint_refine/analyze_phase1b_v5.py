#!/usr/bin/env python3
"""Build the machine-readable and Markdown reports for the v5 smoke run."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_OLD = Path(
    "/ssd/czx/czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_step_v4_phase1_ab"
)
DEFAULT_NEW = Path(
    "/ssd/czx/czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_step_v5_phase1_ab_semantic_judge_smoke27"
)
DEFAULT_SELECTION = Path(__file__).with_name("configs") / (
    "qwen3_8b_397b_wrong76_step_v5_phase1_ab_semantic_judge_smoke27_manifest.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def result_map(root: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(root / "robustpa" / "blueprint" / "results.jsonl")
    return {str(row["source_id"]): row for row in rows}


def manifest_check(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("cot_manifest_json")
    if not raw:
        return {"present": False}
    value = json.loads(raw)
    source = str(value.get("source_text", ""))
    steps = list(value.get("steps", []))
    cursor = 0
    exact = True
    for step in steps:
        start = int(step.get("source_start", -1))
        end = int(step.get("source_end", -1))
        text = str(step.get("source_text", ""))
        exact = exact and start == cursor and source[start:end] == text
        cursor = end
    forbidden = {"requires_formalization", "role", "duplicates_step_id"}
    return {
        "present": True,
        "schemaVersion": value.get("schema_version"),
        "stepCount": len(steps),
        "continuousCoverage": exact and cursor == len(source),
        "hasExcludedFields": any(forbidden.intersection(step) for step in steps),
        "finalStepId": steps[-1].get("step_id") if steps else None,
        "finalStepText": steps[-1].get("source_text", "") if steps else "",
    }


def trace_stats(row: dict[str, Any]) -> dict[str, Any]:
    trace_path = Path(str(row.get("trace_path") or ""))
    events = read_jsonl(trace_path)
    ends = [event for event in events if event.get("kind") == "llm_request_end"]
    starts = [event for event in events if event.get("kind") == "llm_request_start"]
    tool_starts = [event for event in events if event.get("kind") == "tool_call"]
    tool_ends = [event for event in events if event.get("kind") == "tool_result"]
    searches = [event for event in events if event.get("kind") == "phase1BMathlibSearchResult"]
    validations = [event for event in events if event.get("kind") == "phase1BValidationResult"]
    judge_starts = [event for event in events if event.get("kind") == "phase1BSemanticJudgeStart"]
    judges = [event for event in events if event.get("kind") == "phase1BSemanticJudgeResult"]
    judge_ends = [event for event in events if event.get("kind") == "phase1BSemanticJudgeEnd"]
    judge_requests = [
        event for event in ends
        if (event.get("args") or {}).get("operation") == "phase1b_semantic_judge"
    ]
    canonical = [event for event in events if event.get("kind") == "phase1ACanonicalCheck"]
    standalone = [event for event in events if event.get("kind") == "phase2StandaloneCheckEnd"]
    unique_judges: list[dict[str, Any]] = []
    for event in judges:
        args = event.get("args") or {}
        if not args.get("cacheHit"):
            unique_judges.append(args)
    verdicts = [bool(item.get("passed")) for item in unique_judges]
    fail_to_pass = any(
        not verdicts[index] and any(verdicts[index + 1 :])
        for index in range(len(verdicts))
    )
    search_rounds = {int((event.get("args") or {}).get("round", -1)) for event in searches}
    edit_history = list(row.get("phase1b_edit_history") or [])
    edit_rounds = {
        int(item.get("round", -1))
        for item in edit_history
        if item.get("accepted")
    }
    search_then_edit = bool(search_rounds.intersection(edit_rounds))
    finish_reasons = Counter(str((event.get("args") or {}).get("finish_reason")) for event in ends)
    timestamps = [float(event["ts"]) for event in events if event.get("ts") is not None]
    llm_start_spans = Counter(event.get("span_id") for event in starts if event.get("span_id"))
    llm_end_spans = Counter(event.get("span_id") for event in ends if event.get("span_id"))
    tool_start_spans = Counter(event.get("span_id") for event in tool_starts if event.get("span_id"))
    tool_end_spans = Counter(event.get("span_id") for event in tool_ends if event.get("span_id"))
    return {
        "tracePath": str(trace_path),
        "eventCount": len(events),
        "llmRequestCount": len(ends),
        "llmStartCount": len(starts),
        "llmUnpairedSpanCount": sum((llm_start_spans - llm_end_spans).values()) + sum((llm_end_spans - llm_start_spans).values()),
        "toolStartCount": len(tool_starts),
        "toolEndCount": len(tool_ends),
        "toolUnpairedSpanCount": sum((tool_start_spans - tool_end_spans).values()) + sum((tool_end_spans - tool_start_spans).values()),
        "promptTokens": sum(int((event.get("args") or {}).get("prompt_tokens", 0)) for event in ends),
        "completionTokens": sum(int((event.get("args") or {}).get("completion_tokens", 0)) for event in ends),
        "totalTokens": sum(int((event.get("args") or {}).get("total_tokens", 0)) for event in ends),
        "finishReasons": dict(finish_reasons),
        "lengthFinishCount": finish_reasons.get("length", 0),
        "durationSeconds": max(timestamps) - min(timestamps) if timestamps else 0.0,
        "phase1ACanonicalChecks": len(canonical),
        "phase1ACanonicalPassed": any(event.get("ok") is True for event in canonical),
        "phase1AStandaloneChecks": sum(
            1 for event in standalone if (event.get("args") or {}).get("phase") == "phase1A"
        ),
        "validationCount": len(validations),
        "maxValidationRound": max(
            (int((event.get("args") or {}).get("round", 0)) for event in validations),
            default=0,
        ),
        "searchCallCount": len(searches),
        "searchCacheHitCount": sum(
            bool((event.get("args") or {}).get("cacheHit")) for event in searches
        ),
        "searchQueries": [
            {
                "round": (event.get("args") or {}).get("round"),
                "query": (event.get("args") or {}).get("query"),
                "targetNodeNames": (event.get("args") or {}).get("targetNodeNames", []),
                "cacheHit": bool((event.get("args") or {}).get("cacheHit")),
            }
            for event in searches
        ],
        "searchThenEdit": search_then_edit,
        "judgeAuditCount": len(judge_starts),
        "judgeCallCount": len(judge_requests),
        "judgeParsedCandidateCount": len(unique_judges),
        "judgeVerdicts": verdicts,
        "judgeFailToPass": fail_to_pass,
        "judgeFormatRetryCount": sum(
            max(0, int((event.get("args") or {}).get("attemptCount", 0)) - 1)
            for event in judge_ends
            if not (event.get("args") or {}).get("cacheHit")
        ),
        "judgeLengthCount": sum(
            str((event.get("args") or {}).get("finish_reason", "")).lower() == "length"
            for event in judge_requests
        ),
        "judgeResults": unique_judges,
    }


def classify_failure(row: dict[str, Any]) -> str:
    if row.get("status") != "error":
        return "accepted"
    error = str(row.get("error") or "")
    if "Phase 1A" in error:
        return "phase1A"
    if "semantic Judge response was invalid" in error:
        return "semanticJudgeFormat"
    if "Phase 1B failed" in error:
        return "phase1BRoundsExhausted"
    if "infra" in error.lower() or "connection" in error.lower():
        return "infrastructure"
    return str(row.get("phase") or "error")


def validation_summary(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("phase1b_validation") or {}
    judge = value.get("semanticJudge") or {}
    return {
        "passed": bool(value.get("passed")),
        "wholeFileLeanSuccess": value.get("wholeFileLeanSuccess"),
        "semanticErrorCount": len(value.get("semanticErrors") or []),
        "semanticWarningCount": len(value.get("semanticWarnings") or []),
        "phase2StructuralErrorCount": len(value.get("phase2StructuralErrors") or []),
        "phase2StandaloneErrorCount": len(value.get("phase2StandaloneErrors") or []),
        "pendingNodeCount": int(value.get("pendingNodeCount") or 0),
        "semanticJudgePassed": judge.get("passed"),
        "unfaithfulSteps": [
            item.get("step_id") for item in judge.get("steps", [])
            if not item.get("faithful")
        ],
        "semanticJudgeSummary": judge.get("summary", []),
    }


def edit_stats(row: dict[str, Any]) -> dict[str, Any]:
    history = list(row.get("phase1b_edit_history") or [])
    return {
        "roundCount": len(history),
        "acceptedEditCount": sum(len(item.get("accepted") or []) for item in history),
        "rejectedEditCount": sum(len(item.get("rejected") or []) for item in history),
        "identicalEditCount": sum(len(item.get("identical") or []) for item in history),
        "atomicRejectedRoundCount": sum(
            not bool(item.get("candidateApplied")) for item in history
        ),
        "acceptedEdits": [
            {"round": item.get("round"), **edit}
            for item in history for edit in (item.get("accepted") or [])
        ],
    }


def artifact_paths(row: dict[str, Any]) -> dict[str, str]:
    blueprint_dir = Path(str(row.get("blueprint_dir") or ""))
    candidates = [
        blueprint_dir / "phase1b_final.lean",
        blueprint_dir / "phase1_failed_last.json",
        blueprint_dir / "phase1a_canonical_attempt_2.lean",
        blueprint_dir / "phase1a_canonical_attempt_1.lean",
    ]
    selected = next((path for path in candidates if path.exists()), None)
    return {
        "blueprintDir": str(blueprint_dir),
        "representativeArtifact": str(selected) if selected else "",
    }


def compact_error(row: dict[str, Any], limit: int = 320) -> str:
    value = " ".join(str(row.get("error") or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def make_case(
    source_id: str,
    baseline_status: str,
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> dict[str, Any]:
    if new is None:
        return {
            "sourceId": source_id,
            "baselineStatus": baseline_status,
            "newStatus": "missing",
            "transition": "missing",
        }
    old_status = str((old or {}).get("status") or baseline_status)
    new_status = str(new.get("status"))
    if old_status == "error" and new_status == "phase1_accepted":
        transition = "recovered"
    elif old_status == "phase1_accepted" and new_status == "phase1_accepted":
        transition = "retained"
    elif old_status == "phase1_accepted" and new_status == "error":
        transition = "regressed"
    else:
        transition = "stillFailed"
    trace = trace_stats(new)
    edits = edit_stats(new)
    case = {
        "sourceId": source_id,
        "baselineStatus": old_status,
        "newStatus": new_status,
        "transition": transition,
        "failureStage": classify_failure(new),
        "error": compact_error(new),
        "manifest": manifest_check(new),
        "artifacts": artifact_paths(new),
        "phase1AAttempts": len(list(Path(str(new.get("blueprint_dir") or "")).glob("phase1a_attempt_*.lean"))),
        "edits": edits,
        "validation": validation_summary(new),
        "trace": trace,
        "nodeCount": int(new.get("total_nodes") or 0),
    }
    mechanisms = []
    if case["manifest"].get("schemaVersion") == 3:
        mechanisms.append("splitterV4")
    if trace["phase1ACanonicalPassed"]:
        mechanisms.append("phase1ACanonicalGate")
    if trace["searchThenEdit"]:
        mechanisms.append("mathlibSearchThenEdit")
    if trace["judgeCallCount"]:
        mechanisms.append("semanticJudge")
    case["observedMechanisms"] = mechanisms
    return case


def build_report(old_root: Path, new_root: Path, selection_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    old_rows = result_map(old_root)
    new_rows = result_map(new_root)
    cases = [
        make_case(
            record["source_id"], record["baseline_status"],
            old_rows.get(record["source_id"]), new_rows.get(record["source_id"]),
        )
        for record in selection["records"]
    ]
    transition_counts = Counter(case["transition"] for case in cases)
    status_counts = Counter(case["newStatus"] for case in cases)
    failure_counts = Counter(
        case.get("failureStage") for case in cases if case["newStatus"] == "error"
    )
    accepted = [case for case in cases if case["newStatus"] == "phase1_accepted"]
    accepted_contract_count = sum(
        case.get("validation", {}).get("passed") is True for case in accepted
    )
    old_trace_values = [
        trace_stats(old_rows[record["source_id"]])
        for record in selection["records"]
        if record["source_id"] in old_rows
    ]
    old_comparison = {
        "terminal": len(old_trace_values),
        "accepted": sum(
            old_rows[record["source_id"]].get("status") == "phase1_accepted"
            for record in selection["records"] if record["source_id"] in old_rows
        ),
        "llmRequestCount": sum(item["llmRequestCount"] for item in old_trace_values),
        "promptTokens": sum(item["promptTokens"] for item in old_trace_values),
        "completionTokens": sum(item["completionTokens"] for item in old_trace_values),
        "totalTokens": sum(item["totalTokens"] for item in old_trace_values),
        "summedCaseSeconds": sum(item["durationSeconds"] for item in old_trace_values),
    }
    totals = {
        "selected": len(cases),
        "terminal": sum(case["newStatus"] in {"phase1_accepted", "error"} for case in cases),
        "newStatusCounts": dict(status_counts),
        "transitionCounts": dict(transition_counts),
        "failureStageCounts": dict(failure_counts),
        "schemaV3Count": sum(case.get("manifest", {}).get("schemaVersion") == 3 for case in cases),
        "continuousCoverageCount": sum(case.get("manifest", {}).get("continuousCoverage") is True for case in cases),
        "excludedFieldCount": sum(case.get("manifest", {}).get("hasExcludedFields") is True for case in cases),
        "canonicalPassedCount": sum(case.get("trace", {}).get("phase1ACanonicalPassed") is True for case in cases),
        "searchCallCount": sum(case.get("trace", {}).get("searchCallCount", 0) for case in cases),
        "searchThenEditCaseCount": sum(case.get("trace", {}).get("searchThenEdit") is True for case in cases),
        "judgeAuditCount": sum(case.get("trace", {}).get("judgeAuditCount", 0) for case in cases),
        "judgeCallCount": sum(case.get("trace", {}).get("judgeCallCount", 0) for case in cases),
        "judgeParsedCandidateCount": sum(case.get("trace", {}).get("judgeParsedCandidateCount", 0) for case in cases),
        "judgeFailToPassCaseCount": sum(case.get("trace", {}).get("judgeFailToPass") is True for case in cases),
        "judgeLengthCount": sum(case.get("trace", {}).get("judgeLengthCount", 0) for case in cases),
        "llmLengthFinishCount": sum(case.get("trace", {}).get("lengthFinishCount", 0) for case in cases),
        "llmPromptTokens": sum(case.get("trace", {}).get("promptTokens", 0) for case in cases),
        "llmCompletionTokens": sum(case.get("trace", {}).get("completionTokens", 0) for case in cases),
        "llmTotalTokens": sum(case.get("trace", {}).get("totalTokens", 0) for case in cases),
        "infraErrorCount": sum(case.get("failureStage") == "infrastructure" for case in cases),
        "acceptedContractCount": accepted_contract_count,
        "llmSpanStartCount": sum(case.get("trace", {}).get("llmStartCount", 0) for case in cases),
        "llmUnpairedSpanCount": sum(case.get("trace", {}).get("llmUnpairedSpanCount", 0) for case in cases),
        "toolSpanStartCount": sum(case.get("trace", {}).get("toolStartCount", 0) for case in cases),
        "toolUnpairedSpanCount": sum(case.get("trace", {}).get("toolUnpairedSpanCount", 0) for case in cases),
    }
    return {
        "schemaVersion": 1,
        "oldExperiment": str(old_root),
        "newExperiment": str(new_root),
        "selectionManifest": str(selection_path),
        "oldComparison": old_comparison,
        "totals": totals,
        "cases": cases,
    }


def markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    old = report["oldComparison"]
    token_delta = (
        (totals["llmTotalTokens"] / old["totalTokens"] - 1.0) * 100.0
        if old["totalTokens"] else 0.0
    )
    lines = [
        "# Phase 1B v5 smoke27 对比报告",
        "",
        "## 总览",
        "",
        f"- 终态：{totals['terminal']}/{totals['selected']}。",
        f"- 新结果：{json.dumps(totals['newStatusCounts'], ensure_ascii=False)}。",
        f"- 状态迁移：{json.dumps(totals['transitionCounts'], ensure_ascii=False)}。",
        f"- 失败阶段：{json.dumps(totals['failureStageCounts'], ensure_ascii=False)}。",
        f"- manifest schema v3 / 连续覆盖：{totals['schemaV3Count']}/{totals['continuousCoverageCount']}；含旧 excluded 字段：{totals['excludedFieldCount']}。",
        f"- Mathlib search：{totals['searchCallCount']} 次，search→edit 样例 {totals['searchThenEditCaseCount']} 条。",
        f"- 语义 Judge：候选审计 {totals['judgeAuditCount']} 次、真实请求 {totals['judgeCallCount']} 次、成功解析 {totals['judgeParsedCandidateCount']} 次，FAIL→PASS 样例 {totals['judgeFailToPassCaseCount']} 条，截断 {totals['judgeLengthCount']} 次。",
        f"- LLM token：prompt={totals['llmPromptTokens']}，completion={totals['llmCompletionTokens']}，total={totals['llmTotalTokens']}；finish_reason=length {totals['llmLengthFinishCount']} 次。",
        f"- accepted 六项终验：{totals['acceptedContractCount']}/{totals['newStatusCounts'].get('phase1_accepted', 0)}；LLM span 未配对 {totals['llmUnpairedSpanCount']}/{totals['llmSpanStartCount']}，tool span 未配对 {totals['toolUnpairedSpanCount']}/{totals['toolSpanStartCount']}。",
        f"- 同一 27 条旧基线：accepted={old['accepted']}、LLM requests={old['llmRequestCount']}、tokens={old['totalTokens']}；新流程 requests={totals['llmSpanStartCount']}、tokens={totals['llmTotalTokens']}（{token_delta:+.1f}%）。",
        "",
        "## 27 条逐条结果",
        "",
        "| source_id | old→new | stage | A attempts | B rounds/edits | search | Judge | artifact |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for case in report["cases"]:
        trace = case.get("trace", {})
        edits = case.get("edits", {})
        artifact = case.get("artifacts", {}).get("representativeArtifact", "")
        artifact_cell = f"`{artifact}`" if artifact else "—"
        verdicts = trace.get("judgeVerdicts", [])
        judge = "→".join("PASS" if value else "FAIL" for value in verdicts) or "—"
        lines.append(
            f"| `{case['sourceId']}` | {case['baselineStatus']}→{case['newStatus']} "
            f"({case['transition']}) | {case.get('failureStage', '—')} | "
            f"{case.get('phase1AAttempts', 0)} | {edits.get('roundCount', 0)}/"
            f"{edits.get('acceptedEditCount', 0)} | {trace.get('searchCallCount', 0)} | "
            f"{judge} | {artifact_cell} |"
        )
    lines.extend(["", "## 失败明细", ""])
    failures = [case for case in report["cases"] if case["newStatus"] == "error"]
    if not failures:
        lines.append("无。")
    for case in failures:
        validation = case.get("validation", {})
        judge_reasons = validation.get("semanticJudgeSummary", [])
        lines.extend([
            f"### `{case['sourceId']}`",
            "",
            f"- 阶段：{case['failureStage']}；迁移：{case['transition']}。",
            f"- 最终错误：{case['error']}",
            f"- 观察到的机制：{', '.join(case.get('observedMechanisms', [])) or '无'}。",
            f"- 最终门：Lean={validation.get('wholeFileLeanSuccess')}，semanticErrors={validation.get('semanticErrorCount')}，phase2Structural={validation.get('phase2StructuralErrorCount')}，standalone={validation.get('phase2StandaloneErrorCount')}，Pending={validation.get('pendingNodeCount')}，Judge={validation.get('semanticJudgePassed')}。",
            *( [f"- Judge 首个剩余原因：{judge_reasons[0].get('step_id')} — {judge_reasons[0].get('reason')}"] if judge_reasons else [] ),
            "",
        ])
    lines.extend(["## Judge FAIL→编辑→PASS", ""])
    repaired = [case for case in report["cases"] if case.get("trace", {}).get("judgeFailToPass")]
    if not repaired:
        lines.append("真实运行未观察到 Judge FAIL→PASS；详见结论，不能把无观测事件包装为收益。")
    for case in repaired:
        lines.append(f"- `{case['sourceId']}`：{case['trace']['judgeVerdicts']}")
    lines.extend(["", "## Search→edit", ""])
    searched = [case for case in report["cases"] if case.get("trace", {}).get("searchThenEdit")]
    if not searched:
        lines.append("真实运行未触发 search→edit。")
    for case in searched:
        first = case["trace"]["searchQueries"][0]
        lines.append(
            f"- `{case['sourceId']}`：round {first['round']}，targets="
            f"{first['targetNodeNames']}，query={first['query']}"
        )
    lines.extend([
        "",
        "## 关键样例的因果分析",
        "",
        "### `brumo_2025/12`：Phase 1A 可重建门确实解决了旧接口不一致",
        "",
        "旧 skeleton 把普通顶层 `a_1`…`a_10` 放入 `compute_a_10` 的 `sorry_using`；它们可被 Lean 引用，却不是 Blueprint node，因此旧 Phase 1B 的 16 轮候选全部无法原子重建。v5 不再允许这种 skeleton 进入 B。新结果仅保留 7 个 Blueprint node，5 轮分别替换 `problem_setup`、`recurrence_holds`、`compute_values`、`a_10_equals_101` 和 root，最终六项门通过。这里的恢复可以归因于 Phase 1A canonical/dependency closure，而不是生成随机性。",
        "",
        "但该 accepted 仍暴露 Judge 假阳性：COT 的 `-(-1)^n` 在新 `recurrence_relation` 中被写成减去 `natAbs`，奇数项的符号语义反了；Judge 的翻译明确读出了 absolute value，却仍判 faithful。故 accepted 只能解释为‘通过当前自动门’，不能等同于人工确认的完全忠实。",
        "",
        "### `MATH-500/prealgebra/378`：观察到 Judge FAIL→edit→PASS",
        "",
        "Judge 序列为 FAIL→FAIL→FAIL→PASS。它先发现 `total_shaded_area` 只重复 `second_triangle_area = 6`，没有表达总阴影面积；round 2 将其改为 `first_triangle_area + second_triangle_area - first_triangle_area = 6`。之后 Judge 又定位 `first_triangle_contained_in_second` 的不等式方向，round 3 的 `y ≥ second_triangle_line x` 在 round 4 改为 `y ≤ second_triangle_line x` 后才通过。这是语义反馈直接改变具体 Lean proposition 的清晰正例。",
        "",
        "Judge 在该样例的早期理由中也出现长篇自我反驳，说明二值结论可用于门控，但自由文本 reason 仍有噪声；生产时应进一步约束 reason 长度和只描述 source/formal mismatch。",
        "",
        "### 最终答案复述：`aime_2024/81` 与 `brumo_2025/30`",
        "",
        "schema v3 没有排除末尾 Step：`aime_2024/81` 的 S005 同时包含唯一矩形说明和 `boxed{15}`，root 映射 S005；`brumo_2025/30` 的纯 `boxed{sqrt(143)}` 成为普通 S010，root 也映射 S010。两条旧 accepted 均保持通过，说明删除 excluded/duplicateRoot gate 没有造成这两个回归。",
        "",
        "不过 `aime_2024/81` 的 root 只断言‘存在一个 card=15 的 rectangles Finset’，没有断言它枚举且仅枚举正十二边形内的全部矩形；这是无约束存在式的弱化，而 Judge 判了 PASS。它与 `brumo_2025/12` 一起证明当前 Judge 对对象完备性、exact-count 与存在式弱化仍不够敏感。",
        "",
        "### Mathlib search 的实际作用",
        "",
        f"27 条共发出 {totals['searchCallCount']} 次 search，25 条观察到同轮 search→edit；工具接线与预算控制是有效的。但大量查询返回 `No Mathlib results` 或通用 lemma，且多数被改的是问题专用 statement/Pending，而非缺失的 Mathlib 名称。当前数据只能证明 search 被使用，不能证明 14 条恢复由 search 导致。最明确的 `brumo_2025/12` 恢复来自 canonical gate；search 更像伴随动作。",
        "",
        "## 总体判断",
        "",
        "v5 在工程准入和可修复性上显著改进：22 条旧失败恢复 14 条，5 条旧 accepted 全部保持，且没有 infra error；canonical skeleton 避免了 BRUMO/12 类不可编辑死锁，Judge 确实产生 8 条 FAIL→PASS 修复链。",
        "",
        f"代价也很明确：同一 27 条 trace token 从 {old['totalTokens']} 增至 {totals['llmTotalTokens']}（{token_delta:+.1f}%），主要来自逐轮整 Blueprint 上下文、233 次 search 前后请求和 Judge。并且两个 accepted 的人工复核已找到 Judge 假阳性。因此本轮证明了 pipeline 更可修、自动通过率更高，但尚未证明 19 条都达到严格的人类级语义忠实。下一步最值得做的是强化 Judge 对‘符号方向/绝对值替换’、‘exact count→无约束存在’和‘把派生结论变成前提’三类弱化，而不是继续增加 search 次数。",
    ])
    lines.extend(["", "## 每条简洁 case note", ""])
    for case in report["cases"]:
        if case["transition"] == "recovered":
            note = "旧失败恢复；"
        elif case["transition"] == "retained":
            note = "旧通过保持；"
        elif case["transition"] == "regressed":
            note = "发生回归；"
        else:
            note = "仍失败；"
        note += (
            f"A canonical={case.get('trace', {}).get('phase1ACanonicalPassed')}, "
            f"search={case.get('trace', {}).get('searchCallCount', 0)}, "
            f"Judge={case.get('trace', {}).get('judgeVerdicts', [])}, "
            f"final={case.get('failureStage')}."
        )
        lines.append(f"- `{case['sourceId']}`：{note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args()
    report = build_report(args.old, args.new, args.selection)
    output_dir = args.new / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase1b_v5_smoke27_analysis.json"
    markdown_path = output_dir / "phase1b_v5_smoke27_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
