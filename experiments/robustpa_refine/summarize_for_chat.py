from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXP_DIR = Path("/ssd/czx/czx_work/robustpa_refine/qwen3_5_397b_MiniF2F_orig_refine")


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _csv_block(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "```csv\n\n```"
    out = []
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in columns})
    out.append(buf.getvalue().strip())
    return "```csv\n" + "\n".join(out) + "\n```"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, round(p * (len(arr) - 1))))
    return arr[idx]


def _compact_ids(ids: list[str], max_ids: int = 0) -> str:
    if not ids:
        return "-"
    if max_ids and len(ids) > max_ids:
        shown = ", ".join(ids[:max_ids])
        return f"{shown}, ... (+{len(ids) - max_ids})"
    return ", ".join(ids)


@dataclass
class TraceSummary:
    source_id: str
    record_id: str
    unique_id: str
    status: str = ""
    root_proved: bool = False
    iterations: int | None = None
    total_nodes: int = 0
    proved_node_count: int = 0
    failed_node_count: int = 0
    duration_s: float = 0.0
    kind_counts: collections.Counter[str] = field(default_factory=collections.Counter)
    tool_counts: collections.Counter[str] = field(default_factory=collections.Counter)
    node_tool_counts: collections.Counter[str] = field(default_factory=collections.Counter)
    node_lean_counts: collections.Counter[str] = field(default_factory=collections.Counter)
    node_mathlib_counts: collections.Counter[str] = field(default_factory=collections.Counter)
    node_token_counts: collections.Counter[str] = field(default_factory=collections.Counter)
    turn_tool_counts: collections.Counter[tuple[str, int]] = field(default_factory=collections.Counter)
    turn_tool_breakdown: dict[tuple[str, int], collections.Counter[str]] = field(default_factory=dict)
    llm_usage: int = 0
    llm_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    phase_llm_calls: collections.Counter[str] = field(default_factory=collections.Counter)
    final_verify_ok: int = 0
    final_verify_false: int = 0
    final_verify_wall_sum: float = 0.0
    compile_ok_results: int = 0
    compile_fail_results: int = 0
    mathlib_queries: list[str] = field(default_factory=list)
    search_empty: int = 0
    context_overflow_errors: int = 0
    unavailable_tool_calls: collections.Counter[str] = field(default_factory=collections.Counter)
    llm_errors: list[str] = field(default_factory=list)
    sample_errors: list[str] = field(default_factory=list)

    @property
    def tool_calls(self) -> int:
        return sum(self.tool_counts.values())

    @property
    def lean_calls(self) -> int:
        return self.tool_counts.get("lean_compile", 0)

    @property
    def mathlib_calls(self) -> int:
        return self.tool_counts.get("mathlib_search", 0)

    @property
    def unique_mathlib_queries(self) -> int:
        return len({q.lower().strip() for q in self.mathlib_queries if q.strip()})

    @property
    def repeated_mathlib_queries(self) -> int:
        return len([q for q in self.mathlib_queries if q.strip()]) - self.unique_mathlib_queries

    @property
    def top_node_by_tools(self) -> tuple[str, int]:
        return self.node_tool_counts.most_common(1)[0] if self.node_tool_counts else ("", 0)

    @property
    def top_node_by_mathlib(self) -> tuple[str, int]:
        return self.node_mathlib_counts.most_common(1)[0] if self.node_mathlib_counts else ("", 0)

    @property
    def top_node_by_tokens(self) -> tuple[str, int]:
        return self.node_token_counts.most_common(1)[0] if self.node_token_counts else ("", 0)

    @property
    def max_tool_burst(self) -> tuple[str, int, int, dict[str, int]]:
        if not self.turn_tool_counts:
            return ("", 0, 0, {})
        (node, turn), count = self.turn_tool_counts.most_common(1)[0]
        return (node, turn, count, dict(self.turn_tool_breakdown.get((node, turn), {})))


def _trace_path_to_record_id(path: Path) -> str:
    return path.stem


def summarize_traces(exp_dir: Path, results_by_record: dict[str, dict[str, Any]], max_error_chars: int) -> dict[str, TraceSummary]:
    traces_dir = exp_dir / "traces"
    summaries: dict[str, TraceSummary] = {}
    if not traces_dir.exists():
        return summaries

    for path in sorted(traces_dir.rglob("*.jsonl")):
        record_id = _trace_path_to_record_id(path)
        result_row = results_by_record.get(record_id, {})
        unique_id = str(result_row.get("id") or record_id)
        summary = TraceSummary(
            source_id=str(result_row.get("source_id") or record_id),
            record_id=record_id,
            unique_id=unique_id,
            status=str(result_row.get("status") or ""),
            root_proved=bool(result_row.get("root_proved")),
            iterations=result_row.get("iterations"),
            total_nodes=int(result_row.get("total_nodes") or 0),
            proved_node_count=int(result_row.get("proved_node_count") or 0),
            failed_node_count=len(result_row.get("failed_nodes") or []),
        )
        first_ts: float | None = None
        last_ts: float | None = None

        for event in _jsonl(path):
            kind = str(event.get("kind") or "")
            tool = str(event.get("tool_name") or "")
            args = event.get("args") or {}
            result = str(event.get("result") or "")
            thm_name = str(event.get("thm_name") or "")
            turn = int(event.get("turn") or 0)
            ts = event.get("ts")

            summary.kind_counts[kind] += 1
            if isinstance(ts, (int, float)):
                first_ts = ts if first_ts is None else min(first_ts, float(ts))
                last_ts = ts if last_ts is None else max(last_ts, float(ts))

            if kind == "tool_call":
                summary.tool_counts[tool] += 1
                summary.node_tool_counts[thm_name] += 1
                summary.turn_tool_counts[(thm_name, turn)] += 1
                summary.turn_tool_breakdown.setdefault((thm_name, turn), collections.Counter())[tool] += 1
                if tool == "lean_compile":
                    summary.node_lean_counts[thm_name] += 1
                elif tool == "mathlib_search":
                    summary.node_mathlib_counts[thm_name] += 1
                    summary.mathlib_queries.append(str(args.get("query") or ""))
                elif tool and tool not in {"repo_search"}:
                    summary.unavailable_tool_calls[tool] += 1

            elif kind == "tool_result":
                if tool == "lean_compile":
                    if event.get("ok"):
                        summary.compile_ok_results += 1
                    else:
                        summary.compile_fail_results += 1
                    for error in args.get("errors") or []:
                        if len(summary.sample_errors) < 12:
                            summary.sample_errors.append(_norm(error)[:max_error_chars])
                elif tool == "mathlib_search":
                    text = result.strip()
                    if text in {"", "[]", "No results found."}:
                        summary.search_empty += 1

            elif kind == "llm_usage":
                summary.llm_usage += 1
                total = int(args.get("total_tokens") or 0)
                summary.llm_tokens += total
                summary.prompt_tokens += int(args.get("prompt_tokens") or 0)
                summary.completion_tokens += int(args.get("completion_tokens") or 0)
                phase = str(args.get("phase") or "")
                if phase:
                    summary.phase_llm_calls[phase] += 1
                summary.node_token_counts[thm_name] += total

            elif kind == "final_verify":
                if event.get("ok"):
                    summary.final_verify_ok += 1
                else:
                    summary.final_verify_false += 1
                summary.final_verify_wall_sum += float(args.get("wall_time_s") or 0.0)

            elif kind == "lean_check_result":
                for error in args.get("errors") or []:
                    if len(summary.sample_errors) < 12:
                        summary.sample_errors.append(_norm(error)[:max_error_chars])

            elif kind == "llm_error":
                message = _norm(args.get("message") or result)
                if "maximum context length" in message:
                    summary.context_overflow_errors += 1
                if len(summary.llm_errors) < 5:
                    summary.llm_errors.append(message[:max_error_chars])

        if first_ts is not None and last_ts is not None:
            summary.duration_s = last_ts - first_ts
        summaries[unique_id] = summary
    return summaries


def latest_rounds(exp_dir: Path) -> tuple[dict[str, dict[str, Any]], collections.Counter[str]]:
    latest: dict[str, dict[str, Any]] = {}
    counts: collections.Counter[str] = collections.Counter()
    for row in _jsonl(exp_dir / "rounds.jsonl"):
        row_id = str(row.get("id") or "")
        if not row_id:
            continue
        latest[row_id] = row
        counts[row_id] += 1
    return latest, counts


def failed_nodes(latest_round_by_id: dict[str, dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    latest = latest_round_by_id.get(str(row.get("id") or ""))
    if not latest:
        return []
    return [node for node in latest.get("nodes") or [] if node.get("signal") != "solved"]


def failure_text(latest_round_by_id: dict[str, dict[str, Any]], row: dict[str, Any]) -> str:
    parts = [str(row.get("error") or "")]
    for node in failed_nodes(latest_round_by_id, row):
        parts.extend([
            str(node.get("name") or ""),
            str(node.get("signal") or ""),
            str(node.get("analysis") or ""),
            str(node.get("suggested_fix") or ""),
        ])
        parts.extend(str(error) for error in (node.get("lean_errors") or []))
    return "\n".join(parts)


def proof_failure_category(
    row: dict[str, Any],
    latest_round_by_id: dict[str, dict[str, Any]],
    trace_summary: TraceSummary | None,
) -> str:
    error = str(row.get("error") or "")
    if row.get("status") == "error" and row.get("phase") == "phase1":
        return "P1_blueprint_generation_invalid"
    if row.get("status") == "error" and row.get("phase") == "phase3":
        return "P2_refinement_generated_invalid_blueprint"
    if "phase2-ready" in error:
        return "P3_refinement_contract_violation"
    if (trace_summary and trace_summary.context_overflow_errors) or int(row.get("infra_error_node_count") or 0):
        return "P4_infra_context_or_timeout"

    text = failure_text(latest_round_by_id, row)
    low = text.lower()
    if any(pattern.lower() in low for pattern in [
        "unknown identifier `h2`",
        "unknown identifier `h3`",
        "unknown identifier `d`",
        "unknown identifier `f`",
        "unknown identifier `parity`",
        "unknown identifier `is_arithmetic_sequence`",
        "function expected at\n  f",
        "function expected at\n  d",
        "not in scope",
        "unknown identifiers",
    ]):
        return "P5_missing_context_or_blueprint_symbol_mismatch"
    if any(pattern.lower() in low for pattern in [
        "noncomputable",
        "invalid `⟨...⟩` notation",
        "failed to synthesize instance",
        "type expected, got",
        "expected type `posrat`",
        "unknown identifier `π`",
        "unknown identifier `cos`",
        "unknown identifier `sin`",
        "unknown constant `complex.abs`",
        "unknown constant `real.csc`",
        "unknown constant `real.sec`",
    ]):
        return "P6_invalid_statement_type_or_lean_api"
    if any(pattern.lower() in low for pattern in [
        "unknown constant",
        "unknown identifier `nat.",
        "unknown identifier `fib_add_two`",
        "unknown identifier `div_",
        "unknown constant `finset.",
        "unknown constant `real.sqrt_eq_iff_sq_eq`",
    ]):
        return "P7_wrong_mathlib_name_or_missing_lemma"
    if any(pattern.lower() in low for pattern in [
        "rewrite` failed",
        "rfl` failed",
        "simp` made no progress",
        "linarith failed",
        "omega could not prove",
        "ring_nf",
        "nlinarith",
        "no goals to be solved",
        "unsolved goals",
        "type mismatch",
        "application type mismatch",
    ]):
        return "P8_tactic_algebra_search_stuck"
    if any(node.get("signal") in {"statement_wrong", "formally_negated"} for node in failed_nodes(latest_round_by_id, row)):
        return "P9_bad_or_false_sublemma"
    return "P10_other_proof_too_hard"


def proof_symptoms(latest_round_by_id: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    patterns = {
        "missing_context_symbol": [
            "Unknown identifier `h2`", "Unknown identifier `h3`", "Unknown identifier `D`",
            "Unknown identifier `f`", "Unknown identifier `parity`", "Unknown identifier `is_arithmetic_sequence`",
            "Function expected at\n  f", "Function expected at\n  D", "not in scope", "unknown identifiers",
        ],
        "invalid_stmt_type_or_api": [
            "noncomputable", "Invalid `⟨...⟩` notation", "failed to synthesize instance",
            "type expected, got", "expected type `PosRat`",
        ],
        "wrong_mathlib_name_or_missing": [
            "Unknown constant", "Unknown identifier `Nat.", "Unknown identifier `fib_add_two`",
            "Unknown identifier `div_", "Unknown constant `Finset.", "Unknown identifier `π`",
            "Unknown identifier `cos`", "Unknown identifier `sin`",
        ],
        "algebra_tactic_stuck": [
            "rewrite` failed", "rfl` failed", "simp` made no progress", "linarith failed",
            "omega could not prove", "ring_nf", "nlinarith", "No goals to be solved", "unsolved goals",
        ],
        "incomplete_or_sorry": ["Proof contains `sorry`", "unexpected end of input"],
        "dependency_cascade": ["Skipped without attempting a proof: dependency"],
        "bad_or_false_sublemma_signal": ["statement_wrong", "formally_negated", "formally refuted", "counterexample"],
        "forbidden_construct": ["forbidden construct `native_decide`"],
    }
    out: dict[str, list[str]] = {}
    for name, needles in patterns.items():
        ids: list[str] = []
        for row in rows:
            if row.get("root_proved"):
                continue
            text = failure_text(latest_round_by_id, row)
            if any(needle.lower() in text.lower() for needle in needles):
                ids.append(str(row.get("source_id") or row.get("record_id") or row.get("id")))
        out[name] = ids
    return out


def trace_rows(trace_summaries: dict[str, TraceSummary], round_counts: collections.Counter[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in trace_summaries.values():
        node, turn, burst_count, burst_tools = item.max_tool_burst
        top_node, top_node_count = item.top_node_by_tools
        rows.append({
            "source_id": item.source_id,
            "status": item.status,
            "ok": int(item.root_proved),
            "iter": item.iterations if item.iterations is not None else "",
            "dur_h": round(item.duration_s / 3600, 2),
            "rounds": round_counts.get(item.unique_id, 0),
            "nodes": item.total_nodes,
            "proved": item.proved_node_count,
            "failed": item.failed_node_count,
            "tools": item.tool_calls,
            "lean": item.lean_calls,
            "mathlib": item.mathlib_calls,
            "rep_q": item.repeated_mathlib_queries,
            "llm": item.llm_usage,
            "tok_m": round(item.llm_tokens / 1_000_000, 2),
            "top_node": top_node,
            "top_n": top_node_count,
            "burst_n": burst_count,
            "burst_node": node,
            "burst_turn": turn,
            "burst_tools": json.dumps(burst_tools, sort_keys=True),
            "ctx_err": item.context_overflow_errors,
        })
    return rows


def time_categories(trace_summaries: dict[str, TraceSummary]) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, float]]:
    items = list(trace_summaries.values())
    thresholds = {
        "duration_p75_s": _percentile([x.duration_s for x in items], 0.75),
        "tool_calls_p90": _percentile([float(x.tool_calls) for x in items], 0.90),
        "mathlib_calls_p90": _percentile([float(x.mathlib_calls) for x in items], 0.90),
        "llm_tokens_p90": _percentile([float(x.llm_tokens) for x in items], 0.90),
    }

    def is_slow(item: TraceSummary) -> bool:
        return (
            item.duration_s >= thresholds["duration_p75_s"]
            or item.tool_calls >= thresholds["tool_calls_p90"]
            or item.mathlib_calls >= thresholds["mathlib_calls_p90"]
            or item.llm_tokens >= thresholds["llm_tokens_p90"]
            or item.context_overflow_errors > 0
        )

    def primary(item: TraceSummary) -> str:
        if item.context_overflow_errors or item.unavailable_tool_calls or (item.status == "error" and (item.iterations or 0) and (item.iterations or 0) < 8):
            return "T1_framework_error_or_context_overflow"
        if item.mathlib_calls >= 100 and item.repeated_mathlib_queries >= 50:
            return "T2_repeated_mathlib_search_loop"
        if item.tool_calls >= 700 and item.lean_calls / max(1, item.tool_calls) >= 0.75:
            return "T3_repeated_lean_compile_loop"
        if item.llm_tokens >= thresholds["llm_tokens_p90"] or item.llm_usage >= 400:
            return "T4_context_history_token_bloat"
        if (item.iterations or 0) >= 8 and not item.root_proved:
            return "T5_max_iter_refinement_churn"
        if item.total_nodes >= 25 or item.failed_node_count >= 10:
            return "T6_large_blueprint_many_nodes"
        return "T7_other_slow"

    cats: dict[str, list[str]] = collections.defaultdict(list)
    for item in items:
        if is_slow(item):
            cats[primary(item)].append(item.source_id)

    symptoms = {
        "max_iter_not_solved": [x.source_id for x in items if (x.iterations or 0) >= 8 and not x.root_proved],
        "solved_only_after_iter_ge5": [x.source_id for x in items if x.root_proved and (x.iterations or 0) >= 5],
        "repeated_lean_compile_ge700": [x.source_id for x in items if x.lean_calls >= 700],
        "single_node_tool_loop_ge200": [x.source_id for x in items if x.top_node_by_tools[1] >= 200],
        "mathlib_calls_ge100": [x.source_id for x in items if x.mathlib_calls >= 100],
        "repeated_mathlib_queries_ge50": [x.source_id for x in items if x.repeated_mathlib_queries >= 50],
        "llm_tokens_ge3m": [x.source_id for x in items if x.llm_tokens >= 3_000_000],
        "large_blueprint_ge25_nodes": [x.source_id for x in items if x.total_nodes >= 25],
        "many_failed_nodes_ge10": [x.source_id for x in items if x.failed_node_count >= 10],
        "context_overflow": [x.source_id for x in items if x.context_overflow_errors > 0],
        "bad_unavailable_tool": [x.source_id for x in items if x.unavailable_tool_calls],
        "single_response_burst_gt8": [x.source_id for x in items if x.max_tool_burst[2] > 8],
        "single_response_burst_ge100": [x.source_id for x in items if x.max_tool_burst[2] >= 100],
    }
    return dict(cats), symptoms, thresholds


def top_rows(rows: list[dict[str, Any]], key: str, top_k: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key) or 0), reverse=True)[:top_k]


def build_report(args: argparse.Namespace) -> str:
    exp_dir = args.exp_dir.resolve()
    results = list(_jsonl(exp_dir / "results.jsonl"))
    results_by_id = {str(row.get("id")): row for row in results}
    results_by_record = {str(row.get("record_id")): row for row in results}
    metrics = _read_json(exp_dir / "metrics.json")
    lean_runtime = _read_json(exp_dir / "lean_runtime.json")
    latest_round_by_id, round_counts = latest_rounds(exp_dir)
    trace_summaries = summarize_traces(exp_dir, results_by_record, args.max_error_chars)
    rows = trace_rows(trace_summaries, round_counts)

    status_counts = collections.Counter(str(row.get("status") or "") for row in results)
    success_by_iter = collections.Counter(int(row.get("iterations") or 0) for row in results if row.get("root_proved"))
    failure_by_iter = collections.Counter(int(row.get("iterations") or 0) for row in results if not row.get("root_proved"))
    total = len(results)
    root_proved = sum(1 for row in results if row.get("root_proved"))

    proof_cats: dict[str, list[str]] = collections.defaultdict(list)
    for row in results:
        if row.get("root_proved"):
            continue
        trace_summary = trace_summaries.get(str(row.get("id")))
        proof_cats[proof_failure_category(row, latest_round_by_id, trace_summary)].append(str(row.get("source_id") or row.get("record_id")))

    time_cats, time_symptoms, thresholds = time_categories(trace_summaries)
    proof_symptom_hits = proof_symptoms(latest_round_by_id, results)

    all_kinds: collections.Counter[str] = collections.Counter()
    all_tools: collections.Counter[str] = collections.Counter()
    for summary in trace_summaries.values():
        all_kinds.update(summary.kind_counts)
        all_tools.update(summary.tool_counts)

    burst_values = [summary.max_tool_burst[2] for summary in trace_summaries.values()]
    burst_quantiles = {
        "p50": _percentile([float(x) for x in burst_values], 0.50),
        "p75": _percentile([float(x) for x in burst_values], 0.75),
        "p90": _percentile([float(x) for x in burst_values], 0.90),
        "p95": _percentile([float(x) for x in burst_values], 0.95),
        "p99": _percentile([float(x) for x in burst_values], 0.99),
        "max": max(burst_values) if burst_values else 0,
    }

    lines: list[str] = []
    lines.append("# RobustPA Refine Chat Bundle")
    lines.append("")
    lines.append("This is a compact, trace-aware experiment bundle for web-chat review. Raw trace JSONL is summarized, not pasted verbatim.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- exp_dir: `{exp_dir}`")
    lines.append("- inputs used: `metrics.json`, `lean_runtime.json`, `results.jsonl`, `rounds.jsonl`, `traces/**/*.jsonl`")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append(f"- total: {total}")
    lines.append(f"- root_proved: {root_proved} ({root_proved / total:.2%})" if total else "- root_proved: 0")
    lines.append(f"- status_counts: `{dict(status_counts)}`")
    if metrics:
        lines.append("- metrics_json_global:")
        lines.append("```json")
        lines.append(json.dumps(metrics.get("groups", [{}])[0], ensure_ascii=False, indent=2))
        lines.append("```")
    if lean_runtime:
        lines.append(f"- lean_runtime: `{json.dumps(lean_runtime, ensure_ascii=False, sort_keys=True)}`")
    lines.append(f"- trace_event_kinds: `{dict(all_kinds.most_common())}`")
    lines.append(f"- trace_tool_calls: `{dict(all_tools.most_common())}`")
    lines.append(f"- aggregate_llm_tokens: {sum(x.llm_tokens for x in trace_summaries.values())}")
    lines.append("")
    lines.append("## Root Success By Iter")
    iter_rows: list[dict[str, Any]] = []
    cumulative = 0
    for iteration in range(0, max([8, *success_by_iter.keys(), *failure_by_iter.keys()]) + 1):
        newly = success_by_iter.get(iteration, 0)
        cumulative += newly
        iter_rows.append({
            "iter": iteration,
            "new_root_proved": newly,
            "cumulative": cumulative,
            "cum_acc": f"{cumulative / total:.2%}" if total else "0.00%",
            "failed_ended": failure_by_iter.get(iteration, 0),
        })
    lines.append(_csv_block(iter_rows, ["iter", "new_root_proved", "cumulative", "cum_acc", "failed_ended"]))
    lines.append("")

    lines.append("## Time And Trace Diagnosis")
    lines.append(f"- slow_thresholds: `{json.dumps(thresholds, sort_keys=True)}`")
    lines.append(f"- tool_call_burst_quantiles: `{json.dumps(burst_quantiles, sort_keys=True)}`")
    lines.append(f"- single_response_burst_gt8_count: {len(time_symptoms.get('single_response_burst_gt8', []))}")
    lines.append(f"- single_response_burst_ge100_count: {len(time_symptoms.get('single_response_burst_ge100', []))}")
    lines.append("")
    lines.append("### Time Primary Categories")
    for name, ids in sorted(time_cats.items()):
        lines.append(f"- {name} ({len(ids)}): {_compact_ids(ids, args.max_ids_per_category)}")
    lines.append("")
    lines.append("### Time Symptoms Multilabel")
    for name, ids in sorted(time_symptoms.items()):
        lines.append(f"- {name} ({len(ids)}): {_compact_ids(ids, args.max_ids_per_category)}")
    lines.append("")

    lines.append("## Proof Failure Diagnosis")
    lines.append("### Primary Categories")
    for name, ids in sorted(proof_cats.items()):
        lines.append(f"- {name} ({len(ids)}): {_compact_ids(ids, args.max_ids_per_category)}")
    lines.append("")
    lines.append("### Symptoms Multilabel")
    for name, ids in sorted(proof_symptom_hits.items()):
        lines.append(f"- {name} ({len(ids)}): {_compact_ids(ids, args.max_ids_per_category)}")
    lines.append("")

    compact_cols = [
        "source_id", "status", "ok", "iter", "dur_h", "rounds", "nodes", "proved", "failed",
        "tools", "lean", "mathlib", "rep_q", "llm", "tok_m", "top_node", "top_n", "burst_n",
        "burst_node", "ctx_err",
    ]
    lines.append(f"## Top {args.top_k} By Duration")
    lines.append(_csv_block(top_rows(rows, "dur_h", args.top_k), compact_cols))
    lines.append("")
    lines.append(f"## Top {args.top_k} By Tool Calls")
    lines.append(_csv_block(top_rows(rows, "tools", args.top_k), compact_cols))
    lines.append("")
    lines.append(f"## Top {args.top_k} By Mathlib Calls")
    lines.append(_csv_block(top_rows(rows, "mathlib", args.top_k), compact_cols))
    lines.append("")
    lines.append(f"## Top {args.top_k} By LLM Tokens")
    lines.append(_csv_block(top_rows(rows, "tok_m", args.top_k), compact_cols))
    lines.append("")

    lines.append("## Failed Problem Details")
    failure_detail_rows: list[dict[str, Any]] = []
    for row in results:
        if row.get("root_proved"):
            continue
        row_id = str(row.get("id"))
        summary = trace_summaries.get(row_id)
        nodes = failed_nodes(latest_round_by_id, row)
        first_node = nodes[0] if nodes else {}
        first_errors = first_node.get("lean_errors") or []
        category = proof_failure_category(row, latest_round_by_id, summary)
        failure_detail_rows.append({
            "source_id": row.get("source_id"),
            "status": row.get("status"),
            "phase": row.get("phase"),
            "iter": row.get("iterations"),
            "category": category,
            "failed_nodes": len(nodes),
            "first_failed_node": first_node.get("name") or "",
            "first_signal": first_node.get("signal") or "",
            "error": _norm(row.get("error") or (first_errors[0] if first_errors else first_node.get("analysis") or ""))[:args.max_error_chars],
        })
    lines.append(_csv_block(
        failure_detail_rows,
        ["source_id", "status", "phase", "iter", "category", "failed_nodes", "first_failed_node", "first_signal", "error"],
    ))
    lines.append("")

    if args.problem_rows != "none":
        selected_rows = rows
        if args.problem_rows == "failures":
            selected_rows = [row for row in rows if not row.get("ok")]
        elif args.problem_rows == "top":
            keep: dict[str, dict[str, Any]] = {}
            for key in ("dur_h", "tools", "mathlib", "tok_m"):
                for row in top_rows(rows, key, args.top_k):
                    keep[str(row["source_id"])] = row
            selected_rows = list(keep.values())
        lines.append(f"## Per Problem Trace Summary ({args.problem_rows})")
        selected_rows = sorted(selected_rows, key=lambda row: str(row.get("source_id")))
        lines.append(_csv_block(selected_rows, compact_cols))
        lines.append("")

    lines.append("## Optimization Hints")
    lines.append("- Enforce `node_timeout_s` and `llm_api_timeout_s`; this run's config may override runner defaults with null.")
    lines.append("- Cap tool calls before executing an assistant message; one response can contain hundreds of calls and bypass `node_max_tool_calls=8`.")
    lines.append("- Disable or sharply limit parallel tool calls from the model if the serving stack supports it.")
    lines.append("- Cache and deduplicate identical `mathlib_search` queries per node and globally.")
    lines.append("- Trim prover message history; keep current goal, last few Lean errors, and accepted parent lemma declarations instead of full tool history.")
    lines.append("- Stop refinement early when blueprint hash, failed nodes, and first error class do not change across rounds.")
    lines.append("- Run Phase1/Phase3 contract checks before saving refined blueprints, especially the `sorry_using` placeholder contract.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one compact Markdown bundle from a RobustPA refine experiment for web-chat review.",
    )
    parser.add_argument("exp_dir", type=Path, nargs="?", default=DEFAULT_EXP_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Default: <exp_dir>/chat_bundle.md")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--problem-rows",
        choices=["all", "failures", "top", "none"],
        default="failures",
        help="Which per-problem trace rows to include. `all` is still compact CSV.",
    )
    parser.add_argument(
        "--max-ids-per-category",
        type=int,
        default=0,
        help="0 means list every ID; use e.g. 40 for a shorter chat bundle.",
    )
    parser.add_argument("--max-error-chars", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.max_error_chars <= 0:
        raise ValueError("--max-error-chars must be positive")
    exp_dir = args.exp_dir.resolve()
    if not (exp_dir / "results.jsonl").exists():
        raise FileNotFoundError(f"results.jsonl not found under {exp_dir}")
    output = args.output or (exp_dir / "chat_bundle.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    output.write_text(report, encoding="utf-8")
    print(f"[wrote] {output}")
    print(f"[chars] {len(report)}")


if __name__ == "__main__":
    main()
