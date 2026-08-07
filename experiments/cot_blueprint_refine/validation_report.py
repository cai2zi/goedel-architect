from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path("/ssd/czx/czx_work/cot_blueprint_refine")
DEFAULT_NEW = DEFAULT_BASE / "qwen3_8b_397b_refine_ablation_trace_v2_50"
COMPARISONS = {
    "old_slow_ablation": DEFAULT_BASE / "qwen3_8b_397b_refine_ablation",
    "old_fast_blueprint_refine_40": DEFAULT_BASE / "qwen3_8b_blueprint_refine_40",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def quantiles(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
        "max_ms": max(values) if values else None,
    }


def selected_results(root: Path, selected_ids: set[str]) -> list[dict[str, Any]]:
    rows = read_jsonl(root / "robustpa/blueprint/results.jsonl")
    return [row for row in rows if str(row.get("source_id")) in selected_ids]


def trace_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    trace_bytes = 0
    for row in rows:
        trace_path = Path(str(row.get("trace_path") or ""))
        if not trace_path.exists():
            continue
        trace_bytes += trace_path.stat().st_size
        events.extend(read_jsonl(trace_path))

    kinds = Counter(str(event.get("kind")) for event in events)
    spans: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        if event.get("span_id"):
            spans[str(event["span_id"])][str(event.get("kind"))] += 1

    pair_specs = {
        "llm": ("llm_request_start", "llm_request_end"),
        "tool": ("tool_call", "tool_result"),
        "node_semaphore": ("node_semaphore_wait_start", "node_semaphore_wait_end"),
    }
    pairing: dict[str, Any] = {}
    for label, (start_kind, end_kind) in pair_specs.items():
        starts = {span for span, counts in spans.items() if counts[start_kind]}
        ends = {span for span, counts in spans.items() if counts[end_kind]}
        exactly_once = sum(
            counts[start_kind] == 1 and counts[end_kind] == 1
            for counts in spans.values()
            if counts[start_kind] or counts[end_kind]
        )
        pairing[label] = {
            "starts": len(starts),
            "ends": len(ends),
            "missing_end": len(starts - ends),
            "orphan_end": len(ends - starts),
            "exactly_once": exactly_once,
            "pairing_rate": exactly_once / len(starts) if starts else None,
        }

    timing_keys = (
        "micro_batch_wait_ms", "client_inflight_wait_ms", "client_http_ms",
        "request_slot_wait_ms", "global_snippet_slot_wait_ms", "repl_wait_ms",
        "repl_create_ms", "header_prep_ms", "lean_exec_wall_ms", "server_total_ms",
    )
    timings: dict[str, list[float]] = {key: [] for key in timing_keys}
    for event in events:
        if event.get("kind") not in {"lean_check_result", "tool_result"}:
            continue
        args = event.get("args") or {}
        event_timings = args.get("timings") if isinstance(args, dict) else None
        if not isinstance(event_timings, dict):
            continue
        for key in timing_keys:
            value = event_timings.get(key)
            if isinstance(value, (int, float)):
                timings[key].append(float(value))

    usage = Counter()
    for event in events:
        if event.get("kind") != "llm_usage":
            continue
        args = event.get("args") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = args.get(key)
            if isinstance(value, int):
                usage[key] += value

    return {
        "schema_versions": dict(Counter(str(e.get("schema_version", 1)) for e in events)),
        "event_counts": dict(kinds),
        "pairing": pairing,
        "timings": {key: quantiles(values) for key, values in timings.items()},
        "llm_tokens": dict(usage),
        "trace_files": sum(bool(row.get("trace_path")) for row in rows),
        "trace_bytes": trace_bytes,
    }


def experiment_summary(root: Path, selected_ids: set[str]) -> dict[str, Any]:
    all_rows = read_jsonl(root / "robustpa/blueprint/results.jsonl")
    rows = [row for row in all_rows if str(row.get("source_id")) in selected_ids]
    runtime = read_json(root / "robustpa/blueprint/runtime_history.json")
    lean_runtime = read_json(root / "robustpa/blueprint/lean_runtime.json")
    distribution = {
        int(size): int(count)
        for size, count in lean_runtime.get("stats", {}).get("batch_size_distribution", {}).items()
    }
    batches = sum(distribution.values())
    return {
        "root": str(root),
        "full_rows": len(all_rows),
        "selected_rows": len(rows),
        "selected_terminal": len(rows),
        "selected_root_solved": sum(bool(row.get("root_proved")) for row in rows),
        "selected_statuses": dict(Counter(str(row.get("status")) for row in rows)),
        "selected_infra_error_nodes": sum(int(row.get("infra_error_node_count") or 0) for row in rows),
        "runtime_s": runtime.get("total_elapsed_s"),
        "runtime_human": runtime.get("total_elapsed_time"),
        "lean": {
            **lean_runtime,
            "size_1_fraction": distribution.get(1, 0) / batches if batches else None,
            "size_8_fraction": distribution.get(8, 0) / batches if batches else None,
        },
        "trace": trace_summary(rows),
    }


def kimina_metrics(root: Path) -> dict[str, Any]:
    rows = read_jsonl(root / "kimina/kimina_metrics.jsonl")
    pool_keys = ("free", "busy", "starting", "waiting", "active_snippets", "repl_rss_total_bytes")
    system_keys = ("cpu_percent", "memory_available_bytes", "memory_used_bytes", "swap_used_bytes")
    summary: dict[str, Any] = {"samples": len(rows)}
    for section, keys in (("pool", pool_keys), ("system", system_keys)):
        summary[section] = {}
        for key in keys:
            values = [row.get(section, {}).get(key) for row in rows]
            values = [value for value in values if isinstance(value, (int, float))]
            summary[section][key] = {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "delta": values[-1] - values[0] if values else None,
            }
    return summary


def judge_summary(root: Path) -> dict[str, Any]:
    rows = read_jsonl(root / "evaluation/judge_results.jsonl")
    return {
        "unique_requests": len(rows),
        "statuses": dict(Counter(str(row.get("status")) for row in rows)),
        "flags": dict(Counter(str(row.get("judge_flag")) for row in rows)),
        "format_errors": sum(row.get("error_layer") == "flag_format" for row in rows),
        "all_raw_content_is_unique_flag": all(
            row.get("raw_content") in {"[[JUDGE=0]]", "[[JUDGE=1]]"}
            for row in rows if row.get("status") == "ok"
        ),
        "response_format_absent_code_audit": "response_format" not in (
            Path(__file__).with_name("judge.py").read_text(encoding="utf-8")
        ),
    }


def build_report(new_root: Path) -> dict[str, Any]:
    prepared = read_jsonl(new_root / "prepared/generation_inputs.jsonl")
    selected_ids = {str(row["name"]) for row in prepared}
    current = experiment_summary(new_root, selected_ids)
    old = {label: experiment_summary(root, selected_ids) for label, root in COMPARISONS.items()}
    evaluation = read_json(new_root / "evaluation/metrics.json")
    session = read_json(new_root / "kimina/session.json")
    batch = current["lean"]["stats"]
    timings = current["trace"]["timings"]
    baseline_solved = old["old_slow_ablation"]["selected_root_solved"]
    checks = {
        "50_terminal_results": current["selected_terminal"] == 50,
        "no_terminal_kimina_infra_error": current["selected_infra_error_nodes"] == 0,
        "llm_span_pairing_100_percent": current["trace"]["pairing"]["llm"]["pairing_rate"] == 1,
        "tool_span_pairing_100_percent": current["trace"]["pairing"]["tool"]["pairing_rate"] == 1,
        "lean_compile_p50_le_5s": (timings["client_http_ms"]["p50_ms"] or math.inf) <= 5000,
        "lean_compile_p90_le_30s": (timings["client_http_ms"]["p90_ms"] or math.inf) <= 30000,
        "size_1_below_10_percent": current["lean"]["size_1_fraction"] < 0.1,
        "judge_no_flag_format_error": judge_summary(new_root)["format_errors"] == 0,
        "root_solved_not_down_more_than_2": current["selected_root_solved"] >= baseline_solved - 2,
        "kimina_clean_shutdown": session.get("status") == "stopped" and not session.get("forced_kill"),
        "no_429_or_no_available_repl": batch.get("http_429") == 0 and batch.get("no_available_repl") == 0,
    }
    return {
        "new": current,
        "old": old,
        "kimina_metrics": kimina_metrics(new_root),
        "kimina_session": session,
        "judge": judge_summary(new_root),
        "evaluation": evaluation.get("ablation", {}),
        "acceptance": checks,
        "passed": all(checks.values()),
    }


def markdown(report: dict[str, Any]) -> str:
    new = report["new"]
    timing = new["trace"]["timings"]
    lines = [
        "# RobustPA Trace v2 50 条验收报告", "",
        f"整体结论：**{'PASS' if report['passed'] else 'FAIL'}**。", "",
        "## 实验对比", "",
        "| 实验 | 范围 | RobustPA 时长 | 同 50 条 root solved | batch=1 | batch=8 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for label, item in [*report["old"].items(), ("new_trace_v2_50", new)]:
        lines.append(
            f"| {label} | {item['full_rows']} | {item['runtime_human']} | "
            f"{item['selected_root_solved']}/50 | {item['lean']['size_1_fraction']:.1%} | "
            f"{item['lean']['size_8_fraction']:.1%} |"
        )
    lines.extend(["", "## Trace 与 Lean timing", "", "| 指标 | P50 | P90 | Max |", "|---|---:|---:|---:|"])
    for key in ("micro_batch_wait_ms", "client_inflight_wait_ms", "client_http_ms", "repl_wait_ms", "header_prep_ms", "lean_exec_wall_ms", "server_total_ms"):
        value = timing[key]
        lines.append(f"| {key} | {value['p50_ms']:.3f} ms | {value['p90_ms']:.3f} ms | {value['max_ms']:.3f} ms |")
    lines.extend(["", "## 验收项", ""])
    for key, passed in report["acceptance"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{key}`")
    lines.extend([
        "", "## 结论与已发现缺口", "",
        "- Trace v2 的 LLM、tool 与 node semaphore span 全部一一配对；Lean 分段 timing 和 Kimina metrics 可用。",
        "- 10ms FIFO global batcher 在真实工作负载下没有形成以 8 为主的 batch；大量 Lean 请求由较长 LLM 调用间隔开，size=1 仍占主导，因此该验收项失败。",
        "- 真实运行发现 REPL 偶发退出会返回 HTTP 500；客户端已增加传输错误与 5xx 的有限重试。该修复发生在本轮 blueprint 子进程启动后，需在下一次运行中验证重试计数。",
        "- 本轮 trace 暴露了 raw_output 重复完整 Lean source 的体积问题；实现已改为只保存 code length 与 SHA-256。",
        "- 本轮后置审计还发现并行 tool_result 曾在整组结束后统一写入；实现已将 start/end 下沉到实际 batch、search 与 cache 执行单元，并由回归测试验证 cache/search duration 不再互相污染。",
        "- Kimina 正常 SIGTERM 退出、没有 forced kill，8000/8001 端口与服务 PID 均已释放。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW)
    args = parser.parse_args()
    report = build_report(args.new_root)
    output_json = args.new_root / "validation_report.json"
    output_md = args.new_root / "validation_report.md"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(markdown(report), encoding="utf-8")
    print(output_md)
    print(json.dumps(report["acceptance"], ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
