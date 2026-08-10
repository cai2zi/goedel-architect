from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


WORK = Path("/ssd/czx/czx_work/cot_blueprint_refine")
RUNS = {
    "progressController": WORK / "qwen3_8b_397b_wrong10_step_v9_progress_controller",
    "planDirect": WORK / "qwen3_8b_397b_wrong10_step_v9_plan_direct",
    "directEdit": WORK / "qwen3_8b_397b_wrong10_step_v9_direct_edit",
}
OUT = WORK / "qwen3_8b_397b_wrong10_step_v9_comparison_report"
CONTROLS = {
    "MATH-500/test/counting_and_probability/765.json",
    "MATH-500/test/prealgebra/378.json",
    "MATH-500/test/intermediate_algebra/662.json",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def comparator_defect_count(result: dict[str, Any]) -> int:
    count = 0
    for step in result.get("steps") or ():
        for field in (
            "missing_clauses",
            "weakened_clauses",
            "unbound_objects",
            "wrong_relations",
        ):
            count += len(step.get(field) or ())
    root = result.get("root") or {}
    count += not bool(root.get("target_object_preserved"))
    count += not bool(root.get("answer_grounded"))
    count += sum(
        not bool(item.get("justified_side_branch"))
        for item in result.get("unreachable_steps") or ()
    )
    count += len(result.get("dependency_issues") or ())
    return count


def elapsed_seconds(run_root: Path) -> float | None:
    path = run_root / "kimina" / "session.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ready = data.get("ready_at") or data.get("started_at")
    stopped = data.get("stopped_at")
    if not ready or not stopped:
        return None
    return (datetime.fromisoformat(stopped) - datetime.fromisoformat(ready)).total_seconds()


def summarize_case(strategy: str, row: dict[str, Any]) -> dict[str, Any]:
    trace = read_jsonl(Path(row["trace_path"]))
    history = list(row.get("phase1b_edit_history") or ())
    usage: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    finish_length = 0
    for event in trace:
        if event.get("kind") != "llm_request_end" or not event.get("ok"):
            continue
        args = event.get("args") or {}
        phase = str(args.get("phase") or "unknown")
        usage[phase] += int(args.get("total_tokens") or 0)
        calls[phase] += 1
        finish_length += str(args.get("finish_reason") or "").lower() == "length"

    comparator_results = [
        (event.get("args") or {}).get("result") or {}
        for event in trace
        if event.get("kind") == "phase1BStrictCompareResult"
    ]
    decompiler_results = [
        (event.get("args") or {}).get("result") or {}
        for event in trace
        if event.get("kind") == "phase1BFormalDecompileResult"
    ]

    def vacuous_count(result: dict[str, Any]) -> int:
        return sum(
            str(node.get("semantic_effect") or "") == "vacuous"
            for node in result.get("nodes") or ()
        )

    initial_defects = None
    if comparator_results:
        initial_defects = comparator_defect_count(comparator_results[0])
        if decompiler_results:
            initial_defects += vacuous_count(decompiler_results[0])
    final_defects = None
    if comparator_results and row.get("status") != "structuralRejected":
        final_defects = comparator_defect_count(comparator_results[-1])
        if decompiler_results:
            final_defects += vacuous_count(decompiler_results[-1])

    attempts = [attempt for turn in history for attempt in turn.get("attempts") or ()]
    hard_failures = [attempt for attempt in attempts if attempt.get("hardErrors")]
    controller_decisions = Counter(
        str(attempt.get("controllerDecision"))
        for attempt in attempts
        if attempt.get("controllerDecision")
    )
    committed_attempts = sum(bool(turn.get("committed")) for turn in history)
    effective_counts = [len(attempt.get("effectiveNodes") or ()) for attempt in attempts]
    noop_count = sum(len(attempt.get("noOpNodes") or ()) for attempt in attempts)
    soft_invalid_commits = 0
    for turn in history:
        if not turn.get("committed"):
            continue
        committed = next(
            (
                attempt
                for attempt in reversed(turn.get("attempts") or ())
                if attempt.get("candidateHash") == turn.get("committedHash")
            ),
            None,
        )
        diag = (committed or {}).get("softDiagnostics") or {}
        if (
            not diag.get("whole_file_lean_success", True)
            or diag.get("semantic_errors")
            or diag.get("structural_errors")
            or diag.get("standalone_errors")
            or diag.get("pending_nodes")
        ):
            soft_invalid_commits += 1

    return {
        "source_id": row["source_id"],
        "strategy": strategy,
        "status": row.get("status"),
        "error": row.get("error") or "",
        "outer_turns": len(history),
        "editor_attempts": len(attempts),
        "hard_failures": len(hard_failures),
        "hard_failure_codes": dict(
            Counter(code for attempt in hard_failures for code in attempt.get("hardErrors") or ())
        ),
        "committed_turns": committed_attempts,
        "rolled_back_turns": len(history) - committed_attempts,
        "controller_decisions": dict(controller_decisions),
        "effective_edit_count": sum(effective_counts),
        "multi_node_candidate_count": sum(count >= 2 for count in effective_counts),
        "noop_edit_count": noop_count,
        "soft_invalid_commits": soft_invalid_commits,
        "initial_strict_defects": initial_defects,
        "final_strict_defects": final_defects,
        "strict_defect_delta": (
            final_defects - initial_defects
            if initial_defects is not None and final_defects is not None
            else None
        ),
        "initial_vacuous_nodes": vacuous_count(decompiler_results[0]) if decompiler_results else None,
        "final_vacuous_nodes": vacuous_count(decompiler_results[-1]) if decompiler_results else None,
        "tokens_by_phase": dict(usage),
        "calls_by_phase": dict(calls),
        "total_tokens": sum(usage.values()),
        "llm_calls": sum(calls.values()),
        "length_finishes": finish_length,
        "blueprint_dir": row.get("blueprint_dir"),
        "trace_path": row.get("trace_path"),
    }


def summarize_run(strategy: str, run_root: Path) -> dict[str, Any]:
    rows = read_jsonl(run_root / "robustpa" / "blueprint" / "results.jsonl")
    cases = [summarize_case(strategy, row) for row in rows]
    statuses = Counter(case["status"] for case in cases)
    accepted = {
        case["source_id"]
        for case in cases
        if case["status"] in {"strictAccepted", "acceptedWithJustifiedSideBranches"}
    }
    difficult = {case["source_id"] for case in cases} - CONTROLS
    duration = elapsed_seconds(run_root)
    summary = {
        "strategy": strategy,
        "run_root": str(run_root),
        "total": len(cases),
        "status_counts": dict(statuses),
        "accepted": sorted(accepted),
        "controls_kept": sorted(accepted & CONTROLS),
        "controls_regressed": sorted(CONTROLS - accepted),
        "difficult_recovered": sorted(accepted & difficult),
        "outer_turns": sum(case["outer_turns"] for case in cases),
        "editor_attempts": sum(case["editor_attempts"] for case in cases),
        "hard_failures": sum(case["hard_failures"] for case in cases),
        "committed_turns": sum(case["committed_turns"] for case in cases),
        "rolled_back_turns": sum(case["rolled_back_turns"] for case in cases),
        "soft_invalid_commits": sum(case["soft_invalid_commits"] for case in cases),
        "controller_commits": sum(
            case["controller_decisions"].get("COMMIT", 0) for case in cases
        ),
        "controller_retries": sum(
            case["controller_decisions"].get("RETRY_EDIT", 0) for case in cases
        ),
        "effective_edits": sum(case["effective_edit_count"] for case in cases),
        "noop_edits": sum(case["noop_edit_count"] for case in cases),
        "multi_node_candidates": sum(case["multi_node_candidate_count"] for case in cases),
        "initial_strict_defects": sum(case["initial_strict_defects"] or 0 for case in cases),
        "final_strict_defects": sum(case["final_strict_defects"] or 0 for case in cases),
        "total_tokens": sum(case["total_tokens"] for case in cases),
        "llm_calls": sum(case["llm_calls"] for case in cases),
        "length_finishes": sum(case["length_finishes"] for case in cases),
        "duration_seconds": duration,
    }
    return {"summary": summary, "cases": cases}


def main() -> None:
    runs = {name: summarize_run(name, path) for name, path in RUNS.items()}
    incomplete = {
        name: run["summary"]["total"] for name, run in runs.items() if run["summary"]["total"] != 10
    }
    if incomplete:
        raise RuntimeError(f"v9 experiments are incomplete: {incomplete}")

    payload = {
        "runs": runs,
        "decision": {
            "promotion_gate_passed": False,
            "reason": "No strategy recovered any of the seven difficult cases, and every strategy regressed one of three controls.",
            "preferred_base_for_next_iteration": "directEdit",
            "preferred_base_reason": "It matches the best acceptance count with the smallest orchestration surface, preserves deterministically valid final states, and reduces the aggregate strict defect inventory more than the planned variants.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    source_ids = sorted({case["source_id"] for run in runs.values() for case in run["cases"]})
    by_strategy = {
        name: {case["source_id"]: case for case in run["cases"]}
        for name, run in runs.items()
    }
    lines = [
        "# Phase 1B v9 三臂对比报告",
        "",
        "## 总览",
        "",
        "| 策略 | 通过 | 控制保持 | 难例恢复 | semantic/structural rejected | turns | editor attempts | hard failures | commits | tokens | LLM calls | duration |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in RUNS:
        s = runs[name]["summary"]
        statuses = s["status_counts"]
        lines.append(
            f"| {name} | {len(s['accepted'])}/10 | {len(s['controls_kept'])}/3 | "
            f"{len(s['difficult_recovered'])}/7 | "
            f"{statuses.get('semanticRejected', 0)}/{statuses.get('structuralRejected', 0)} | "
            f"{s['outer_turns']} | {s['editor_attempts']} | {s['hard_failures']} | "
            f"{s['committed_turns']} | {s['total_tokens']} | {s['llm_calls']} | "
            f"{s['duration_seconds']:.1f}s |"
        )

    lines.extend([
        "",
        "## 逐样例状态",
        "",
        "| source_id | progressController | planDirect | directEdit |",
        "|---|---|---|---|",
    ])
    for source_id in source_ids:
        lines.append(
            f"| {source_id} | "
            f"{by_strategy['progressController'][source_id]['status']} | "
            f"{by_strategy['planDirect'][source_id]['status']} | "
            f"{by_strategy['directEdit'][source_id]['status']} |"
        )

    lines.extend([
        "",
        "## 逐样例成本与语义缺陷变化",
        "",
        "`initial→final` 为 Strict Comparator 缺陷加 Formal Decompiler vacuous node 数；结构失败没有有效最终审计，显示为 N/A。`T/A/H/C` 为 outer turns、Editor attempts、hard failures、committed turns。",
        "",
        "| source_id | strategy | defects | T/A/H/C | effective/no-op edits | tokens |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for source_id in source_ids:
        for name in RUNS:
            case = by_strategy[name][source_id]
            final_defects = case["final_strict_defects"]
            final_label = "N/A" if final_defects is None else str(final_defects)
            lines.append(
                f"| {source_id} | {name} | {case['initial_strict_defects']}→{case['final_strict_defects']} | "
                f"{case['outer_turns']}/{case['editor_attempts']}/{case['hard_failures']}/{case['committed_turns']} | "
                f"{case['effective_edit_count']}/{case['noop_edit_count']} | {case['total_tokens']} |"
            )
            lines[-1] = lines[-1].replace(f"→{case['final_strict_defects']}", f"→{final_label}")

    lines.extend([
        "",
        "## 判定",
        "",
        "三个变体都只有 2/10 strictAccepted，均恢复 0/7 个困难样例，并各自回归 1/3 个控制样例；因此全部未达到 promotion gate，不应直接运行完整 76 条。",
        "",
        "- `progressController` 只在 75 个 turn 中提交 4 次，却产生 160 次 Editor attempt、70 次 Controller retry 和 334 万 token。它与 `directEdit` 的通过集合完全相同，但 token 为后者的 2.48 倍、耗时为 1.37 倍。二值 Controller 仍明显过度拒绝，不值得保留在主流程。",
        "- `planDirect` 最快，但 15 次 commit 中有 10 次带确定性 soft error，最终 5 条落入 structuralRejected；它证明了“只做 hard DAG check 就提交”会让不可编译或 Step mapping 错误污染后续基线。",
        "- `directEdit` 没有 Plan，仍以几乎相同于 `planDirect` 的总 token 获得相同通过数；58 次 commit 将汇总严格缺陷从 52 降到 25，并且最终 9/10 保持 whole-file/静态/standalone 合法。它是下一轮最合适的简洁基线，但当前质量仍不足。",
        "- no-op 放宽有效：`directEdit` 过滤 76 个 no-op 后仍完成 58 次 commit，没有再因单个 identical replacement 丢弃整个有效批次。",
        "",
        "## 关键样例",
        "",
        "- `hmmt_feb_2025/18`：Controller 0 commit、最终 4 个缺陷；Direct Edit 连续提交后只剩 1 个缺陷，即 `window_of_influence` 没有把窗口显式约束为以当前位置为中心。说明固定 Plan/Controller 反而阻止了可恢复的渐进编辑。",
        "- `precalculus/1056`：Direct Edit 已把 root 改为同时包含 `≤ 36` 的 closed-ball 对象，root target/answer grounding 得到修复；只剩 S011 对“surface 与 enclosed volume”的正式区分仍不充分。Controller 0 commit，维持 3 个缺陷。",
        "- `counting_and_probability/430`：Direct Edit 从 5 降至 2 个缺陷，最终显式写出 5/6 个选择与转移条件；仍没有真正的随机选择机制，也没有完整绑定 Alice 的五种选择与 Bob 的条件概率。",
        "- `counting_and_probability/731`：Direct Edit 从 9 降至 2，root 已直接陈述测度比；剩余问题是 rhombus 构型仍以存在点和长度约束近似，及最终求解关系仍被 Judge 视为弱绑定。",
        "- `geometry/434`：Direct Edit 从 6 降至 3，但声明膨胀为大量重复 conjunct/comment。形式层仍未提供平行线内错角、等腰三角形 base-angle 与图中 x 的共享角对象；增加文字和局部等式没有替代几何关系。",
        "- `aime_2024/81`：Direct Edit 只从 9 降至 7。最终 root 仍在计数 index set，而不是满足边/对角线约束的几何矩形；局部节点的推理方向也未对齐 COT。",
        "- `cmimc_2025/23`：Plan Direct 的 Comparator 缺陷曾降至 0，但 Decompiler 仍识别 3 个 vacuous arithmetic node，所以没有 strictAccepted；Direct Edit 还留下 5 个语义缺陷和 7 个 standalone syntax error。这说明自然语言 Plan 对一类全局几何链有帮助，但不足以稳定产出可独立编译、非空壳的节点。",
        "",
        "## 建议",
        "",
        "下一轮以 `directEdit` 为基线，保留 no-op 过滤与每 turn 最多三次 hard-format retry；恢复一个很小的提交安全线：候选可以暂时存在语义错误，但 whole-file Lean、Step mapping 和 standalone 失败不得成为 committed baseline。不要恢复 Planner 或 Controller。重点改 Editor prompt/训练数据，使其编辑正式对象、关系和 root dependency，而不是继续增加编排层级。",
        "",
    ])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
