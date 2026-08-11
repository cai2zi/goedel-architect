from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


WORK = Path("/ssd/czx/czx_work/cot_blueprint_refine")
RUNS = {
    "v9PlanDirect": WORK / "qwen3_8b_397b_wrong10_step_v9_plan_direct",
    "v10PlanDirect": WORK / "qwen3_8b_397b_wrong10_step_v10_plan_direct_stable_closure",
    "v9DirectEdit": WORK / "qwen3_8b_397b_wrong10_step_v9_direct_edit",
    "v10DirectEdit": WORK / "qwen3_8b_397b_wrong10_step_v10_direct_edit_stable_closure",
}
OUT = WORK / "qwen3_8b_397b_wrong10_step_v10_comparison_report"
CONTROLS = {
    "MATH-500/test/counting_and_probability/765.json",
    "MATH-500/test/prealgebra/378.json",
    "MATH-500/test/intermediate_algebra/662.json",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def elapsed_seconds(run_root: Path) -> float | None:
    path = run_root / "kimina" / "session.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    start = row.get("ready_at") or row.get("started_at")
    end = row.get("stopped_at")
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def comparator_defects(comparator: dict[str, Any]) -> int:
    total = 0
    for step in comparator.get("steps") or ():
        for field in ("missing_clauses", "weakened_clauses", "unbound_objects", "wrong_relations"):
            total += len(step.get(field) or ())
    root = comparator.get("root") or {}
    total += not bool(root.get("target_object_preserved"))
    total += not bool(root.get("answer_grounded"))
    total += sum(not bool(x.get("justified_side_branch")) for x in comparator.get("unreachable_steps") or ())
    total += len(comparator.get("dependency_issues") or ())
    return total


def final_strict_defects(row: dict[str, Any]) -> int | None:
    audit = ((row.get("phase1b_validation") or {}).get("semanticAudit") or {})
    decompiler = audit.get("formalDecompiler") or {}
    comparator = audit.get("strictComparator") or {}
    if not comparator:
        return None
    vacuous = sum(x.get("semantic_effect") == "vacuous" for x in decompiler.get("nodes") or ())
    return comparator_defects(comparator) + vacuous


def summarize_case(label: str, row: dict[str, Any]) -> dict[str, Any]:
    trace = read_jsonl(Path(row["trace_path"]))
    history = list(row.get("phase1b_edit_history") or ())
    attempts = [a for turn in history for a in turn.get("attempts") or ()]
    tokens: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    trace_counts: Counter[str] = Counter()
    stable_reasons: Counter[str] = Counter()
    semantic_reasons: Counter[str] = Counter()
    for event in trace:
        kind = str(event.get("kind") or "")
        trace_counts[kind] += 1
        args = event.get("args") or {}
        if kind == "llm_request_end" and event.get("ok"):
            phase = str(args.get("phase") or "unknown")
            tokens[phase] += int(args.get("total_tokens") or 0)
            calls[phase] += 1
        if kind == "phase1BStableGate" and not event.get("ok"):
            stable_reasons.update(str(x).split(":", 1)[0] for x in args.get("errors") or ())
        if kind == "phase1BCommitAssessment" and not event.get("ok"):
            semantic_reasons[str(args.get("reason") or "unknown")] += 1

    commits = [turn for turn in history if turn.get("committed")]
    committed_assessments = []
    for turn in commits:
        matching = [
            a for a in turn.get("attempts") or ()
            if a.get("candidateHash") == turn.get("committedHash")
        ]
        assessment = (matching[-1].get("commitAssessment") or {}) if matching else {}
        committed_assessments.append(assessment)
    def adds_deterministic_debt(assessment: dict[str, Any]) -> bool:
        before = assessment.get("baselineDebt") or {}
        after = assessment.get("candidateDebt") or {}
        if not bool(after.get("leanSuccess", True)):
            return True
        return any(
            set(after.get(field) or ()) - set(before.get(field) or ())
            for field in (
                "semanticErrors", "structuralErrors", "standaloneErrors", "pendingNodes",
            )
        )

    invalid_commits = sum(
        adds_deterministic_debt(assessment)
        for assessment in committed_assessments if assessment
    )
    closure_commits = [x for x in committed_assessments if x.get("closureMode")]
    closure_nonprogress = sum(
        not (
            len(x.get("normalizedOpenAfter") or {}) < len(x.get("normalizedOpenBefore") or {})
            or int(x.get("strictDefectAfter") or 0) < int(x.get("strictDefectBefore") or 0)
        )
        for x in closure_commits
    )
    first_assessment = next(
        (a.get("commitAssessment") for a in attempts if a.get("commitAssessment")), None
    )
    initial_defects = (
        int(first_assessment.get("strictDefectBefore")) if first_assessment else None
    )
    return {
        "source_id": row["source_id"],
        "label": label,
        "status": row.get("status"),
        "outer_turns": len(history),
        "editor_attempts": len(attempts),
        "hard_failures": sum(bool(a.get("hardErrors")) for a in attempts),
        "stable_retries": sum(stable_reasons.values()),
        "semantic_retries": sum(semantic_reasons.values()),
        "stable_reasons": dict(stable_reasons),
        "semantic_reasons": dict(semantic_reasons),
        "commits": len(commits),
        "closure_commits": len(closure_commits),
        "closure_nonprogress_commits": closure_nonprogress,
        "invalid_commits": invalid_commits,
        "foundation_commits": sum(bool(x.get("foundationOnly")) for x in committed_assessments),
        "effective_edits": sum(len(a.get("effectiveNodes") or ()) for a in attempts),
        "noop_edits": sum(len(a.get("noOpNodes") or ()) for a in attempts),
        "initial_strict_defects": initial_defects,
        "final_strict_defects": final_strict_defects(row),
        "total_tokens": sum(tokens.values()),
        "tokens_by_phase": dict(tokens),
        "llm_calls": sum(calls.values()),
        "calls_by_phase": dict(calls),
        "trace_counts": dict(trace_counts),
        "blueprint_dir": row.get("blueprint_dir"),
        "trace_path": row.get("trace_path"),
    }


def summarize_run(label: str, root: Path) -> dict[str, Any]:
    rows = read_jsonl(root / "robustpa" / "blueprint" / "results.jsonl")
    cases = [summarize_case(label, row) for row in rows]
    accepted = {x["source_id"] for x in cases if x["status"] in {"strictAccepted", "acceptedWithJustifiedSideBranches"}}
    statuses = Counter(x["status"] for x in cases)
    return {
        "summary": {
            "label": label,
            "run_root": str(root),
            "total": len(cases),
            "status_counts": dict(statuses),
            "accepted": sorted(accepted),
            "controls_kept": sorted(accepted & CONTROLS),
            "controls_regressed": sorted(CONTROLS - accepted),
            "difficult_recovered": sorted(accepted - CONTROLS),
            "turns": sum(x["outer_turns"] for x in cases),
            "editor_attempts": sum(x["editor_attempts"] for x in cases),
            "hard_failures": sum(x["hard_failures"] for x in cases),
            "stable_retries": sum(x["stable_retries"] for x in cases),
            "semantic_retries": sum(x["semantic_retries"] for x in cases),
            "commits": sum(x["commits"] for x in cases),
            "closure_commits": sum(x["closure_commits"] for x in cases),
            "closure_nonprogress_commits": sum(x["closure_nonprogress_commits"] for x in cases),
            "invalid_commits": sum(x["invalid_commits"] for x in cases),
            "tokens": sum(x["total_tokens"] for x in cases),
            "llm_calls": sum(x["llm_calls"] for x in cases),
            "duration_seconds": elapsed_seconds(root),
        },
        "cases": cases,
    }


def main() -> None:
    runs = {label: summarize_run(label, root) for label, root in RUNS.items()}
    incomplete = {k: v["summary"]["total"] for k, v in runs.items() if v["summary"]["total"] != 10}
    if incomplete:
        raise RuntimeError(f"incomplete experiments: {incomplete}")
    payload = {
        "runs": runs,
        "decision": {
            "quality_gate_passed": False,
            "reason": "Neither v10 arm recovered any of seven difficult cases; each kept only two of three controls.",
            "engineering_gate_passed": True,
            "engineering_reason": "Both v10 arms finished without structural/infra results or deterministically invalid commits; closure made no non-progress commits.",
            "preferred_strategy": "directEdit",
            "preferred_reason": "Same acceptance with no Planner call layer and no plan-node mismatch surface.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    order = list(RUNS)
    by_run = {k: {x["source_id"]: x for x in v["cases"]} for k, v in runs.items()}
    ids = sorted(next(iter(by_run.values())))
    lines = [
        "# Phase 1B v10 Stable Commit / Closure 双臂报告", "",
        "## 总览", "",
        "| run | accepted | controls | difficult | semantic/structural/infra | turns | attempts | hard/stable/semantic retry | commits | invalid commits | closure non-progress | tokens | calls | duration |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in order:
        s = runs[label]["summary"]
        statuses = s["status_counts"]
        duration = s["duration_seconds"]
        lines.append(
            f"| {label} | {len(s['accepted'])}/10 | {len(s['controls_kept'])}/3 | {len(s['difficult_recovered'])}/7 | "
            f"{statuses.get('semanticRejected', 0)}/{statuses.get('structuralRejected', 0)}/{statuses.get('infraError', 0)} | "
            f"{s['turns']} | {s['editor_attempts']} | {s['hard_failures']}/{s['stable_retries']}/{s['semantic_retries']} | "
            f"{s['commits']} | {s['invalid_commits']} | {s['closure_nonprogress_commits']} | {s['tokens']} | {s['llm_calls']} | "
            f"{duration:.1f}s |"
        )
    lines += ["", "## 逐样例状态与缺陷", "", "| source_id | v9 plan | v10 plan | v9 direct | v10 direct | v10 plan defects/commits | v10 direct defects/commits |", "|---|---|---|---|---|---:|---:|"]
    for source_id in ids:
        vp = by_run["v10PlanDirect"][source_id]
        vd = by_run["v10DirectEdit"][source_id]
        lines.append(
            f"| {source_id} | {by_run['v9PlanDirect'][source_id]['status']} | {vp['status']} | "
            f"{by_run['v9DirectEdit'][source_id]['status']} | {vd['status']} | "
            f"{vp['initial_strict_defects']}→{vp['final_strict_defects']}/{vp['commits']} | "
            f"{vd['initial_strict_defects']}→{vd['final_strict_defects']}/{vd['commits']} |"
        )
    lines += [
        "", "## 结论", "",
        "- 工程目标达成：v10 两臂均为 0 structuralRejected、0 infraError、0 deterministically invalid commit；Closure 中也没有不降低 obligation/strict defect 的 commit。Stable Gate 成功阻止 Lean、静态、structural、standalone 新债务进入 committed baseline。",
        "- 质量目标未达成：两臂都只有 2/10 strictAccepted，困难样例恢复 0/7，且控制样例都只保持 2/3。因此不应直接扩展到完整 76 条。",
        "- `planDirect` 的节点白名单继续制造大量 hard retry；它没有换来更高通过率。`directEdit` 更简洁，应作为后续基线，但当前 Editor 的共享对象/关系重建能力仍不足。",
        "- Closure 能防止最后两轮继续扩大债务，但不能替代早期正确建模。多数难例进入 Closure 时仍有 root-object 或 missing relation obligation，已经没有足够空间重建整条 formal path。",
        "", "## 重点样例", "",
        "- `hmmt/18`：两臂都至少产生稳定 commit，但 `window_of_influence` 只存在量化一个任意 `iterate_max`，没有把它绑定到题目的逐秒更新算子；最终 S002 仍不忠实。no-op/稳定提交问题已解决，剩余瓶颈是 relation modeling。",
        "- `precalculus/1056`：directEdit 已使 Judge 的 root target/grounding 都为 true，但 S011 对 sphere surface 与 enclosed `≤ 36` volume region 的完整解释仍缺一项 formal clause。Closure 能保存正确 root，不能凭局部补丁补齐所有中间语义。",
        "- `counting/731`：最终 `region_bounded_by_bisectors` 和 `rhombus_area` 仍用局部 `True` 表示菱形性质，root 又把 `probability` 定义为目标常量后证明自等式；事件区域、菱形构型、测度比之间仍未形成 formal binding。",
        "- `cmimc/23`：新增交点/圆对象若未被最终面积 root 的 formal type/dependency 消费，会被 Semantic Progress Gate 拒绝；这避免了 v7 的假进展，但模型仍没有产出可接受的完整路径。",
        "- `geometry/434`：`x_interpretation` 仍是 `x = x`，`angle_A_value` 又先把角定义为 62；root 虽写出 arccos 对象，但推导链仍以常量绑定替代平行线、等腰三角形与共享角关系。",
        "", "完整逐 case 指标、拒绝原因和 token 分相见 `report.json`。", "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
