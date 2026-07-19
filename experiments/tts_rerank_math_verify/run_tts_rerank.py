from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from shared.io_utils import (  # noqa: E402
    append_jsonl,
    default_output_root,
    load_json,
    rows_by_id,
    safe_stem,
    to_bool,
    unlink_if_exists,
    write_json,
    write_jsonl,
)
from shared.config_utils import (  # noqa: E402
    add_config_arg,
    apply_config_environment,
    config_path_from_argv,
    load_yaml_config,
    set_defaults_from_config,
)
from shared.lean_runtime import (  # noqa: E402
    LeanRuntime,
    add_lean_runtime_args,
    make_lean_runtime,
    prepare_lean_runtime_metadata,
)
from shared.onepass import run_onepass_phase1, run_onepass_phase2_async  # noqa: E402
from shared.phase0 import formalize_candidate  # noqa: E402
from shared.scoring import pick_best_rollout, vote_by_answer  # noqa: E402
from lean_compiler import LeanCompiler  # noqa: E402


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "base.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    config_path = config_path_from_argv(argv, DEFAULT_CONFIG)
    config = load_yaml_config(config_path)
    apply_config_environment(config)

    parser = argparse.ArgumentParser(description="Run Goedel one-pass TTS reranking on bench.json.")
    add_config_arg(parser, DEFAULT_CONFIG)
    parser.add_argument("--bench-path", type=Path, default=REPO_ROOT.parent / "czx_work" / "bench.json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eligible parent problems.")
    parser.add_argument("--problem-id", default=None)
    parser.add_argument("--rollout-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-empty-extracted", action="store_true")
    parser.add_argument("--phase0-max-attempts", type=int, default=3)
    parser.add_argument("--node-timeout-s", type=int, default=300)
    parser.add_argument("--phase1-concurrency", type=int, default=4)
    parser.add_argument("--phase2-blueprint-concurrency", type=int, default=4)
    parser.add_argument("--phase2-node-concurrency", type=int, default=8)
    add_lean_runtime_args(parser)
    if "exclude_empty_extracted_answer" in config:
        parser.set_defaults(include_empty_extracted=not bool(config["exclude_empty_extracted_answer"]))
    set_defaults_from_config(parser, config, ignore={"config", "include_empty_extracted"})
    return parser.parse_args(argv)


def _iter_problems(bench: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(bench.get("problems"), list):
        return list(bench["problems"])
    if isinstance(bench.get("items"), list):
        return list(bench["items"])
    raise ValueError("bench.json must contain a top-level 'problems' or 'items' list")


def _rollouts(problem: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rollouts", "samples", "answers"):
        value = problem.get(key)
        if isinstance(value, list):
            return value
    return []


def _problem_id(problem: dict[str, Any]) -> str:
    return str(problem.get("id") or problem.get("parent_id") or problem.get("problem_id"))


def _question(problem: dict[str, Any], rollout: dict[str, Any]) -> str:
    return str(
        problem.get("question")
        or problem.get("problem")
        or problem.get("prompt")
        or rollout.get("question")
        or rollout.get("problem")
        or rollout.get("prompt")
        or ""
    )


def _canonical_answer(rollout: dict[str, Any]) -> str:
    value = rollout.get("canonical_extracted_answer")
    if value is None:
        value = rollout.get("extracted_answer_first")
    if value is None:
        extracted = rollout.get("extracted_answer")
        if isinstance(extracted, list) and extracted:
            value = extracted[0]
        elif isinstance(extracted, str):
            value = extracted
    return "" if value is None else str(value)


def _candidate_text(rollout: dict[str, Any]) -> str:
    return str(rollout.get("answer") or rollout.get("nl_proof") or rollout.get("response") or "")


def _gold(problem: dict[str, Any], rollout: dict[str, Any]) -> str:
    return str(problem.get("gold") or rollout.get("gold") or "")


def _rollout_id(rollout: dict[str, Any], fallback: int) -> int:
    value = rollout.get("rollout_id")
    if value is None:
        value = rollout.get("rollout")
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _rollout_record_id(parent_id: str, rollout_id: int) -> str:
    return f"{parent_id}__rollout_{rollout_id}"


def _is_terminal_score(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    if row.get("root_proved"):
        return True
    # Phase 0 failures have no blueprint to resume. Retrying formalization is
    # a separate policy from Phase 1/2 checkpoint resume.
    return not bool(row.get("phase0_success", True))


def _phase0_score_row(phase0_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": phase0_row["id"],
        "parent_id": phase0_row["parent_id"],
        "rollout_id": phase0_row["rollout_id"],
        "phase0_success": False,
        "blueprint_success": False,
        "root_theorem": "",
        "root_proved": False,
        "total_nodes": 0,
        "proved_node_count": 0,
        "proved_ratio": 0.0,
        "failed_nodes": [],
        "checkpoint_path": "",
        "trace_path": "",
        "math_verify_is_correct": phase0_row["math_verify_is_correct"],
        "canonical_extracted_answer": phase0_row["candidate_answer"],
        "extracted_answer_consistent": phase0_row.get("extracted_answer_consistent", True),
        "warning": phase0_row.get("warning", ""),
        "error": phase0_row.get("phase0_error", ""),
        "lean_runtime": phase0_row.get("lean_runtime"),
    }


def _selected_problem_ids(
    problems: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> tuple[list[str], int]:
    selected: list[str] = []
    excluded_empty = 0
    for problem in problems:
        parent_id = _problem_id(problem)
        if args.problem_id and parent_id != args.problem_id:
            continue
        rollouts = _rollouts(problem)
        has_empty = any(not _canonical_answer(r) for r in rollouts)
        if args.rollout_id is not None:
            rollouts = [r for i, r in enumerate(rollouts, 1) if _rollout_id(r, i) == args.rollout_id]
        if has_empty and not args.include_empty_extracted:
            excluded_empty += 1
            print(f"[skip-empty] {parent_id}: at least one rollout has empty canonical_extracted_answer")
            continue
        selected.append(parent_id)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected, excluded_empty


def _make_phase0_row(
    *,
    problem: dict[str, Any],
    rollout: dict[str, Any],
    rollout_index: int,
    model: str,
    max_attempts: int,
    compiler: LeanCompiler,
    lean_runtime: dict[str, Any],
) -> dict[str, Any]:
    parent_id = _problem_id(problem)
    rollout_id = _rollout_id(rollout, rollout_index)
    record_id = _rollout_record_id(parent_id, rollout_id)
    theorem_name = safe_stem(record_id, prefix="tts_")
    candidate_answer = _canonical_answer(rollout)
    extracted_answer_consistent = bool(rollout.get("extracted_answer_consistent", True))
    warning = "" if extracted_answer_consistent else "extracted_answer_consistent=false; using canonical first value"

    phase0 = formalize_candidate(
        question=_question(problem, rollout),
        candidate_answer=candidate_answer,
        nl_proof=_candidate_text(rollout),
        theorem_name=theorem_name,
        model=model,
        max_attempts=max_attempts,
        compiler=compiler,
    )
    if not phase0.success:
        print(f"[phase0-fail] {record_id}: {phase0.error[:500]}")

    return {
        "id": record_id,
        "parent_id": parent_id,
        "rollout_id": rollout_id,
        "question": _question(problem, rollout),
        "gold": _gold(problem, rollout),
        "candidate_answer": candidate_answer,
        "canonical_extracted_answer": candidate_answer,
        "nl_proof": _candidate_text(rollout),
        "theorem_stmt": phase0.theorem_stmt,
        "phase0_success": phase0.success,
        "phase0_error": phase0.error,
        "phase0_attempts": phase0.attempts,
        "math_verify_is_correct": to_bool(rollout.get("is_correct")),
        "extracted_answer_consistent": extracted_answer_consistent,
        "warning": warning,
        "lean_runtime": lean_runtime,
    }


def _compute_metrics(
    *,
    selected_ids: list[str],
    score_rows: list[dict[str, Any]],
    excluded_empty: int,
    bench_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_set = set(selected_ids)
    for row in score_rows:
        if row.get("parent_id") in selected_set:
            by_parent[str(row["parent_id"])].append(row)

    best_rows: list[dict[str, Any]] = []
    pass1 = passn = vote_acc = goedel_acc = hit_on_passn = 0
    eligible_for_hit = 0
    for parent_id in selected_ids:
        rows = sorted(by_parent.get(parent_id, []), key=lambda r: int(r.get("rollout_id") or 10**9))
        if not rows:
            continue
        first = next((r for r in rows if int(r.get("rollout_id") or -1) == 1), rows[0])
        has_correct = any(r.get("math_verify_is_correct") for r in rows)
        vote_row = vote_by_answer(rows)
        best_row = pick_best_rollout(rows)
        if best_row is None:
            continue

        pass1 += int(bool(first.get("math_verify_is_correct")))
        passn += int(has_correct)
        vote_acc += int(bool(vote_row and vote_row.get("math_verify_is_correct")))
        goedel_acc += int(bool(best_row.get("math_verify_is_correct")))
        if has_correct:
            eligible_for_hit += 1
            hit_on_passn += int(bool(best_row.get("math_verify_is_correct")))

        best_rows.append({
            "parent_id": parent_id,
            "selected_id": best_row["id"],
            "selected_rollout_id": best_row.get("rollout_id"),
            "selected_canonical_extracted_answer": best_row.get("canonical_extracted_answer"),
            "selected_math_verify_is_correct": best_row.get("math_verify_is_correct"),
            "vote_selected_id": vote_row.get("id") if vote_row else "",
            "vote_math_verify_is_correct": bool(vote_row and vote_row.get("math_verify_is_correct")),
            "pass_at_1": bool(first.get("math_verify_is_correct")),
            "pass_at_n": has_correct,
            "root_proved": best_row.get("root_proved"),
            "proved_ratio": best_row.get("proved_ratio"),
            "proved_node_count": best_row.get("proved_node_count"),
            "total_nodes": best_row.get("total_nodes"),
            "lean_runtime": best_row.get("lean_runtime"),
        })

    denom = len(best_rows)
    metrics = {
        "problem_count": denom,
        "rollout_count": sum(len(by_parent.get(pid, [])) for pid in selected_ids),
        "pass@1": pass1 / denom if denom else 0.0,
        "pass@N": passn / denom if denom else 0.0,
        "vote": vote_acc / denom if denom else 0.0,
        "goedel_best": goedel_acc / denom if denom else 0.0,
        "selection_hit_rate_on_pass@N_correct": hit_on_passn / eligible_for_hit if eligible_for_hit else 0.0,
        "excluded_empty_extracted_answer_problem_count": bench_metadata.get(
            "empty_extracted_answer_problem_count",
            excluded_empty,
        ),
        "selected_empty_extracted_answer_problem_count": excluded_empty,
    }
    return best_rows, metrics


def _validate_concurrency(args: argparse.Namespace) -> None:
    for name in ("phase1_concurrency", "phase2_blueprint_concurrency", "phase2_node_concurrency"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")


def _error_onepass_row(*, record_id: str, output_root: Path, error: str, include_traceback: bool = False) -> dict[str, Any]:
    row = {
        "id": record_id,
        "blueprint_success": False,
        "blueprint_reused": False,
        "phase1_skipped": False,
        "error": error,
        "checkpoint_path": str(output_root / "checkpoints" / f"{record_id}.json"),
        "trace_path": str(output_root / "traces" / f"{record_id}.jsonl"),
        "blueprint_path": str(output_root / "blueprints" / f"{record_id}.lean"),
        "root_theorem": "",
        "root_proved": False,
        "total_nodes": 0,
        "proved_node_count": 0,
        "proved_ratio": 0.0,
        "failed_nodes": [],
        "proved_nodes": [],
        "all_proved": False,
    }
    if include_traceback:
        row["traceback"] = traceback.format_exc()
    return row


def _score_from_onepass(onepass: dict[str, Any], phase0_row: dict[str, Any], runtime: LeanRuntime) -> dict[str, Any]:
    return {
        **onepass,
        "parent_id": phase0_row["parent_id"],
        "rollout_id": phase0_row["rollout_id"],
        "phase0_success": True,
        "math_verify_is_correct": phase0_row["math_verify_is_correct"],
        "canonical_extracted_answer": phase0_row["candidate_answer"],
        "extracted_answer_consistent": phase0_row.get("extracted_answer_consistent", True),
        "warning": phase0_row.get("warning", ""),
        "lean_runtime": runtime.metadata,
    }


async def _run_phase1_batch(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_root: Path,
    runtime: LeanRuntime,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=args.phase1_concurrency) as executor:
        async def run_one(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            phase0_row = record["phase0_row"]
            try:
                result = await loop.run_in_executor(
                    executor,
                    partial(
                        run_onepass_phase1,
                        record_id=record["record_id"],
                        theorem_stmt=phase0_row["theorem_stmt"],
                        nl_proof=phase0_row.get("nl_proof", ""),
                        model=args.model,
                        output_root=output_root,
                        resume=args.resume,
                        compiler=runtime.compiler,
                    ),
                )
            except Exception as exc:
                result = {
                    "status": "failed",
                    "row": _error_onepass_row(
                        record_id=record["record_id"],
                        output_root=output_root,
                        error=str(exc),
                        include_traceback=True,
                    ),
                }
            print(f"[phase1-{result['status']}] {record['record_id']}", flush=True)
            return record, result
        return await asyncio.gather(*(run_one(record) for record in records))


async def _run_phase2_batch(
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    args: argparse.Namespace,
    output_root: Path,
    runtime: LeanRuntime,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    blueprint_sem = asyncio.Semaphore(args.phase2_blueprint_concurrency)
    node_sem = asyncio.Semaphore(args.phase2_node_concurrency)
    with ThreadPoolExecutor(max_workers=args.phase2_node_concurrency) as node_executor:
        async def run_one(record: dict[str, Any], phase1: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            phase0_row = record["phase0_row"]
            async with blueprint_sem:
                print(f"[phase2-start] {record['record_id']}", flush=True)
                result = await run_onepass_phase2_async(
                    record_id=record["record_id"],
                    theorem_stmt=phase0_row["theorem_stmt"],
                    model=args.model,
                    output_root=output_root,
                    node_timeout_s=args.node_timeout_s,
                    compiler=runtime.compiler,
                    compiler_factory=runtime.compiler_factory,
                    blueprint_reused=bool(phase1.get("blueprint_reused")),
                    phase1_skipped=bool(phase1.get("phase1_skipped")),
                    node_executor=node_executor,
                    node_semaphore=node_sem,
                )
                print(f"[phase2-done] {record['record_id']} root_proved={bool(result.get('root_proved'))}", flush=True)
                return record, result
        return await asyncio.gather(*(run_one(record, phase1) for record, phase1 in records))


def _run_experiment(args: argparse.Namespace, output_root: Path, runtime: LeanRuntime) -> None:
    phase0_path = output_root / "phase0_results.jsonl"
    scores_path = output_root / "rollout_scores.jsonl"
    best_path = output_root / "goedel_best.jsonl"
    metrics_path = output_root / "metrics.json"
    metrics_csv_path = output_root / "metrics.csv"

    if not args.resume:
        for path in (phase0_path, scores_path, best_path, metrics_path, metrics_csv_path):
            unlink_if_exists(path)

    bench = load_json(args.bench_path)
    problems = _iter_problems(bench)
    selected_ids, excluded_empty = _selected_problem_ids(problems, args=args)
    selected_set = set(selected_ids)
    print(f"[select] problems={len(selected_ids)} excluded_empty={excluded_empty} output={output_root}")

    phase0_by_id = rows_by_id(phase0_path)
    score_by_id = rows_by_id(scores_path)

    print(
        "[concurrency] "
        f"phase1={args.phase1_concurrency} "
        f"phase2_blueprint={args.phase2_blueprint_concurrency} "
        f"phase2_node={args.phase2_node_concurrency} "
        f"lean_check={args.lean_check_concurrency}",
        flush=True,
    )
    pending: list[dict[str, Any]] = []
    for problem in problems:
        parent_id = _problem_id(problem)
        if parent_id not in selected_set:
            continue
        rollouts = _rollouts(problem)
        for idx, rollout in enumerate(rollouts, 1):
            rollout_id = _rollout_id(rollout, idx)
            if args.rollout_id is not None and rollout_id != args.rollout_id:
                continue
            record_id = _rollout_record_id(parent_id, rollout_id)
            if args.resume and _is_terminal_score(score_by_id.get(record_id)):
                print(f"[resume] skip terminal score {record_id}")
                continue

            phase0_row = phase0_by_id.get(record_id)
            if phase0_row is None:
                phase0_row = _make_phase0_row(
                    problem=problem,
                    rollout=rollout,
                    rollout_index=idx,
                    model=args.model,
                    max_attempts=args.phase0_max_attempts,
                    compiler=runtime.compiler,
                    lean_runtime=runtime.metadata,
                )
                append_jsonl(phase0_path, phase0_row)
                phase0_by_id[record_id] = phase0_row

            if not phase0_row.get("phase0_success"):
                score_row = _phase0_score_row(phase0_row)
                append_jsonl(scores_path, score_row)
                score_by_id[record_id] = score_row
                continue

            pending.append({
                "record_id": record_id,
                "phase0_row": phase0_row,
            })

    phase1_results = asyncio.run(_run_phase1_batch(pending, args=args, output_root=output_root, runtime=runtime))
    phase2_ready: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record, phase1 in phase1_results:
        if phase1["status"] == "ready":
            phase2_ready.append((record, phase1))
            continue
        score_row = _score_from_onepass(phase1["row"], record["phase0_row"], runtime)
        append_jsonl(scores_path, score_row)
        score_by_id[record["record_id"]] = score_row

    phase2_results = asyncio.run(_run_phase2_batch(phase2_ready, args=args, output_root=output_root, runtime=runtime))
    for record, onepass in phase2_results:
        score_row = _score_from_onepass(onepass, record["phase0_row"], runtime)
        append_jsonl(scores_path, score_row)
        score_by_id[record["record_id"]] = score_row

    score_rows = list(score_by_id.values())
    best_rows, metrics = _compute_metrics(
        selected_ids=selected_ids,
        score_rows=score_rows,
        excluded_empty=excluded_empty,
        bench_metadata=bench.get("metadata", {}),
    )
    write_jsonl(best_path, best_rows)
    write_json(metrics_path, metrics)
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"[done] phase0={phase0_path}")
    print(f"[done] scores={scores_path}")
    print(f"[done] best={best_path}")
    print(f"[metrics] {metrics}")


def main() -> None:
    args = parse_args()
    _validate_concurrency(args)
    output_root = args.output_root or default_output_root(REPO_ROOT, "tts_rerank_math_verify", args.model)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = make_lean_runtime(args)
    try:
        prepare_lean_runtime_metadata(
            output_root,
            resume=args.resume,
            metadata=runtime.metadata,
        )
        _run_experiment(args, output_root, runtime)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
