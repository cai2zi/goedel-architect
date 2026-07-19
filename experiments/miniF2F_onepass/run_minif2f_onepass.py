from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from shared.io_utils import (  # noqa: E402
    append_jsonl,
    default_output_root,
    read_jsonl,
    rows_by_id,
    safe_stem,
    unlink_if_exists,
    write_json,
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


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "base.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    config_path = config_path_from_argv(argv, DEFAULT_CONFIG)
    config = load_yaml_config(config_path)
    apply_config_environment(config)
    cot_config = config.get("cot") or {}
    if cot_config and not isinstance(cot_config, dict):
        raise ValueError("Config key 'cot' must be an object.")

    parser = argparse.ArgumentParser(description="Run miniF2F one-pass blueprint/proof experiment.")
    add_config_arg(parser, DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["test", "valid"], default="test")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "minif2f")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problem-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--node-timeout-s", type=int, default=300)
    parser.add_argument("--phase1-concurrency", type=int, default=4)
    parser.add_argument("--phase2-blueprint-concurrency", type=int, default=4)
    parser.add_argument("--phase2-node-concurrency", type=int, default=8)
    cot_path_default = cot_config.get("path")
    parser.add_argument("--cot-path", type=Path, default=Path(cot_path_default) if cot_path_default else None)
    parser.add_argument("--cot-id-field", default=cot_config.get("id_field", "name"))
    parser.add_argument("--cot-text-field", default=cot_config.get("text_field", "nl_proof"))
    parser.add_argument(
        "--cot-allow-missing",
        action=argparse.BooleanOptionalAction,
        default=bool(cot_config.get("allow_missing", False)),
    )
    add_lean_runtime_args(parser)
    set_defaults_from_config(parser, config, ignore={"config", "cot_path", "cot_id_field", "cot_text_field", "cot_allow_missing"})
    return parser.parse_args(argv)


def _problem_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("name") or row.get("id") or row.get("problem_id") or f"miniF2F_{index}")


def _theorem_stmt(row: dict[str, Any]) -> str:
    value = row.get("formal_statement") or row.get("statement") or row.get("theorem")
    if not value:
        raise ValueError("miniF2F row has no formal_statement/statement/theorem field")
    return str(value)


def _nl_proof(row: dict[str, Any]) -> str:
    return str(row.get("informal_proof") or row.get("proof") or "")


def _load_cot_map(args: argparse.Namespace) -> dict[str, str]:
    if args.cot_path is None:
        return {}
    if not args.cot_path.exists():
        raise FileNotFoundError(f"COT file not found: {args.cot_path}")
    rows = read_jsonl(args.cot_path)
    cot_by_id: dict[str, str] = {}
    for row in rows:
        if args.cot_id_field not in row:
            raise ValueError(f"COT row missing id field '{args.cot_id_field}'")
        if args.cot_text_field not in row:
            raise ValueError(f"COT row missing text field '{args.cot_text_field}'")
        key = str(row[args.cot_id_field])
        if key in cot_by_id:
            raise ValueError(f"Duplicate COT id: {key}")
        cot_by_id[key] = str(row[args.cot_text_field] or "")
    return cot_by_id


def _validate_concurrency(args: argparse.Namespace) -> None:
    for name in ("phase1_concurrency", "phase2_blueprint_concurrency", "phase2_node_concurrency"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")


def _select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(rows, 1):
        problem_id = _problem_id(row, idx)
        if args.problem_id and problem_id != args.problem_id:
            continue
        selected.append((idx, row))
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def _is_completed_result(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("root_proved"))


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
            try:
                result = await loop.run_in_executor(
                    executor,
                    partial(
                        run_onepass_phase1,
                        record_id=record["record_id"],
                        theorem_stmt=record["theorem_stmt"],
                        nl_proof=record["nl_proof"],
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
            tqdm.write(f"[phase1-{result['status']}] {record['record_id']}")
            return record, result
        tasks = [asyncio.create_task(run_one(record)) for record in records]
        results: list[tuple[dict[str, Any], dict[str, Any]]] = []
        with tqdm(total=len(tasks), desc="phase1", unit="record") as progress:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
                progress.update(1)
        return results


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
            async with blueprint_sem:
                tqdm.write(f"[phase2-start] {record['record_id']}")
                result = await run_onepass_phase2_async(
                    record_id=record["record_id"],
                    theorem_stmt=record["theorem_stmt"],
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
                tqdm.write(f"[phase2-done] {record['record_id']} root_proved={bool(result.get('root_proved'))}")
                return record, result
        tasks = [asyncio.create_task(run_one(record, phase1)) for record, phase1 in records]
        results: list[tuple[dict[str, Any], dict[str, Any]]] = []
        with tqdm(total=len(tasks), desc="phase2", unit="record") as progress:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
                progress.update(1)
        return results


def _decorate_result(row: dict[str, Any], *, source_id: str, split: str, runtime: LeanRuntime) -> dict[str, Any]:
    return {
        **row,
        "source_id": source_id,
        "split": split,
        "phase0_success": True,
        "success": bool(row.get("root_proved")),
        "lean_runtime": runtime.metadata,
    }


def _run_experiment(args: argparse.Namespace, output_root: Path, runtime: LeanRuntime) -> None:
    data_path = args.data_dir / f"{args.split}.jsonl"
    rows = read_jsonl(data_path)
    selected = _select_rows(rows, args)
    cot_by_id = _load_cot_map(args)

    results_path = output_root / "results.jsonl"
    metrics_path = output_root / "metrics.json"
    metrics_csv_path = output_root / "metrics.csv"
    if not args.resume:
        for path in (results_path, metrics_path, metrics_csv_path):
            unlink_if_exists(path)

    done = rows_by_id(results_path)
    print(f"[select] split={args.split} problems={len(selected)} output={output_root}")
    print(
        "[concurrency] "
        f"phase1={args.phase1_concurrency} "
        f"phase2_blueprint={args.phase2_blueprint_concurrency} "
        f"phase2_node={args.phase2_node_concurrency} "
        f"lean_check={args.lean_check_concurrency}",
        flush=True,
    )
    pending: list[dict[str, Any]] = []
    for idx, row in selected:
        source_id = _problem_id(row, idx)
        record_id = safe_stem(source_id, prefix="miniF2F_")
        if args.resume and _is_completed_result(done.get(record_id)):
            print(f"[resume] skip completed {record_id}")
            continue

        try:
            theorem_stmt = _theorem_stmt(row)
            if args.cot_path is not None:
                if source_id in cot_by_id:
                    nl_proof = cot_by_id[source_id]
                elif args.cot_allow_missing:
                    nl_proof = _nl_proof(row)
                else:
                    raise ValueError(f"COT missing for problem id '{source_id}'")
            else:
                nl_proof = _nl_proof(row)
            pending.append({
                "record_id": record_id,
                "source_id": source_id,
                "theorem_stmt": theorem_stmt,
                "nl_proof": nl_proof,
            })
        except Exception as exc:
            onepass = _error_onepass_row(record_id=record_id, output_root=output_root, error=str(exc))
            result = _decorate_result(onepass, source_id=source_id, split=args.split, runtime=runtime)
            append_jsonl(results_path, result)
            done[record_id] = result

    phase1_start = time.perf_counter()
    phase1_results = asyncio.run(_run_phase1_batch(pending, args=args, output_root=output_root, runtime=runtime))
    phase1_duration_s = round(time.perf_counter() - phase1_start, 3)
    print(f"[runtime] phase1_duration_s={phase1_duration_s}", flush=True)
    phase2_ready: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record, phase1 in phase1_results:
        if phase1["status"] == "ready":
            phase2_ready.append((record, phase1))
            continue
        result = _decorate_result(phase1["row"], source_id=record["source_id"], split=args.split, runtime=runtime)
        append_jsonl(results_path, result)
        done[record["record_id"]] = result

    phase2_start = time.perf_counter()
    phase2_results = asyncio.run(_run_phase2_batch(phase2_ready, args=args, output_root=output_root, runtime=runtime))
    phase2_duration_s = round(time.perf_counter() - phase2_start, 3)
    print(f"[runtime] phase2_duration_s={phase2_duration_s}", flush=True)
    for record, onepass in phase2_results:
        result = _decorate_result(onepass, source_id=record["source_id"], split=args.split, runtime=runtime)
        append_jsonl(results_path, result)
        done[record["record_id"]] = result

    result_rows = [row for row in done.values() if not args.problem_id or row.get("source_id") == args.problem_id]
    if args.limit is not None:
        selected_ids = {safe_stem(_problem_id(row, idx), prefix="miniF2F_") for idx, row in selected}
        result_rows = [row for row in result_rows if row.get("id") in selected_ids]
    total = len(result_rows)
    solved = sum(1 for row in result_rows if row.get("success"))
    metrics = {
        "split": args.split,
        "problem_count": total,
        "solved_count": solved,
        "accuracy": solved / total if total else 0.0,
        "root_proved_count": sum(1 for row in result_rows if row.get("root_proved")),
        "blueprint_success_count": sum(1 for row in result_rows if row.get("blueprint_success")),
        "phase1_duration_s": phase1_duration_s,
        "phase2_duration_s": phase2_duration_s,
    }
    write_json(metrics_path, metrics)
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"[done] results={results_path}")
    print(f"[metrics] {metrics}")


def main() -> None:
    args = parse_args()
    _validate_concurrency(args)
    output_root = args.output_root or default_output_root(REPO_ROOT, "miniF2F_onepass", args.model) / args.split
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
