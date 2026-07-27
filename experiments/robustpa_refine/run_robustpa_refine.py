from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from blueprint import Blueprint, generate_blueprint_from_informal  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from shared.config_utils import (  # noqa: E402
    add_config_arg,
    apply_config_environment,
    config_path_from_argv,
    load_yaml_config,
    set_defaults_from_config,
)
from shared.io_utils import append_jsonl, safe_stem, unlink_if_exists, write_json  # noqa: E402
from shared.lean_runtime import (  # noqa: E402
    LeanRuntime,
    add_lean_runtime_args,
    make_lean_runtime,
    prepare_lean_runtime_metadata,
)
from mathlib_retrieval import MathlibRetrieval  # noqa: E402
from pipeline import run_phase2_async, run_phase3  # noqa: E402
from tracer import JsonlTracer, TraceEvent  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "base.yaml"
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "czx_work" / "RobustPABench"
DEFAULT_OUTPUT_BASE = REPO_ROOT.parent / "czx_work" / "robustpa_refine"
DEFAULT_MODEL = "deepseek-v4-flash"


def _exp_name_component(values: str | list[str] | None, fallback: str) -> str:
    if values is None:
        return fallback
    if isinstance(values, str):
        return safe_stem(values)
    parts = [safe_stem(str(value)) for value in values if str(value).strip()]
    return "_".join(parts) if parts else fallback


def default_exp_name(model: str, splits: list[str] | None, subsets: list[str] | None) -> str:
    model_part = safe_stem(model)
    split_part = _exp_name_component(splits, "all")
    subset_part = _exp_name_component(subsets, "all")
    return f"{model_part}_{split_part}_{subset_part}"


def _optional_timeout(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timeout = float(value)
    else:
        text = str(value).strip().lower()
        if text in {"", "none", "null", "no", "false"}:
            return None
        timeout = float(text)
    if timeout == 0:
        return None
    return timeout


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"", "none", "null", "default", "omit"}:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected one of: true/false/null, 1/0, yes/no, on/off"
    )


@dataclass(frozen=True)
class Record:
    unique_id: str
    record_id: str
    source_id: str
    subset: str
    split: str
    parquet_path: Path
    row_index: int
    theorem_name: str
    informal_statement: str
    informal_proof: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    config = load_yaml_config(config_path_from_argv(argv, DEFAULT_CONFIG))
    apply_config_environment(config)

    parser = argparse.ArgumentParser(description="Run RobustPABench informal-only blueprint/refine experiment.")
    add_config_arg(parser, DEFAULT_CONFIG)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problem-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--max-refinement-iterations", type=int, default=8)
    parser.add_argument("--blueprint-max-retries", type=int, default=8)
    parser.add_argument("--refine-max-retries", type=int, default=8)
    parser.add_argument("--node-max-tool-calls", type=int, default=8)
    parser.add_argument(
        "--parallel-tool-calls",
        type=_optional_bool,
        default=None,
        metavar="{true,false,null}",
        help=(
            "Whether to pass parallel_tool_calls to chat.completions. "
            "true/1 allows multiple tool calls in one assistant response; "
            "false/0 asks for at most one; null/none omits the field."
        ),
    )
    parser.add_argument("--node-timeout-s", type=_optional_timeout, default=300.0)
    parser.add_argument("--llm-api-timeout-s", type=_optional_timeout, default=120.0)
    parser.add_argument("--phase1-concurrency", type=int, default=4)
    parser.add_argument("--phase2-blueprint-concurrency", type=int, default=4)
    parser.add_argument("--phase2-node-concurrency", type=int, default=8)
    parser.add_argument("--refine-concurrency", type=int, default=2)
    add_lean_runtime_args(parser)
    parser.set_defaults(lean_backend="local")
    set_defaults_from_config(parser, config, ignore={"config"})
    args = parser.parse_args(argv)
    if not args.exp_name:
        args.exp_name = default_exp_name(args.model, args.split, args.subset)
    if args.openai_base_url:
        os.environ["GOEDEL_OPENAI_BASE_URL"] = args.openai_base_url.rstrip("/")
        os.environ.setdefault("GOEDEL_OPENAI_API_KEY", "dummy")
    args.node_timeout_s = _optional_timeout(args.node_timeout_s)
    args.llm_api_timeout_s = _optional_timeout(args.llm_api_timeout_s)
    args.parallel_tool_calls = _optional_bool(args.parallel_tool_calls)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "phase1_concurrency",
        "phase2_blueprint_concurrency",
        "phase2_node_concurrency",
        "refine_concurrency",
        "blueprint_max_retries",
        "refine_max_retries",
        "node_max_tool_calls",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_refinement_iterations < 0:
        raise ValueError("max_refinement_iterations must be non-negative")
    if args.node_timeout_s is not None and args.node_timeout_s <= 0:
        raise ValueError("node_timeout_s must be positive or none/null/0")
    if args.llm_api_timeout_s is not None and args.llm_api_timeout_s <= 0:
        raise ValueError("llm_api_timeout_s must be positive or none/null/0")


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read RobustPABench parquet files.") from exc
    return pq.read_table(path).to_pylist()


def _split_from_path(path: Path) -> str:
    name = path.name
    return name.split("-", 1)[0] if "-" in name else path.stem


def _iter_parquets(data_root: Path, subsets: list[str] | None, splits: list[str] | None) -> list[tuple[str, str, Path]]:
    if not data_root.exists():
        raise FileNotFoundError(f"RobustPABench data root not found: {data_root}")
    subset_filter = set(subsets or [])
    split_filter = set(splits or [])
    out: list[tuple[str, str, Path]] = []
    for subset_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        subset = subset_dir.name
        if subset_filter and subset not in subset_filter:
            continue
        for parquet_path in sorted(subset_dir.glob("*.parquet")):
            split = _split_from_path(parquet_path)
            if split_filter and split not in split_filter:
                continue
            out.append((subset, split, parquet_path))
    return out


def _source_id(row: dict[str, Any], row_index: int) -> str:
    return str(row.get("name") or row.get("id") or f"row_{row_index}")


def _make_record(subset: str, split: str, parquet_path: Path, row_index: int, row: dict[str, Any]) -> Record:
    source_id = _source_id(row, row_index)
    record_id = safe_stem(source_id, prefix="robustpa_")
    safe_subset = safe_stem(subset)
    safe_split = safe_stem(split)
    theorem_name = safe_stem(f"robustpa_{safe_subset}_{safe_split}_{record_id}")
    unique_id = f"{subset}__{split}__{record_id}"
    informal_statement = str(row.get("informal_statement") or "")
    informal_proof = str(row.get("informal_proof") or "")
    if not informal_statement:
        raise ValueError(f"row has no informal_statement: {parquet_path}:{row_index}")
    return Record(
        unique_id=unique_id,
        record_id=record_id,
        source_id=source_id,
        subset=subset,
        split=split,
        parquet_path=parquet_path,
        row_index=row_index,
        theorem_name=theorem_name,
        informal_statement=informal_statement,
        informal_proof=informal_proof,
    )


def _select_records(args: argparse.Namespace) -> list[Record]:
    if args.limit is not None and args.limit <= 0:
        return []
    records: list[Record] = []
    for subset, split, parquet_path in _iter_parquets(args.data_root, args.subset, args.split):
        for row_index, row in enumerate(_read_parquet_rows(parquet_path), 1):
            record = _make_record(subset, split, parquet_path, row_index, row)
            if args.problem_id and args.problem_id not in {
                record.source_id,
                record.record_id,
                record.unique_id,
                record.theorem_name,
            }:
                continue
            records.append(record)
            if args.limit is not None and len(records) >= args.limit:
                return records
    return records


def _record_paths(output_root: Path, record: Record) -> tuple[Path, Path, Path]:
    base = Path(record.subset) / record.split / record.record_id
    checkpoint_path = output_root / "checkpoints" / record.subset / record.split / f"{record.record_id}.json"
    trace_path = output_root / "traces" / record.subset / record.split / f"{record.record_id}.jsonl"
    blueprint_dir = output_root / "blueprints" / base
    return checkpoint_path, trace_path, blueprint_dir


def _write_blueprint_snapshot(
    output_root: Path,
    record: Record,
    *,
    iteration: int,
    label: str,
    blueprint: Blueprint,
) -> Path:
    _checkpoint_path, _trace_path, blueprint_dir = _record_paths(output_root, record)
    blueprint_dir.mkdir(parents=True, exist_ok=True)
    path = blueprint_dir / f"round_{iteration:02d}_{label}.lean"
    path.write_text(blueprint.lean_file, encoding="utf-8")
    return path


def _node_rows(blueprint: Blueprint | None, state: CheckpointState | None) -> list[dict[str, Any]]:
    if blueprint is None:
        return []
    node_results = state.node_results if state else {}
    proved_cache = state.proved_cache if state else {}
    rows: list[dict[str, Any]] = []
    for node in blueprint.nodes:
        result = node_results.get(node.name, {})
        signal = result.get("signal")
        if not signal and node.name in proved_cache:
            signal = "solved"
        rows.append({
            "name": node.name,
            "kind": node.kind,
            "dependencies": list(node.dependencies),
            "signal": signal or "pending",
            "proof_body": result.get("proof_body", proved_cache.get(node.name, "")),
            "lean_errors": result.get("lean_errors", []),
            "analysis": result.get("analysis", ""),
            "suggested_fix": result.get("suggested_fix", ""),
        })
    return rows


def _score_state(blueprint: Blueprint | None, state: CheckpointState | None) -> dict[str, Any]:
    if blueprint is None:
        return {
            "root_theorem": "",
            "root_proved": False,
            "all_nodes_proved": False,
            "total_nodes": 0,
            "proved_node_count": 0,
            "proved_ratio": 0.0,
            "proved_nodes": [],
            "failed_nodes": [],
            "infra_error_nodes": [],
            "infra_error_node_count": 0,
        }
    node_names = {node.name for node in blueprint.nodes}
    proved = node_names & set(state.proved_cache if state else {})
    failed = node_names - proved
    total_nodes = len(node_names)
    infra_error_nodes = sorted(
        name for name, result in (state.node_results if state else {}).items()
        if result.get("signal") == "infra_error"
    )
    return {
        "root_theorem": blueprint.target_theorem,
        "root_proved": blueprint.target_theorem in proved,
        "all_nodes_proved": total_nodes > 0 and not failed,
        "total_nodes": total_nodes,
        "proved_node_count": len(proved),
        "proved_ratio": round(len(proved) / total_nodes, 4) if total_nodes else 0.0,
        "proved_nodes": sorted(proved),
        "failed_nodes": sorted(failed),
        "infra_error_nodes": infra_error_nodes,
        "infra_error_node_count": len(infra_error_nodes),
    }


def _append_round(
    rounds_path: Path,
    lock: asyncio.Lock,
    record: Record,
    *,
    phase: str,
    iteration: int,
    blueprint_path: Path | None,
    blueprint: Blueprint | None,
    state: CheckpointState | None,
):
    async def write() -> None:
        lean_text = blueprint.lean_file if blueprint else ""
        row = {
            "id": record.unique_id,
            "record_id": record.record_id,
            "source_id": record.source_id,
            "subset": record.subset,
            "split": record.split,
            "theorem_name": record.theorem_name,
            "iteration": iteration,
            "phase": phase,
            "blueprint_path": str(blueprint_path) if blueprint_path else "",
            "blueprint_hash": hashlib.sha256(lean_text.encode("utf-8")).hexdigest() if lean_text else "",
            "nodes": _node_rows(blueprint, state),
        }
        async with lock:
            append_jsonl(rounds_path, row)

    return write()


def _result_row(
    record: Record,
    output_root: Path,
    *,
    status: str,
    phase: str,
    blueprint: Blueprint | None,
    state: CheckpointState | None,
    runtime: LeanRuntime,
    error: str = "",
    traceback_text: str = "",
) -> dict[str, Any]:
    checkpoint_path, trace_path, blueprint_dir = _record_paths(output_root, record)
    score = _score_state(blueprint, state)
    row = {
        "id": record.unique_id,
        "record_id": record.record_id,
        "source_id": record.source_id,
        "subset": record.subset,
        "split": record.split,
        "row_index": record.row_index,
        "parquet_path": str(record.parquet_path),
        "theorem_name": record.theorem_name,
        "status": status,
        "phase": phase,
        "success": bool(score["root_proved"]),
        "terminal": True,
        "blueprint_success": bool(blueprint and blueprint.nodes and state),
        "iterations": state.iteration if state else 0,
        "checkpoint_path": str(checkpoint_path),
        "trace_path": str(trace_path),
        "blueprint_dir": str(blueprint_dir),
        "error": error,
        "lean_runtime": runtime.metadata,
        **score,
    }
    if traceback_text:
        row["traceback"] = traceback_text
    return row


def _existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        import json

        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def _clear_record_outputs(output_root: Path, record: Record) -> None:
    checkpoint_path, trace_path, blueprint_dir = _record_paths(output_root, record)
    unlink_if_exists(checkpoint_path)
    unlink_if_exists(trace_path)
    if blueprint_dir.exists():
        shutil.rmtree(blueprint_dir)


async def _run_record(
    record: Record,
    *,
    args: argparse.Namespace,
    output_root: Path,
    runtime: LeanRuntime,
    phase_executor: ThreadPoolExecutor,
    node_executor: ThreadPoolExecutor,
    phase1_sem: asyncio.Semaphore,
    phase2_sem: asyncio.Semaphore,
    node_sem: asyncio.Semaphore,
    refine_sem: asyncio.Semaphore,
    rounds_path: Path,
    rounds_lock: asyncio.Lock,
) -> dict[str, Any]:
    checkpoint_path, trace_path, _blueprint_dir = _record_paths(output_root, record)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    tracer = JsonlTracer(trace_path)
    loop = asyncio.get_running_loop()
    blueprint: Blueprint | None = None
    state: CheckpointState | None = None
    phase = "resume"
    try:
        state = CheckpointState.load_or_none(checkpoint_path) if args.resume else None
        if state is not None and state.theorem_stmt.strip() != record.informal_statement.strip():
            raise RuntimeError(f"checkpoint input mismatch: {checkpoint_path}")
        blueprint = state.get_blueprint() if state else None

        if blueprint is None:
            phase = "phase1"
            phase1_tracer = tracer.with_context(phase="phase1", iteration=0)
            async with phase1_sem:
                tqdm.write(f"[phase1-start] {record.unique_id}")
                blueprint = await loop.run_in_executor(
                    phase_executor,
                    partial(
                        generate_blueprint_from_informal,
                        informal_statement=record.informal_statement,
                        informal_proof=record.informal_proof,
                        target_name=record.theorem_name,
                        model=args.model,
                        compiler=runtime.compiler,
                        tracer=phase1_tracer,
                        thm_name=record.unique_id,
                        max_retries=args.blueprint_max_retries,
                    ),
                )
                state = CheckpointState(theorem_stmt=record.informal_statement, model=args.model)
                state.set_blueprint(blueprint)
                state.save(checkpoint_path)
                path = _write_blueprint_snapshot(
                    output_root, record, iteration=0, label="phase1", blueprint=blueprint
                )
                await _append_round(
                    rounds_path,
                    rounds_lock,
                    record,
                    phase="after_phase1",
                    iteration=0,
                    blueprint_path=path,
                    blueprint=blueprint,
                    state=state,
                )
                tqdm.write(f"[phase1-done] {record.unique_id} nodes={len(blueprint.nodes)}")
        else:
            tracer.emit(TraceEvent(
                kind="resume",
                thm_name=record.unique_id,
                phase="resume",
                iteration=state.iteration,
                args={"iteration": state.iteration, "proved_cache_count": len(state.proved_cache)},
                ok=True,
            ))

        while True:
            state = CheckpointState.load(checkpoint_path)
            blueprint = state.get_blueprint()
            if blueprint is None:
                raise RuntimeError(f"No blueprint in checkpoint {checkpoint_path}")
            if state.done:
                status = "solved" if state.success else "failed"
                return _result_row(
                    record,
                    output_root,
                    status=status,
                    phase="done",
                    blueprint=blueprint,
                    state=state,
                    runtime=runtime,
                )

            phase = "phase2"
            async with phase2_sem:
                tqdm.write(f"[phase2-start] {record.unique_id} iteration={state.iteration}")
                phase2_tracer = tracer.with_context(phase="phase2", iteration=state.iteration)
                orch_result = await run_phase2_async(
                    checkpoint_path=checkpoint_path,
                    compiler=runtime.compiler,
                    compiler_factory=runtime.compiler_factory,
                    retrieval=MathlibRetrieval(),
                    tracer=phase2_tracer,
                    node_timeout_s=args.node_timeout_s,
                    llm_api_timeout_s=args.llm_api_timeout_s,
                    model=args.model,
                    node_max_tool_calls=args.node_max_tool_calls,
                    parallel_tool_calls=args.parallel_tool_calls,
                    node_executor=node_executor,
                    node_semaphore=node_sem,
                )
                state = CheckpointState.load(checkpoint_path)
                blueprint = state.get_blueprint()
                path = _write_blueprint_snapshot(
                    output_root, record, iteration=state.iteration, label="phase2", blueprint=blueprint
                )
                await _append_round(
                    rounds_path,
                    rounds_lock,
                    record,
                    phase="after_phase2",
                    iteration=state.iteration,
                    blueprint_path=path,
                    blueprint=blueprint,
                    state=state,
                )
                tqdm.write(
                    f"[phase2-done] {record.unique_id} iteration={state.iteration} "
                    f"all_proved={orch_result.all_proved()}"
                )

            if state.done and state.success:
                return _result_row(
                    record,
                    output_root,
                    status="solved",
                    phase="phase2",
                    blueprint=blueprint,
                    state=state,
                    runtime=runtime,
                )

            if state.iteration >= args.max_refinement_iterations:
                state.done = True
                state.success = False
                state.save(checkpoint_path)
                return _result_row(
                    record,
                    output_root,
                    status="exhausted",
                    phase="phase2",
                    blueprint=blueprint,
                    state=state,
                    runtime=runtime,
                )

            phase = "phase3"
            async with refine_sem:
                tqdm.write(f"[phase3-start] {record.unique_id} iteration={state.iteration}")
                phase3_tracer = tracer.with_context(phase="phase3", iteration=state.iteration)
                await loop.run_in_executor(
                    phase_executor,
                    partial(
                        run_phase3,
                        checkpoint_path=checkpoint_path,
                        compiler=runtime.compiler,
                        model=args.model,
                        max_iterations=args.max_refinement_iterations,
                        tracer=phase3_tracer,
                        thm_name=record.unique_id,
                        refine_max_retries=args.refine_max_retries,
                    ),
                )
                state = CheckpointState.load(checkpoint_path)
                blueprint = state.get_blueprint()
                path = _write_blueprint_snapshot(
                    output_root, record, iteration=state.iteration, label="refined", blueprint=blueprint
                )
                await _append_round(
                    rounds_path,
                    rounds_lock,
                    record,
                    phase="after_phase3",
                    iteration=state.iteration,
                    blueprint_path=path,
                    blueprint=blueprint,
                    state=state,
                )
                tqdm.write(f"[phase3-done] {record.unique_id} next_iteration={state.iteration}")
    except Exception as exc:
        try:
            state = CheckpointState.load_or_none(checkpoint_path)
            blueprint = state.get_blueprint() if state else blueprint
        except Exception:
            pass
        return _result_row(
            record,
            output_root,
            status="error",
            phase=phase,
            blueprint=blueprint,
            state=state,
            runtime=runtime,
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )
    finally:
        tracer.close()


def _metric_row(scope: str, rows: list[dict[str, Any]], subset: str = "", split: str = "") -> dict[str, Any]:
    total = len(rows)
    root = sum(1 for row in rows if row.get("root_proved"))
    all_nodes = sum(1 for row in rows if row.get("all_nodes_proved"))
    blueprints = sum(1 for row in rows if row.get("blueprint_success"))
    return {
        "scope": scope,
        "subset": subset,
        "split": split,
        "total": total,
        "root_proved_count": root,
        "root_proved_acc": root / total if total else 0.0,
        "all_nodes_proved_count": all_nodes,
        "all_nodes_proved_acc": all_nodes / total if total else 0.0,
        "blueprint_success_count": blueprints,
        "blueprint_success_rate": blueprints / total if total else 0.0,
        "avg_iterations": sum(float(row.get("iterations") or 0) for row in rows) / total if total else 0.0,
        "avg_total_nodes": sum(float(row.get("total_nodes") or 0) for row in rows) / total if total else 0.0,
        "avg_proved_ratio": sum(float(row.get("proved_ratio") or 0) for row in rows) / total if total else 0.0,
        "phase1_failed": sum(1 for row in rows if row.get("status") == "error" and row.get("phase") == "phase1"),
        "refine_failed": sum(1 for row in rows if row.get("status") == "error" and row.get("phase") == "phase3"),
        "exhausted": sum(1 for row in rows if row.get("status") == "exhausted"),
        "infra_error": sum(1 for row in rows if int(row.get("infra_error_node_count") or 0) > 0),
    }


def _write_metrics(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = [_metric_row("global", rows)]

    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subset = str(row.get("subset") or "")
        split = str(row.get("split") or "")
        by_subset[subset].append(row)
        by_split[split].append(row)
        by_pair[(subset, split)].append(row)

    for subset, group in sorted(by_subset.items()):
        metric_rows.append(_metric_row("subset", group, subset=subset))
    for split, group in sorted(by_split.items()):
        metric_rows.append(_metric_row("split", group, split=split))
    for (subset, split), group in sorted(by_pair.items()):
        metric_rows.append(_metric_row("subset_split", group, subset=subset, split=split))

    metrics = {
        "primary_metric": "root_proved_acc",
        "groups": metric_rows,
    }
    write_json(output_root / "metrics.json", metrics)
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    return metrics


def _should_skip_existing(row: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    if not args.resume or row is None:
        return False
    if row.get("root_proved"):
        return True
    if row.get("terminal") and not args.rerun_failed:
        return True
    return False


async def _run_experiment(args: argparse.Namespace, output_root: Path, runtime: LeanRuntime) -> None:
    records = _select_records(args)
    results_path = output_root / "results.jsonl"
    rounds_path = output_root / "rounds.jsonl"
    if not args.resume:
        for path in (results_path, rounds_path, output_root / "metrics.json", output_root / "metrics.csv"):
            unlink_if_exists(path)

    existing = _existing_results(results_path)
    pending: list[Record] = []
    skipped = 0
    for record in records:
        row = existing.get(record.unique_id)
        if _should_skip_existing(row, args):
            skipped += 1
            continue
        if not args.resume:
            _clear_record_outputs(output_root, record)
        if args.rerun_failed and row and not row.get("root_proved"):
            _clear_record_outputs(output_root, record)
        pending.append(record)

    print(f"[select] records={len(records)} pending={len(pending)} skipped={skipped} output={output_root}", flush=True)
    print(
        "[concurrency] "
        f"phase1={args.phase1_concurrency} "
        f"phase2_blueprint={args.phase2_blueprint_concurrency} "
        f"phase2_node={args.phase2_node_concurrency} "
        f"refine={args.refine_concurrency} "
        f"lean_check={args.lean_check_concurrency} "
        f"node_timeout_s={args.node_timeout_s} "
        f"llm_api_timeout_s={args.llm_api_timeout_s} "
        f"parallel_tool_calls={args.parallel_tool_calls}",
        flush=True,
    )

    rounds_lock = asyncio.Lock()
    results_lock = asyncio.Lock()
    phase1_sem = asyncio.Semaphore(args.phase1_concurrency)
    phase2_sem = asyncio.Semaphore(args.phase2_blueprint_concurrency)
    node_sem = asyncio.Semaphore(args.phase2_node_concurrency)
    refine_sem = asyncio.Semaphore(args.refine_concurrency)
    phase_workers = max(args.phase1_concurrency, args.refine_concurrency)
    with (
        ThreadPoolExecutor(max_workers=phase_workers) as phase_executor,
        ThreadPoolExecutor(max_workers=args.phase2_node_concurrency) as node_executor,
    ):
        async def run_one(record: Record) -> None:
            row = await _run_record(
                record,
                args=args,
                output_root=output_root,
                runtime=runtime,
                phase_executor=phase_executor,
                node_executor=node_executor,
                phase1_sem=phase1_sem,
                phase2_sem=phase2_sem,
                node_sem=node_sem,
                refine_sem=refine_sem,
                rounds_path=rounds_path,
                rounds_lock=rounds_lock,
            )
            async with results_lock:
                append_jsonl(results_path, row)
                existing[row["id"]] = row
            tqdm.write(f"[record-{row['status']}] {record.unique_id} root_proved={row.get('root_proved')}")

        tasks = [asyncio.create_task(run_one(record)) for record in pending]
        with tqdm(total=len(tasks), desc="robustpa", unit="record") as progress:
            for task in asyncio.as_completed(tasks):
                await task
                progress.update(1)

    selected_ids = {record.unique_id for record in records}
    final_rows = [row for row_id, row in existing.items() if row_id in selected_ids]
    metrics = _write_metrics(output_root, final_rows)
    print(f"[done] results={results_path}", flush=True)
    print(f"[rounds] {rounds_path}", flush=True)
    print(f"[metrics] primary root_proved_acc={metrics['groups'][0]['root_proved_acc']}", flush=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_base / args.exp_name
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = make_lean_runtime(args)
    try:
        prepare_lean_runtime_metadata(output_root, resume=args.resume, metadata=runtime.metadata)
        start = time.perf_counter()
        asyncio.run(_run_experiment(args, output_root, runtime))
        print(f"[runtime] duration_s={time.perf_counter() - start:.3f}", flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
