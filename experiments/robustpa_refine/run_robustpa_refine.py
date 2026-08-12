from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from blueprint import (  # noqa: E402
    Blueprint,
    BlueprintGenerationError,
    _parse_blueprint,
)
from blueprint_generation import generate_blueprint  # noqa: E402
from blueprint_review_viewer.review_schema import index_entry, write_review_artifact  # noqa: E402
from checkpoint import CheckpointState, RunStatus  # noqa: E402
from robustpa_refine.io_utils import append_jsonl, safe_stem, unlink_if_exists, write_json  # noqa: E402
from robustpa_refine.runtime import (  # noqa: E402
    LeanRuntime,
    make_lean_runtime,
    write_lean_runtime_metadata,
)
from mathlib_retrieval import MathlibRetrieval  # noqa: E402
from orchestrator import active_node_names  # noqa: E402
from pipeline import run_phase2_async, run_phase3  # noqa: E402
from semantic_fidelity import snapshot_blueprint_semantics  # noqa: E402
from tracer import JsonlTracer, TraceEvent  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT.parent / "czx_work" / "RobustPABench"
DEFAULT_OUTPUT_BASE = REPO_ROOT.parent / "czx_work" / "robustpa_refine"
DEFAULT_MODEL = "deepseek-v4-flash"
RUNTIME_HISTORY_FILENAME = "runtime_history.json"


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


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("expected a positive integer or null")
    text = str(value).strip().lower()
    if text in {"", "none", "null", "default", "omit"}:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or null") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or null")
    return parsed


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
    claimed_answer: str


def _apply_environment(config: dict[str, Any]) -> None:
    env = config.get("environment") or {}
    if not isinstance(env, dict):
        raise ValueError("Config key 'environment' must be an object.")
    for key, value in env.items():
        if value is not None:
            os.environ.setdefault(str(key), str(value))


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value] if value else None
    return [str(item) for item in value]


def _resolve_path(value: Any, original_cwd: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else original_cwd / path


def parse_args(cfg: DictConfig) -> argparse.Namespace:
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Hydra config must be a mapping.")
    _apply_environment(config)

    original_cwd = Path(get_original_cwd())
    config.pop("environment", None)
    args = argparse.Namespace(**config)
    args.data_root = _resolve_path(getattr(args, "data_root", DEFAULT_DATA_ROOT), original_cwd)
    args.output_base = _resolve_path(getattr(args, "output_base", DEFAULT_OUTPUT_BASE), original_cwd)
    args.subset = _as_list(getattr(args, "subset", None))
    args.split = _as_list(getattr(args, "split", None))
    args.limit = getattr(args, "limit", None)
    args.problem_id = getattr(args, "problem_id", None)
    include_path = getattr(args, "include_source_ids_path", None)
    args.include_source_ids_path = (
        _resolve_path(include_path, original_cwd) if include_path else None
    )
    args.include_source_ids = None
    if args.include_source_ids_path is not None:
        payload = json.loads(args.include_source_ids_path.read_text(encoding="utf-8"))
        values = payload.get("include_ids") if isinstance(payload, dict) else payload
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ValueError("include_source_ids_path must contain a JSON string list")
        args.include_source_ids = set(values)
    args.resume = bool(getattr(args, "resume", False))
    for name in (
        "phase1_seed_root", "phase1_source_results_path",
        "phase1_source_experiment_root",
    ):
        value = getattr(args, name, None)
        setattr(args, name, _resolve_path(value, original_cwd) if value else None)
    args.retry_error_results = bool(getattr(args, "retry_error_results", False))
    if not args.exp_name:
        args.exp_name = default_exp_name(args.model, args.split, args.subset)
    if args.openai_base_url:
        os.environ["GOEDEL_OPENAI_BASE_URL"] = args.openai_base_url.rstrip("/")
        if not (os.environ.get("GOEDEL_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            os.environ["GOEDEL_OPENAI_API_KEY"] = "dummy"
    args.node_timeout_s = _optional_timeout(args.node_timeout_s)
    args.llm_api_timeout_s = _optional_timeout(args.llm_api_timeout_s)
    args.node_max_prove_turns = _optional_positive_int(args.node_max_prove_turns)
    args.node_max_negation_probe_turns = int(args.node_max_negation_probe_turns)
    args.critical_negation_max_turns = int(args.critical_negation_max_turns)
    args.max_tool_calls_per_turn = int(args.max_tool_calls_per_turn)
    args.generation_max_turns = int(args.generation_max_turns)
    args.generation_prompt_profile = str(args.generation_prompt_profile)
    args.generation_enable_thinking = bool(args.generation_enable_thinking)
    for name in (
        "generation_temperature", "generation_top_p", "generation_min_p",
        "generation_presence_penalty", "generation_repetition_penalty",
    ):
        setattr(args, name, float(getattr(args, name)))
    args.generation_top_k = int(args.generation_top_k)
    args.generation_model_max_context = int(args.generation_model_max_context)
    args.generation_context_safety_margin = int(args.generation_context_safety_margin)
    args.tokenizer_path = os.environ.get(
        "GOEDEL_TOKENIZER_PATH", str(getattr(args, "tokenizer_path", ""))
    ).strip()
    args.formal_decompiler_max_tokens = int(
        args.formal_decompiler_max_tokens
    )
    args.strict_comparator_max_tokens = int(
        args.strict_comparator_max_tokens
    )
    args.semantic_format_max_attempts = int(
        args.semantic_format_max_attempts
    )
    args.semantic_audit_enable_thinking = bool(
        args.semantic_audit_enable_thinking
    )
    for name in (
        "semantic_audit_temperature", "semantic_audit_top_p",
        "semantic_audit_min_p", "semantic_audit_presence_penalty",
        "semantic_audit_repetition_penalty",
    ):
        setattr(args, name, float(getattr(args, name)))
    args.semantic_audit_top_k = int(args.semantic_audit_top_k)
    args.refinement_max_retries = int(args.refinement_max_retries)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.execution_mode not in {"full", "phase1_only", "phase2_only"}:
        raise ValueError(
            "execution_mode must be one of: full, phase1_only, phase2_only"
        )
    if args.generation_prompt_profile not in {"whole_cot_minimal", "standard"}:
        raise ValueError("generation_prompt_profile must be whole_cot_minimal or standard")
    for name in (
        "phase1_concurrency",
        "phase2_blueprint_concurrency",
        "phase2_node_concurrency",
        "refine_concurrency",
        "phase2_contract_check_concurrency",
        "lean_max_inflight_snippets",
        "lean_batch_size",
        "lean_parallel_batches",
        "refinement_max_retries",
        "node_max_prove_turns",
        "generation_max_turns",
        "formal_decompiler_max_tokens",
        "strict_comparator_max_tokens",
        "semantic_format_max_attempts",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_refinement_iterations < 0:
        raise ValueError("max_refinement_iterations must be non-negative")
    if args.execution_mode == "phase2_only":
        if args.max_refinement_iterations != 0:
            raise ValueError(
                "phase2_only requires max_refinement_iterations=0"
            )
        for name in (
            "phase1_seed_root", "phase1_source_results_path",
            "phase1_source_experiment_root",
        ):
            if getattr(args, name) is None:
                raise ValueError(f"phase2_only requires {name}")
    if args.node_max_negation_probe_turns < 0:
        raise ValueError("node_max_negation_probe_turns must be non-negative")
    if args.critical_negation_max_turns < 0:
        raise ValueError("critical_negation_max_turns must be non-negative")
    if args.generation_model_max_context <= 0:
        raise ValueError("generation_model_max_context must be positive")
    if args.generation_context_safety_margin < 0:
        raise ValueError("generation_context_safety_margin must be non-negative")
    if not args.tokenizer_path:
        raise ValueError(
            "tokenizer_path is required for dynamic generation budgeting"
        )
    if args.proof_policy not in {"full", "first_failed_wave", "critical_path"}:
        raise ValueError("proof_policy must be one of: full, first_failed_wave, critical_path")
    if args.node_timeout_s is not None and args.node_timeout_s <= 0:
        raise ValueError("node_timeout_s must be positive or none/null/0")
    if args.llm_api_timeout_s is not None and args.llm_api_timeout_s <= 0:
        raise ValueError("llm_api_timeout_s must be positive or none/null/0")
    if args.max_tool_calls_per_turn <= 0:
        raise ValueError("max_tool_calls_per_turn must be a positive integer")
    if float(args.lean_batch_wait_ms) < 0:
        raise ValueError("lean_batch_wait_ms must be non-negative")


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
    claimed_answer = str(row.get("claimed_answer") or "")
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
        claimed_answer=claimed_answer,
    )


def _select_records(args: argparse.Namespace) -> list[Record]:
    if args.limit is not None and args.limit <= 0:
        return []
    records: list[Record] = []
    for subset, split, parquet_path in _iter_parquets(args.data_root, args.subset, args.split):
        for row_index, row in enumerate(_read_parquet_rows(parquet_path), 1):
            record = _make_record(subset, split, parquet_path, row_index, row)
            if args.include_source_ids is not None and record.source_id not in args.include_source_ids:
                continue
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


def _write_phase1_candidates(
    blueprint_dir: Path,
    candidates: list[str],
    labels: list[str] | None = None,
) -> list[str]:
    """Persist complete Blueprint candidates from each generation round."""
    blueprint_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, candidate in enumerate(candidates):
        label = labels[index] if labels and index < len(labels) else f"phase1_iter{index}"
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        path = blueprint_dir / f"{safe_label}.lean"
        path.write_text(candidate.rstrip() + "\n", encoding="utf-8")
        paths.append(str(path))
    return paths


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
            "title": node.title,
            "source_step_id": node.source_step_id,
            "dependencies": list(node.dependencies),
            "signal": signal or "pending",
            "proof_body": result.get("proof_body", proved_cache.get(node.name, "")),
            "lean_errors": result.get("lean_errors", []),
        })
    return rows


def _score_state(blueprint: Blueprint | None, state: CheckpointState | None) -> dict[str, Any]:
    if blueprint is None:
        return {
            "root_theorem": "",
            "root_proved": False,
            "total_nodes": 0,
            "proved_node_count": 0,
            "proved_ratio": 0.0,
            "proved_nodes": [],
            "failed_nodes": [],
            "infra_error_nodes": [],
            "infra_error_node_count": 0,
        }
    node_names = active_node_names(blueprint)
    definition_names = {
        node.name for node in blueprint.nodes
        if node.kind == "definition" and node.name in node_names
    }
    proved = definition_names | (node_names & set(state.proved_cache if state else {}))
    failed = node_names - proved
    total_nodes = len(node_names)
    infra_error_nodes = sorted(
        name for name, result in (state.node_results if state else {}).items()
        if result.get("signal") == "infra_error"
    )
    return {
        "root_theorem": blueprint.target_theorem,
        "root_proved": bool(state and state.root_proved),
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
    args: argparse.Namespace,
    error: str = "",
    traceback_text: str = "",
) -> dict[str, Any]:
    checkpoint_path, trace_path, blueprint_dir = _record_paths(output_root, record)
    score = _score_state(blueprint, state)
    semantic_gate_results = list(state.semantic_gate_results) if state else []
    semantic_warnings = [
        issue
        for gate in semantic_gate_results
        if isinstance(gate, dict) and gate.get("passed") is True
        for issue in (gate.get("issues") or [])
        if isinstance(issue, dict) and issue.get("severity") == "warning"
        and issue.get("code")
    ]
    semantic_warning_codes = sorted({str(issue["code"]) for issue in semantic_warnings})
    row = {
        "id": record.unique_id,
        "record_id": record.record_id,
        "source_id": record.source_id,
        "subset": record.subset,
        "split": record.split,
        "row_index": record.row_index,
        "parquet_path": str(record.parquet_path),
        "theorem_name": record.theorem_name,
        "claimed_answer": record.claimed_answer,
        "generation_prompt_profile": args.generation_prompt_profile,
        "status": status,
        "phase": phase,
        "success": bool(score["root_proved"]),
        "iterations": state.iteration if state else 0,
        "checkpoint_path": str(checkpoint_path),
        "trace_path": str(trace_path),
        "blueprint_dir": str(blueprint_dir),
        "error": error,
        "lean_runtime": runtime.metadata,
        "node_max_prove_turns": args.node_max_prove_turns,
        "negation_probe_turns": args.node_max_negation_probe_turns,
        "max_tool_calls_per_turn": args.max_tool_calls_per_turn,
        "proof_policy": args.proof_policy,
        "critical_negation_max_turns": args.critical_negation_max_turns,
        "generation_max_turns": args.generation_max_turns,
        "generation_sampling": {
            "enable_thinking": args.generation_enable_thinking,
            "temperature": args.generation_temperature,
            "top_p": args.generation_top_p,
            "top_k": args.generation_top_k,
            "min_p": args.generation_min_p,
            "presence_penalty": args.generation_presence_penalty,
            "repetition_penalty": args.generation_repetition_penalty,
        },
        "formal_decompiler_max_tokens": args.formal_decompiler_max_tokens,
        "strict_comparator_max_tokens": args.strict_comparator_max_tokens,
        "semantic_format_max_attempts": args.semantic_format_max_attempts,
        "semantic_audit_sampling": {
            "enable_thinking": args.semantic_audit_enable_thinking,
            "temperature": args.semantic_audit_temperature,
            "top_p": args.semantic_audit_top_p,
            "top_k": args.semantic_audit_top_k,
            "min_p": args.semantic_audit_min_p,
            "presence_penalty": args.semantic_audit_presence_penalty,
            "repetition_penalty": args.semantic_audit_repetition_penalty,
        },
        "semantic_status": state.semantic_status if state else "",
        "phase1_seed_hash": state.phase1_seed_hash if state else "",
        "phase1_blueprint_hash": state.phase1_blueprint_hash if state else "",
        "phase1_source_root": state.phase1_source_root if state else "",
        "phase2_resumed": state.phase2_last_launch_resumed if state else False,
        "phase2_resume_count": state.phase2_resume_count if state else 0,
        "semantic_gate_results": semantic_gate_results,
        "semantic_warning_codes": semantic_warning_codes,
        "semantic_warning_count": len(semantic_warnings),
        "generation_validation": dict(
            blueprint.generation_validation if blueprint is not None else {}
        ),
        "generation_history": list(
            blueprint.generation_history if blueprint is not None else []
        ),
        **score,
    }
    if traceback_text:
        row["traceback"] = traceback_text
    return row


def _accepted_phase1_status(blueprint: Blueprint) -> str:
    audit = blueprint.generation_validation.get("semanticAudit")
    if isinstance(audit, dict):
        classification = str(audit.get("classification") or "")
        if classification in {
            "strictAccepted", "acceptedWithWarnings",
        }:
            return classification
    raise RuntimeError("validated Blueprint is missing a terminal generation classification")


def _failed_phase1_status(exc: Exception) -> str:
    if not isinstance(exc, BlueprintGenerationError):
        return "infraError"
    if exc.failure_stage == "phase1Semantic":
        return "semanticRejected"
    if exc.failure_stage == "phase1SemanticAuditError":
        return "semanticAuditError"
    if exc.failure_stage in {
        "phase1Deterministic", "phase1DeterministicAndSemantic",
        "phase1ContextBudgetExceeded",
    }:
        return "structuralRejected"
    audit = exc.validation_details.get("semanticAudit")
    if isinstance(audit, dict):
        comparator = audit.get("strictComparator") or {}
        if (
            comparator.get("passed") is False
        ):
            return "semanticRejected"
    return "structuralRejected"


def _existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def _remove_jsonl_rows(path: Path, record_ids: set[str]) -> None:
    if not path.exists() or not record_ids:
        return
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".tmp_{path.name}_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination, path.open(
            "r", encoding="utf-8"
        ) as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("id")) in record_ids:
                    continue
                destination.write(line if line.endswith("\n") else f"{line}\n")
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _clear_record_outputs(output_root: Path, record: Record) -> None:
    checkpoint_path, trace_path, blueprint_dir = _record_paths(output_root, record)
    unlink_if_exists(checkpoint_path)
    unlink_if_exists(trace_path)
    if blueprint_dir.exists():
        shutil.rmtree(blueprint_dir)


def _phase1_seed_checkpoint_path(args: argparse.Namespace, record: Record) -> Path:
    assert args.phase1_seed_root is not None
    relative = Path(record.subset) / record.split / f"{record.record_id}.json"
    immutable = args.phase1_seed_root / "phase1_seeds" / "checkpoints" / relative
    return immutable if immutable.is_file() else _record_paths(args.phase1_seed_root, record)[0]


def _phase1_seed_snapshot_path(output_root: Path, record: Record) -> Path:
    relative = Path(record.subset) / record.split / f"{record.record_id}.json"
    return output_root / "phase1_seeds" / "checkpoints" / relative


def _reset_to_pristine_phase1(
    state: CheckpointState, blueprint: Blueprint,
) -> CheckpointState:
    state.status = RunStatus.RUNNING
    state.iteration = 0
    state.proved_cache = {}
    state.proof_cache_keys = {}
    state.node_results = {}
    state.refinement_history = []
    state.final_lean_file = ""
    state.final_lean_errors = []
    state.phase1_seed_hash = ""
    state.phase1_blueprint_hash = ""
    state.phase1_source_root = ""
    state.phase2_run_initialized = False
    state.phase2_last_launch_resumed = False
    state.phase2_resume_count = 0
    state.set_blueprint(blueprint)
    return state


def _phase1_seed_provenance(
    args: argparse.Namespace, record: Record,
) -> tuple[Path, str, str, CheckpointState, Blueprint]:
    seed_path = _phase1_seed_checkpoint_path(args, record)
    if not seed_path.is_file():
        raise FileNotFoundError(f"Phase 1 seed checkpoint is missing: {seed_path}")
    seed_state = CheckpointState.load(seed_path)
    blueprint = seed_state.get_blueprint()
    if blueprint is None:
        raise RuntimeError(f"Phase 1 seed has no Blueprint: {seed_path}")
    immutable_seed = "phase1_seeds" in seed_path.parts
    if not immutable_seed:
        source_blueprint_dir = _record_paths(args.phase1_seed_root, record)[2]
        phase1_snapshot = source_blueprint_dir / "round_00_phase1.lean"
        if not phase1_snapshot.is_file():
            raise RuntimeError(
                "Legacy Phase 1 checkpoint contains no immutable seed and its "
                f"round_00_phase1.lean is missing: {phase1_snapshot}"
            )
        blueprint = _parse_blueprint(
            phase1_snapshot.read_text(encoding="utf-8"), record.theorem_name,
        )
        seed_state = _reset_to_pristine_phase1(seed_state, blueprint)
    blueprint_hash = hashlib.sha256(blueprint.lean_file.encode("utf-8")).hexdigest()
    seed_payload = json.dumps(
        asdict(seed_state), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    seed_hash = hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()
    return seed_path, seed_hash, blueprint_hash, seed_state, blueprint


def _validate_phase1_seed(
    state: CheckpointState, record: Record, seed_path: Path,
) -> Blueprint:
    blueprint = state.get_blueprint()
    mismatches: list[str] = []
    if blueprint is None:
        mismatches.append("blueprint")
    if state.status != RunStatus.RUNNING:
        mismatches.append(f"status={state.status.value}")
    if state.iteration != 0:
        mismatches.append(f"iteration={state.iteration}")
    if state.node_results or state.proved_cache or state.proof_cache_keys:
        mismatches.append("phase2MutableState")
    if state.informal_statement != record.informal_statement:
        mismatches.append("informal_statement")
    if state.informal_proof != record.informal_proof:
        mismatches.append("informal_proof")
    if state.claimed_answer != record.claimed_answer:
        mismatches.append("claimed_answer")
    if blueprint is not None and blueprint.target_theorem != record.theorem_name:
        mismatches.append("target_theorem")
    if state.semantic_status not in {"strictAccepted", "acceptedWithWarnings"}:
        mismatches.append(f"semantic_status={state.semantic_status}")
    if mismatches:
        raise RuntimeError(
            f"Invalid immutable Phase 1 seed {seed_path}: " + ", ".join(mismatches)
        )
    assert blueprint is not None
    return blueprint


def _load_or_initialize_phase2_checkpoint(
    args: argparse.Namespace,
    record: Record,
    output_checkpoint_path: Path,
) -> tuple[CheckpointState, Blueprint, bool, str, str, Path]:
    (
        seed_path, seed_hash, blueprint_hash, pristine_state, pristine_blueprint,
    ) = _phase1_seed_provenance(args, record)
    source_root = str(args.phase1_source_experiment_root.resolve())
    if output_checkpoint_path.exists():
        state = CheckpointState.load(output_checkpoint_path)
        blueprint = state.get_blueprint()
        if blueprint is None:
            raise RuntimeError(f"Phase 2 checkpoint has no Blueprint: {output_checkpoint_path}")
        mismatches = []
        if state.phase1_seed_hash != seed_hash:
            mismatches.append("phase1_seed_hash")
        if state.phase1_blueprint_hash != blueprint_hash:
            mismatches.append("phase1_blueprint_hash")
        if state.phase1_source_root != source_root:
            mismatches.append("phase1_source_root")
        if not state.phase2_run_initialized:
            mismatches.append("phase2_run_initialized")
        if mismatches:
            raise RuntimeError(
                "Phase 2 checkpoint provenance does not match configured Phase 1 seed: "
                + ", ".join(mismatches)
            )
        state.phase2_last_launch_resumed = True
        state.phase2_resume_count += 1
        state.save(output_checkpoint_path)
        return state, blueprint, True, seed_hash, blueprint_hash, seed_path

    state = pristine_state
    blueprint = _validate_phase1_seed(state, record, seed_path)
    state.phase1_seed_hash = seed_hash
    state.phase1_blueprint_hash = blueprint_hash
    state.phase1_source_root = source_root
    state.phase2_run_initialized = True
    state.phase2_last_launch_resumed = False
    state.phase2_resume_count = 0
    state.save(output_checkpoint_path)
    return state, blueprint, False, seed_hash, blueprint_hash, seed_path


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
    checkpoint_path, trace_path, blueprint_dir = _record_paths(output_root, record)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    tracer = JsonlTracer(trace_path)
    loop = asyncio.get_running_loop()
    blueprint: Blueprint | None = None
    state: CheckpointState | None = None
    phase = "phase1"
    try:
        if args.execution_mode == "phase2_only":
            phase = "phase2"
            (
                state, blueprint, resumed, seed_hash, blueprint_hash, seed_path,
            ) = _load_or_initialize_phase2_checkpoint(args, record, checkpoint_path)
            _write_blueprint_snapshot(
                output_root, record, iteration=0, label="phase1_seed", blueprint=blueprint,
            )
            tracer.emit(TraceEvent(
                kind="phase2SeedLoaded",
                thm_name=record.unique_id,
                args={
                    "sourceCheckpoint": str(seed_path),
                    "sourceSeedHash": seed_hash,
                    "sourceBlueprintHash": blueprint_hash,
                    "outputCheckpoint": str(checkpoint_path),
                    "resumed": resumed,
                    "nodeCount": len(blueprint.nodes),
                },
                ok=True,
            ))
            tqdm.write(
                f"[phase2-{'resume' if resumed else 'seed'}] {record.unique_id} "
                f"nodes={len(blueprint.nodes)} iteration={state.iteration}"
            )
        else:
            phase1_tracer = tracer.with_context(phase="phase1", iteration=0)
            async with phase1_sem:
                tqdm.write(f"[phase1-start] {record.unique_id}")
                blueprint = await loop.run_in_executor(
                    phase_executor,
                    partial(
                        generate_blueprint,
                        informal_statement=record.informal_statement,
                        informal_proof=record.informal_proof,
                        claimed_answer=record.claimed_answer,
                        target_name=record.theorem_name,
                        model=args.model,
                        compiler=runtime.compiler,
                        tracer=phase1_tracer,
                        thm_name=record.unique_id,
                        max_turns=args.generation_max_turns,
                        tokenizer_path=args.tokenizer_path,
                        model_max_context=args.generation_model_max_context,
                        context_safety_margin=args.generation_context_safety_margin,
                        enable_thinking=args.generation_enable_thinking,
                        temperature=args.generation_temperature,
                        top_p=args.generation_top_p,
                        top_k=args.generation_top_k,
                        min_p=args.generation_min_p,
                        presence_penalty=args.generation_presence_penalty,
                        repetition_penalty=args.generation_repetition_penalty,
                        standalone_concurrency=args.phase2_contract_check_concurrency,
                        decompiler_max_tokens=args.formal_decompiler_max_tokens,
                        comparator_max_tokens=args.strict_comparator_max_tokens,
                        semantic_format_attempts=args.semantic_format_max_attempts,
                        semantic_audit_enable_thinking=args.semantic_audit_enable_thinking,
                        semantic_audit_temperature=args.semantic_audit_temperature,
                        semantic_audit_top_p=args.semantic_audit_top_p,
                        semantic_audit_top_k=args.semantic_audit_top_k,
                        semantic_audit_min_p=args.semantic_audit_min_p,
                        semantic_audit_presence_penalty=args.semantic_audit_presence_penalty,
                        semantic_audit_repetition_penalty=args.semantic_audit_repetition_penalty,
                        prompt_profile=args.generation_prompt_profile,
                    ),
                )
                _write_phase1_candidates(
                    blueprint_dir, blueprint.candidate_history, blueprint.candidate_labels,
                )
                state = CheckpointState(
                    informal_statement=record.informal_statement,
                    informal_proof=record.informal_proof,
                    claimed_answer=record.claimed_answer,
                    model=args.model,
                    semantic_fidelity_enabled=True,
                    semantic_static_gate=False,
                    semantic_minimal_ir=False,
                    semantic_freeze_refinement=False,
                )
                state.semantic_gate_results = list(blueprint.semantic_gate_results)
                state.semantic_status = _accepted_phase1_status(blueprint)
                state.semantic_contract_snapshot = asdict(
                    snapshot_blueprint_semantics(blueprint),
                )
                state.set_blueprint(blueprint)
                state.save(checkpoint_path)
                # Preserve an immutable Phase 1 seed before any Phase 2 node
                # result can mutate the working checkpoint.
                state.save(_phase1_seed_snapshot_path(output_root, record))
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

        if args.execution_mode == "phase1_only":
            return _result_row(
                record,
                output_root,
                status=_accepted_phase1_status(blueprint),
                phase="phase1",
                blueprint=blueprint,
                state=state,
                runtime=runtime,
                args=args,
            )

        while True:
            state = CheckpointState.load(checkpoint_path)
            blueprint = state.get_blueprint()
            if blueprint is None:
                raise RuntimeError(f"No blueprint in checkpoint {checkpoint_path}")
            if state.status != RunStatus.RUNNING:
                status = state.status.value
                return _result_row(
                    record,
                    output_root,
                    status=status,
                    phase="terminal",
                    blueprint=blueprint,
                    state=state,
                    runtime=runtime,
                    args=args,
                )

            phase = "phase2"
            async with phase2_sem:
                tqdm.write(f"[phase2-start] {record.unique_id} iteration={state.iteration}")
                phase2_tracer = tracer.with_context(phase="phase2", iteration=state.iteration)
                orch_result = await run_phase2_async(
                    checkpoint_path=checkpoint_path,
                    compiler=runtime.compiler,
                    retrieval=MathlibRetrieval(),
                    tracer=phase2_tracer,
                    node_timeout_s=args.node_timeout_s,
                    llm_api_timeout_s=args.llm_api_timeout_s,
                    model=args.model,
                    node_max_prove_turns=args.node_max_prove_turns,
                    node_max_negation_probe_turns=args.node_max_negation_probe_turns,
                    max_tool_calls_per_turn=args.max_tool_calls_per_turn,
                    proof_policy=args.proof_policy,
                    critical_negation_max_turns=args.critical_negation_max_turns,
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
                    f"root_solved={orch_result.root_solved} "
                    f"status={state.status.value}"
                )

            if state.status != RunStatus.RUNNING:
                return _result_row(
                    record,
                    output_root,
                    status=state.status.value,
                    phase="phase2",
                    blueprint=blueprint,
                    state=state,
                    runtime=runtime,
                    args=args,
                )

            if state.iteration >= args.max_refinement_iterations:
                state.status = RunStatus.EXHAUSTED
                state.save(checkpoint_path)
                return _result_row(
                    record,
                    output_root,
                    status="exhausted",
                    phase="phase2",
                    blueprint=blueprint,
                    state=state,
                    runtime=runtime,
                    args=args,
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
                        blueprint_max_retries=args.refinement_max_retries,
                        phase2_contract_check_concurrency=args.phase2_contract_check_concurrency,
                        semantic_fidelity_enabled=True,
                        semantic_static_gate=False,
                        semantic_freeze_refinement=False,
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
        row = _result_row(
            record,
            output_root,
            status=(
                _failed_phase1_status(exc)
                if phase == "phase1"
                else "infraError"
            ),
            phase=phase,
            blueprint=blueprint,
            state=state,
            runtime=runtime,
            args=args,
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )
        if isinstance(exc, BlueprintGenerationError) and exc.last_candidate.strip():
            blueprint_dir.mkdir(parents=True, exist_ok=True)
            candidate_paths = _write_phase1_candidates(
                blueprint_dir,
                exc.candidate_history or [exc.last_candidate],
                exc.candidate_labels,
            )
            candidate_path = blueprint_dir / "phase1_failed_last.lean"
            diagnostics_path = blueprint_dir / "phase1_failed_last.json"
            candidate_path.write_text(exc.last_candidate.rstrip() + "\n", encoding="utf-8")
            write_json(diagnostics_path, {
                "attempt": exc.attempt,
                "failure_stage": exc.failure_stage,
                "finish_reason": exc.finish_reason,
                "diagnostics": exc.diagnostics,
                "validation": exc.validation_details,
                "generation_history": exc.generation_history,
            })
            row.update({
                "failed_blueprint_candidate_path": str(candidate_path),
                "failed_blueprint_diagnostics_path": str(diagnostics_path),
                "failed_blueprint_failure_stage": exc.failure_stage,
                "phase1_candidate_paths": candidate_paths,
                "generation_validation": exc.validation_details,
                "generation_history": exc.generation_history,
            })
        return row
    finally:
        tracer.close()


def _metric_row(scope: str, rows: list[dict[str, Any]], subset: str = "", split: str = "") -> dict[str, Any]:
    total = len(rows)
    root = sum(1 for row in rows if row.get("root_proved"))
    accepted_statuses = {
        "strictAccepted", "acceptedWithWarnings",
    }
    accepted_rows = [row for row in rows if row.get("status") in accepted_statuses]
    return {
        "scope": scope,
        "subset": subset,
        "split": split,
        "total": total,
        "root_proved_count": root,
        "root_proved_acc": root / total if total else 0.0,
        "avg_iterations": sum(float(row.get("iterations") or 0) for row in rows) / total if total else 0.0,
        "avg_total_nodes": sum(float(row.get("total_nodes") or 0) for row in rows) / total if total else 0.0,
        "avg_proved_ratio": sum(float(row.get("proved_ratio") or 0) for row in rows) / total if total else 0.0,
        "phase1_failed": sum(
            1 for row in rows
            if row.get("status") in {"error", "semanticRejected", "semanticAuditError", "structuralRejected", "infraError"}
            and row.get("phase") == "phase1"
        ),
        "phase1_ineligible": sum(
            1 for row in rows if row.get("status") == "phase1Ineligible"
        ),
        "phase2_eligible": sum(
            1 for row in rows if row.get("status") != "phase1Ineligible"
        ),
        "blueprint_accepted": len(accepted_rows),
        "strict_accepted": sum(1 for row in rows if row.get("status") == "strictAccepted"),
        "accepted_with_warnings": sum(
            1 for row in rows if row.get("status") == "acceptedWithWarnings"
        ),
        "semantic_rejected": sum(1 for row in rows if row.get("status") == "semanticRejected"),
        "structural_rejected": sum(1 for row in rows if row.get("status") == "structuralRejected"),
        "blueprint_accepted_with_warnings": sum(
            1 for row in accepted_rows if row.get("semantic_warning_codes")
        ),
        "blueprint_accepted_without_warnings": sum(
            1 for row in accepted_rows if not row.get("semantic_warning_codes")
        ),
        "refine_failed": sum(1 for row in rows if row.get("status") == "error" and row.get("phase") == "phase3"),
        "exhausted": sum(1 for row in rows if row.get("status") == "exhausted"),
        "infra_error": sum(1 for row in rows if int(row.get("infra_error_node_count") or 0) > 0),
    }


def _format_elapsed_time(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}min")
    if not parts:
        parts.append(f"{secs}s")
    return "".join(parts)


def _parse_elapsed_time(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    total = 0
    token = ""
    i = 0
    while i < len(text):
        char = text[i]
        if char.isdigit():
            token += char
            i += 1
            continue
        if not token:
            return None
        number = int(token)
        token = ""
        if text.startswith("min", i):
            total += number * 60
            i += 3
        elif char == "h":
            total += number * 3600
            i += 1
        elif char == "s":
            total += number
            i += 1
        else:
            return None
    if token:
        total += int(token)
    return float(total)


def _runtime_history_path(output_root: Path) -> Path:
    return output_root / RUNTIME_HISTORY_FILENAME


def _load_runtime_history(output_root: Path) -> dict[str, Any]:
    path = _runtime_history_path(output_root)
    if not path.exists():
        return {"total_elapsed_s": 0.0, "runs": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"total_elapsed_s": 0.0, "runs": []}
    if not isinstance(data, dict):
        return {"total_elapsed_s": 0.0, "runs": []}
    runs = data.get("runs")
    if not isinstance(runs, list):
        runs = []
    total_elapsed_s = data.get("total_elapsed_s", 0.0)
    try:
        total_elapsed_s = float(total_elapsed_s)
    except (TypeError, ValueError):
        total_elapsed_s = 0.0
    return {"total_elapsed_s": total_elapsed_s, "runs": runs}


def _load_previous_elapsed_s(output_root: Path) -> float:
    history = _load_runtime_history(output_root)
    if history["total_elapsed_s"] > 0:
        return float(history["total_elapsed_s"])

    metrics_path = output_root / "metrics.json"
    if not metrics_path.exists():
        return 0.0
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0.0
    if not isinstance(metrics, dict):
        return 0.0
    elapsed_s = metrics.get("elapsed_time_s")
    if isinstance(elapsed_s, (int, float)):
        return float(elapsed_s)
    parsed = _parse_elapsed_time(metrics.get("elapsed_time"))
    return parsed or 0.0


def _record_runtime_run(
    output_root: Path,
    *,
    previous_elapsed_s: float,
    run_elapsed_s: float,
    completed: bool,
) -> dict[str, Any]:
    history = _load_runtime_history(output_root)
    runs = list(history.get("runs") or [])
    total_elapsed_s = previous_elapsed_s + run_elapsed_s
    runs.append({
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(run_elapsed_s, 3),
        "elapsed_time": _format_elapsed_time(run_elapsed_s),
        "completed": completed,
    })
    data = {
        "total_elapsed_s": round(total_elapsed_s, 3),
        "total_elapsed_time": _format_elapsed_time(total_elapsed_s),
        "runs": runs,
    }
    write_json(_runtime_history_path(output_root), data)
    return data


def _new_success_by_refinement_iteration(
    rows: list[dict[str, Any]],
) -> list[dict[str, int]]:
    observed_iterations = [max(0, int(row.get("iterations") or 0)) for row in rows]
    max_iteration = max(observed_iterations, default=0)
    counts: dict[int, int] = defaultdict(int)
    for row, iteration in zip(rows, observed_iterations):
        if row.get("root_proved") is True:
            counts[iteration] += 1
    return [
        {
            "refinement_iterations": iteration,
            "new_success_count": counts[iteration],
        }
        for iteration in range(max_iteration + 1)
    ]


def _format_new_success_by_refinement_iteration(rows: list[dict[str, Any]]) -> str:
    buckets = _new_success_by_refinement_iteration(rows)
    iterations = [str(bucket["refinement_iterations"]) for bucket in buckets]
    counts = [str(bucket["new_success_count"]) for bucket in buckets]
    return "\n".join(
        [
            "| result \\ refine_iterations | " + " | ".join(iterations) + " |",
            "| " + " | ".join(["---"] * (len(iterations) + 1)) + " |",
            "| new_success_count | " + " | ".join(counts) + " |",
        ]
    )


def _generation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generation_rows = [row for row in rows if row.get("generation_validation")]
    if not generation_rows:
        return {}
    warning_codes: dict[str, int] = defaultdict(int)
    deterministic_codes: dict[str, int] = defaultdict(int)
    semantic_codes: dict[str, int] = defaultdict(int)
    success_by_round: dict[str, int] = defaultdict(int)
    warning_ids: list[str] = []
    for row in generation_rows:
        details = row.get("generation_validation") or {}
        warnings = details.get("finalWarnings") or []
        deterministic = details.get("finalDeterministicErrors") or []
        semantic = details.get("finalSemanticErrors") or []
        for item in warnings:
            warning_codes[str(item.get("code") or "unknown")] += 1
        for item in deterministic:
            deterministic_codes[str(item.get("code") or "unknown")] += 1
        for item in semantic:
            semantic_codes[str(item.get("code") or "unknown")] += 1
        if row.get("status") in {"strictAccepted", "acceptedWithWarnings"}:
            rounds = details.get("generationRounds") or []
            success_by_round[str(len(rounds))] += 1
        if row.get("status") == "acceptedWithWarnings":
            warning_ids.append(str(row.get("source_id") or row.get("id") or ""))
    strict = sum(row.get("status") == "strictAccepted" for row in generation_rows)
    with_warnings = sum(row.get("status") == "acceptedWithWarnings" for row in generation_rows)
    accepted = strict + with_warnings
    return {
        "total": len(generation_rows),
        "strictAccepted": strict,
        "acceptedWithWarnings": with_warnings,
        "acceptedTotal": accepted,
        "warningAcceptedRate": with_warnings / len(generation_rows) if generation_rows else 0.0,
        "warningCodeDistribution": dict(sorted(warning_codes.items())),
        "warningAcceptedSourceIds": sorted(warning_ids),
        "successByRound": dict(sorted(success_by_round.items(), key=lambda item: int(item[0]))),
        "finalDeterministicErrorCount": sum(deterministic_codes.values()),
        "finalDeterministicErrorDistribution": dict(sorted(deterministic_codes.items())),
        "finalSemanticErrorCount": sum(semantic_codes.values()),
        "finalSemanticErrorDistribution": dict(sorted(semantic_codes.items())),
    }


def _write_metrics(
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    elapsed_s: float | None = None,
    current_run_elapsed_s: float | None = None,
) -> dict[str, Any]:
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
        "new_success_by_refinement_iteration": _new_success_by_refinement_iteration(rows),
    }
    generation = _generation_summary(rows)
    if generation:
        metrics["blueprint_generation"] = generation
    if elapsed_s is not None:
        metrics["elapsed_time"] = _format_elapsed_time(elapsed_s)
        metrics["elapsed_time_s"] = round(elapsed_s, 3)
    if current_run_elapsed_s is not None:
        metrics["current_run_elapsed_time"] = _format_elapsed_time(current_run_elapsed_s)
        metrics["current_run_elapsed_time_s"] = round(current_run_elapsed_s, 3)
    write_json(output_root / "metrics.json", metrics)
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    return metrics


def _should_skip_existing(row: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    if not args.resume or row is None:
        return False
    if getattr(args, "execution_mode", "full") == "phase2_only":
        status = str(row.get("status") or "")
        if status in {"solved", "exhausted", "phase1Ineligible"}:
            return True
        if status in {
            "error", "infraError", "structuralRejected", "semanticRejected", "semanticAuditError",
        }:
            return not args.retry_error_results
        return False
    if row.get("root_proved") is True or row.get("status") in {
        "exhausted", "strictAccepted",
        "acceptedWithWarnings",
    }:
        return True
    return row.get("status") in {
        "error", "semanticRejected", "semanticAuditError", "structuralRejected", "infraError",
    } and not args.retry_error_results


def _phase1_source_results(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    assert args.phase1_source_results_path is not None
    rows: dict[str, dict[str, Any]] = {}
    with args.phase1_source_results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                source_id = str(row.get("source_id") or "")
                status = str(row.get("status") or "")
                if (
                    str(row.get("phase") or "") == "phase1"
                    and status in {"strictAccepted", "acceptedWithWarnings"}
                ):
                    # Preserve the accepted Phase 1 record even when a legacy
                    # in-place Phase 2 later appended solved/exhausted rows.
                    rows[source_id] = row
                elif source_id not in rows:
                    rows[source_id] = row
    return rows


def _phase1_ineligible_result(
    record: Record,
    args: argparse.Namespace,
    source_result: dict[str, Any] | None,
) -> dict[str, Any]:
    source_status = str((source_result or {}).get("status") or "missing")
    return {
        "id": record.unique_id,
        "record_id": record.record_id,
        "source_id": record.source_id,
        "subset": record.subset,
        "split": record.split,
        "row_index": record.row_index,
        "theorem_name": record.theorem_name,
        "status": "phase1Ineligible",
        "phase": "phase1Seed",
        "success": False,
        "root_proved": False,
        "error": "Phase 1 result is not accepted for Phase 2.",
        "phase1_source_status": source_status,
        "phase1_source_root": str(args.phase1_source_experiment_root),
        "phase1_source_phase": str((source_result or {}).get("phase") or ""),
        "phase1_source_error": str((source_result or {}).get("error") or ""),
        "phase1_source_checkpoint": str(
            (source_result or {}).get("checkpoint_path") or ""
        ),
    }


async def _run_experiment(
    args: argparse.Namespace,
    output_root: Path,
    runtime: LeanRuntime,
    *,
    started_at: float | None = None,
    previous_elapsed_s: float = 0.0,
) -> None:
    records = _select_records(args)
    results_path = output_root / "results.jsonl"
    rounds_path = output_root / "rounds.jsonl"
    if not args.resume:
        for path in (results_path, rounds_path, output_root / "metrics.json", output_root / "metrics.csv"):
            unlink_if_exists(path)

    existing = _existing_results(results_path)
    phase1_results = (
        _phase1_source_results(args) if args.execution_mode == "phase2_only" else {}
    )
    eligible_records: list[Record] = []
    ineligible_rows: list[dict[str, Any]] = []
    for record in records:
        source_result = phase1_results.get(record.source_id)
        if (
            args.execution_mode == "phase2_only"
            and str((source_result or {}).get("status") or "")
            not in {"strictAccepted", "acceptedWithWarnings"}
        ):
            ineligible_rows.append(_phase1_ineligible_result(record, args, source_result))
        else:
            eligible_records.append(record)

    if args.execution_mode == "phase2_only":
        ineligible_ids = {row["id"] for row in ineligible_rows}
        _remove_jsonl_rows(results_path, ineligible_ids)
        for row in ineligible_rows:
            append_jsonl(results_path, row)
            existing[row["id"]] = row
        print(
            f"[phase2-input] total={len(records)} eligible={len(eligible_records)} "
            f"phase1Ineligible={len(ineligible_rows)} source={args.phase1_source_experiment_root}",
            flush=True,
        )

    pending: list[Record] = []
    skipped = 0
    for record in eligible_records:
        row = existing.get(record.unique_id)
        if _should_skip_existing(row, args):
            skipped += 1
            continue
        pending.append(record)

    pending_ids = {record.unique_id for record in pending}
    if args.resume:
        if args.execution_mode != "phase2_only":
            _remove_jsonl_rows(results_path, pending_ids)
            _remove_jsonl_rows(rounds_path, pending_ids)
        else:
            retry_ids = {
                record.unique_id for record in pending
                if str((existing.get(record.unique_id) or {}).get("status") or "") in {
                    "error", "infraError", "structuralRejected", "semanticRejected", "semanticAuditError",
                }
            }
            _remove_jsonl_rows(results_path, pending_ids)
            for record in pending:
                if record.unique_id in retry_ids:
                    _clear_record_outputs(output_root, record)
            _remove_jsonl_rows(rounds_path, retry_ids)
        for record_id in pending_ids:
            existing.pop(record_id, None)
    if not args.resume:
        for record in eligible_records:
            _clear_record_outputs(output_root, record)
    elif args.execution_mode != "phase2_only":
        for record in pending:
            _clear_record_outputs(output_root, record)

    print(f"[select] records={len(records)} pending={len(pending)} skipped={skipped} output={output_root}", flush=True)
    print(
        "[concurrency] "
        f"phase1={args.phase1_concurrency} "
        f"phase2_blueprint={args.phase2_blueprint_concurrency} "
        f"phase2_node={args.phase2_node_concurrency} "
        f"refine={args.refine_concurrency} "
        f"phase2_contract_check={args.phase2_contract_check_concurrency} "
        f"lean_max_inflight_snippets={args.lean_max_inflight_snippets} "
        f"lean_batch_size={args.lean_batch_size} "
        f"lean_global_batching={args.lean_global_batching} "
        f"lean_parallel_batches={args.lean_parallel_batches} "
        f"lean_batch_wait_ms={args.lean_batch_wait_ms} "
        f"node_timeout_s={args.node_timeout_s} "
        f"llm_api_timeout_s={args.llm_api_timeout_s} "
        f"node_max_prove_turns={args.node_max_prove_turns} "
        f"negation_probe_turns={args.node_max_negation_probe_turns} "
        f"max_tool_calls_per_turn={args.max_tool_calls_per_turn} "
        f"proof_policy={args.proof_policy} "
        "blueprint_generation=full_regeneration "
        "semantic_static_gate=false static_gate_mode=shadow semantic_strict_comparator=true",
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
            # Review artifacts are deliberately derived after a terminal row is
            # available.  They are read-only snapshots: a viewer never writes
            # back into this experiment or mutates any Blueprint candidate.
            try:
                review_path, review = write_review_artifact(output_root, row)
                row["review_schema_version"] = review.get("schemaVersion")
                row["review_artifact_path"] = str(review_path)
            except Exception as exc:  # Observability must not invalidate an experiment.
                row["review_artifact_error"] = f"{type(exc).__name__}: {exc}"
            async with results_lock:
                try:
                    append_jsonl(results_path, row)
                except Exception as exc:
                    # A result-persistence bug in one record must not cancel
                    # every concurrent record.  Save a compact terminal row
                    # which is deliberately built only from JSON primitives;
                    # the original terminal status remains visible for
                    # diagnosis and the record can be retried explicitly.
                    persistence_error = f"{type(exc).__name__}: {exc}"
                    row = {
                        "id": record.unique_id,
                        "record_id": record.record_id,
                        "source_id": record.source_id,
                        "subset": record.subset,
                        "split": record.split,
                        "theorem_name": record.theorem_name,
                        "status": "infraError",
                        "phase": "resultPersistence",
                        "success": False,
                        "root_proved": False,
                        "error": persistence_error,
                        "terminal_status_before_result_write": str(
                            row.get("status") or ""
                        ),
                        "trace_path": str(
                            _record_paths(output_root, record)[1]
                        ),
                        "blueprint_dir": str(
                            _record_paths(output_root, record)[2]
                        ),
                    }
                    append_jsonl(results_path, row)
                if row.get("review_artifact_path"):
                    artifact_path = Path(str(row["review_artifact_path"]))
                    try:
                        append_jsonl(
                            output_root / "review_index.jsonl",
                            index_entry(artifact_path, output_root, review),
                        )
                    except Exception as exc:
                        # The primary terminal row is already durable.  A
                        # viewer index failure is observability-only and must
                        # not abort the experiment.
                        tqdm.write(
                            "[review-index-error] "
                            f"{record.unique_id}: {type(exc).__name__}: {exc}"
                        )
                existing[row["id"]] = row
            tqdm.write(f"[record-{row['status']}] {record.unique_id} root_proved={row.get('root_proved')}")

        tasks = [asyncio.create_task(run_one(record)) for record in pending]
        with tqdm(total=len(tasks), desc="robustpa", unit="record") as progress:
            for task in asyncio.as_completed(tasks):
                await task
                progress.update(1)

    selected_ids = {record.unique_id for record in records}
    final_rows = [row for row_id, row in existing.items() if row_id in selected_ids]
    current_run_elapsed_s = time.perf_counter() - started_at if started_at is not None else None
    elapsed_s = (
        previous_elapsed_s + current_run_elapsed_s
        if current_run_elapsed_s is not None
        else None
    )
    metrics = _write_metrics(
        output_root,
        final_rows,
        elapsed_s=elapsed_s,
        current_run_elapsed_s=current_run_elapsed_s,
    )
    print(f"[done] results={results_path}", flush=True)
    print(f"[rounds] {rounds_path}", flush=True)
    print(f"[metrics] primary root_proved_acc={metrics['groups'][0]['root_proved_acc']}", flush=True)
    if metrics.get("blueprint_generation"):
        print("[blueprint-generation-summary] " + json.dumps(
            metrics["blueprint_generation"], ensure_ascii=False
        ), flush=True)
    print("[new-success-by-refinement-iteration]", flush=True)
    print(_format_new_success_by_refinement_iteration(final_rows), flush=True)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    args = parse_args(cfg)
    output_root = args.output_base / args.exp_name
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        unlink_if_exists(_runtime_history_path(output_root))
    previous_elapsed_s = _load_previous_elapsed_s(output_root) if args.resume else 0.0
    runtime = make_lean_runtime(args)
    try:
        write_lean_runtime_metadata(output_root, runtime.current_metadata())
        start = time.perf_counter()
        completed = False
        try:
            asyncio.run(
                _run_experiment(
                    args,
                    output_root,
                    runtime,
                    started_at=start,
                    previous_elapsed_s=previous_elapsed_s,
                )
            )
            completed = True
        finally:
            write_lean_runtime_metadata(output_root, runtime.current_metadata())
            run_elapsed_s = time.perf_counter() - start
            history = _record_runtime_run(
                output_root,
                previous_elapsed_s=previous_elapsed_s,
                run_elapsed_s=run_elapsed_s,
                completed=completed,
            )
            print(
                f"[runtime] duration_s={run_elapsed_s:.3f} "
                f"total_duration_s={history['total_elapsed_s']:.3f}",
                flush=True,
            )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
