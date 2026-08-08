from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_blueprint_refine.common import load_config  # noqa: E402
from cot_blueprint_refine.semantic_quality_report import write_matrix_report  # noqa: E402
from cot_blueprint_refine.vllm_runtime import PersistentVLLMRuntime  # noqa: E402
from blueprint import _render_step_grounded_proof  # noqa: E402
from llm_client import make_client  # noqa: E402
from semantic_audit import run_semantic_audit  # noqa: E402
from tracer import JsonlTracer  # noqa: E402


DEFAULT_PROFILE = "qwen3_8b_397b_wrong76_semantic_matrix"
MANIFEST_NAME = "semantic_run_manifest.json"
FINGERPRINT_SCHEMA = 2
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".sh", ".lean"}
TERMINAL_RESULT_STATUSES = {"solved", "exhausted", "error"}
REFINE_TERMINAL_STATUSES = {"ok"}
_SERVICE_FAILURE_RE = re.compile(
    r"(?:"
    r"KiminaInfrastructureError|NoAvailableREPL|No available REPL|"
    r"APIConnectionError|APITimeoutError|InternalServerError|RateLimitError|"
    r"ServiceUnavailableError|"
    r"(?:httpx\.)?(?:ConnectError|ConnectTimeout|ReadError|ReadTimeout|"
    r"RemoteProtocolError|PoolTimeout)|"
    r"Connection(?:Refused|Reset|Aborted)Error|"
    r"server disconnected|connection refused|failed to establish a new connection|"
    r"Error code:\s*(?:429|5\d\d)\b"
    r")",
    re.IGNORECASE,
)

FEATURE_TO_BLUEPRINT = {
    "fidelity_enabled": "semantic_fidelity_enabled",
    "require_step_ids": "semantic_require_step_ids",
    "static_gate": "semantic_static_gate",
    "minimal_ir": "semantic_minimal_ir",
    "freeze_refinement": "semantic_freeze_refinement",
    "audit_mode": "semantic_audit_mode",
    "max_repair_attempts": "semantic_max_repair_attempts",
    "proof_policy": "proof_policy",
    "critical_negation_max_turns": "critical_negation_max_turns",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def selection_sha256(selected_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for row_id in sorted(set(str(value) for value in selected_ids)):
        digest.update(row_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class MatrixRunLock:
    """Prevent two controllers from mutating or launching the same matrix run."""

    def __init__(self, matrix_root: Path) -> None:
        self.path = matrix_root / ".matrix.lock"
        self.handle: Any = None

    def __enter__(self) -> "MatrixRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.seek(0)
            owner = self.handle.read().strip() or "unknown"
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f"semantic matrix run is already locked: {self.path} owner={owner}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
        self.handle.flush()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def code_fingerprint(config: DictConfig) -> tuple[str, list[str]]:
    paths: list[Path] = []
    for value in config.matrix.code_roots:
        root = (REPO_ROOT / str(value)).resolve()
        if not root.exists():
            continue
        if root.is_file():
            paths.append(root)
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in CODE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    unique = sorted(set(paths), key=lambda path: str(path.relative_to(REPO_ROOT)))
    digest = hashlib.sha256()
    labels: list[str] = []
    for path in unique:
        label = str(path.relative_to(REPO_ROOT))
        labels.append(label)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), labels


def _dot_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def flatten_overrides(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        items: list[str] = []
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            items.extend(flatten_overrides(value[key], child))
        return items
    if not prefix:
        raise ValueError("a scalar override requires a key")
    return [f"{prefix}={_dot_value(value)}"]


def select_arms(config: DictConfig, mode: str, explicit: str = "") -> list[str]:
    available = list(config.matrix.arms.keys())
    if explicit:
        requested = [item.strip().upper() for item in explicit.split(",") if item.strip()]
    elif mode == "add":
        requested = [str(value) for value in config.matrix.add_arms]
    elif mode == "reduce":
        requested = [str(value) for value in config.matrix.reduce_arms]
    else:
        requested = [
            *[str(value) for value in config.matrix.add_arms],
            *[str(value) for value in config.matrix.reduce_arms],
        ]
    unknown = [arm for arm in requested if arm not in available]
    if unknown:
        raise ValueError(f"unknown matrix arms: {unknown}; available={available}")
    if len(requested) != len(set(requested)):
        raise ValueError(f"duplicate matrix arms requested: {requested}")
    return requested


def should_refine(config: DictConfig, arm: str, policy: str) -> bool:
    if policy == "all":
        return True
    if policy == "none":
        return False
    return arm in {str(value) for value in config.matrix.key_refine_arms}


def arm_overrides(
    config: DictConfig,
    arm: str,
    *,
    run_id: str,
    external_vllm: bool = True,
) -> list[str]:
    arm_config = config.matrix.arms[arm]
    features = OmegaConf.to_container(arm_config.features, resolve=True)
    if not isinstance(features, dict):
        raise ValueError(f"matrix.arms.{arm}.features must be a mapping")
    unknown = sorted(set(features) - set(FEATURE_TO_BLUEPRINT))
    missing = sorted(set(FEATURE_TO_BLUEPRINT) - set(features))
    if unknown or missing:
        raise ValueError(
            f"invalid feature map for {arm}: unknown={unknown} missing={missing}"
        )
    output_name = f"{config.matrix.output_prefix}/{run_id}/{arm}"
    overrides = [
        f"exp_name={output_name}",
        "resume=true",
        f"matrix.active_arm={arm}",
        f"matrix.active_run_id={run_id}",
    ]
    if external_vllm:
        overrides.extend(("vllm.auto_start=false", "vllm.auto_destroy=false"))
    for feature_name, blueprint_name in FEATURE_TO_BLUEPRINT.items():
        overrides.append(
            f"blueprint.{blueprint_name}={_dot_value(features[feature_name])}"
        )
    arm_specific = arm_config.get("overrides")
    if arm_specific:
        overrides.extend(flatten_overrides(arm_specific))
    return overrides


def effective_config_fingerprint(profile: str, overrides: list[str]) -> str:
    effective = load_config(profile, overrides)
    payload = OmegaConf.to_container(effective, resolve=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl_source_ids(path: Path, *, key: str) -> set[str]:
    """Read a JSONL identity set strictly enough for resume validation."""
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            identity = str(value.get(key) or "")
            if not identity:
                raise ValueError(f"missing {key!r} at {path}:{line_number}")
            if identity in identities:
                raise ValueError(f"duplicate {key}={identity!r} in {path}")
            identities.add(identity)
    if not identities:
        raise ValueError(f"no rows in {path}")
    return identities


def _jsonl_index(
    path: Path,
    *,
    key: str,
    allow_duplicates: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return a strict JSONL index plus structural errors.

    RobustPA result files append retries, so their dedicated validator permits
    duplicate IDs and treats the last row as authoritative. Stage artifacts
    written as snapshots must contain exactly one row per ID.
    """
    rows: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if not path.exists():
        return rows, [f"missing file: {path}"]
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: expected JSON object")
                continue
            identity = str(value.get(key) or "")
            if not identity:
                errors.append(f"line {line_number}: missing {key!r}")
                continue
            if identity in rows and not allow_duplicates:
                errors.append(f"line {line_number}: duplicate {key}={identity!r}")
            rows[identity] = value
    if not rows:
        errors.append(f"no rows in {path}")
    return rows, errors


def _identity_blockers(
    rows: dict[str, dict[str, Any]],
    errors: list[str],
    expected_ids: set[str],
) -> dict[str, Any]:
    observed_ids = set(rows)
    return {
        key: value
        for key, value in {
            "parse_errors": errors,
            "missing_ids": sorted(expected_ids - observed_ids),
            "unexpected_ids": sorted(observed_ids - expected_ids),
        }.items()
        if value
    }


def _result_is_infrastructure_failure(row: dict[str, Any]) -> bool:
    try:
        infra_count = int(row.get("infra_error_node_count") or 0)
    except (TypeError, ValueError):
        infra_count = 0
    if (
        infra_count > 0
        or bool(row.get("infra_error"))
        or str(row.get("failure_kind") or "").lower() == "infra"
        or bool(row.get("infra_error_nodes"))
    ):
        return True
    # Phase 2/3 record-local Kimina failures are persisted by the pipeline as
    # checkpoint status=error and returned without re-raising, so the result
    # row intentionally has an empty `error` string. Other terminal model or
    # Lean-quality failures carry an exception/error or become exhausted.
    if (
        str(row.get("status") or "") == "error"
        and str(row.get("phase") or "") in {"phase2", "terminal"}
        and not str(row.get("error") or "").strip()
        and not _result_is_semantic_rejection(row)
    ):
        return True
    diagnostic = "\n".join(
        str(row.get(key) or "") for key in ("error", "traceback")
    )
    return bool(_SERVICE_FAILURE_RE.search(diagnostic))


def _result_is_semantic_rejection(row: dict[str, Any]) -> bool:
    failure_stage = str(row.get("failed_blueprint_failure_stage") or "").lower()
    semantic_status = str(row.get("semantic_status") or "").lower()
    traceback_text = str(row.get("traceback") or "")
    return (
        failure_stage.startswith("semantic")
        or "rejected" in semantic_status
        or "SemanticRefinementError" in traceback_text
    )


def validate_blueprint_results(
    arm_root: Path,
    expected_ids: set[str],
) -> dict[str, Any]:
    """Validate record-level completion before committing a matrix stage.

    Semantic-gate rejection and ordinary terminal modeling failures are valid
    experimental outcomes. Missing/nonterminal rows and infrastructure/service
    failures are not: recording the stage complete in those cases would make a
    later matrix resume silently skip unfinished work.
    """
    results_path = arm_root / "robustpa/blueprint/results.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    parse_errors: list[str] = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"line {line_number}: {exc}")
                    continue
                if not isinstance(value, dict):
                    parse_errors.append(f"line {line_number}: expected JSON object")
                    continue
                source_id = str(value.get("source_id") or "")
                if not source_id:
                    parse_errors.append(f"line {line_number}: missing source_id")
                    continue
                # RobustPA appends retries; the last row is authoritative.
                rows[source_id] = value
    else:
        parse_errors.append(f"missing results file: {results_path}")

    observed_ids = set(rows)
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    scoped_rows = {row_id: rows[row_id] for row_id in sorted(expected_ids & observed_ids)}
    nonterminal_ids = sorted(
        row_id
        for row_id, row in scoped_rows.items()
        if str(row.get("status") or "") not in TERMINAL_RESULT_STATUSES
    )
    infra_error_ids = sorted(
        row_id for row_id, row in scoped_rows.items()
        if _result_is_infrastructure_failure(row)
    )
    semantic_rejection_ids = sorted(
        row_id for row_id, row in scoped_rows.items()
        if _result_is_semantic_rejection(row)
    )
    terminal_error_ids = sorted(
        row_id for row_id, row in scoped_rows.items()
        if str(row.get("status") or "") == "error"
    )
    quality_warning_ids = sorted(
        set(terminal_error_ids) - set(infra_error_ids) - set(semantic_rejection_ids)
    )
    blocking_reasons = {
        key: value
        for key, value in {
            "parse_errors": parse_errors,
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "nonterminal_ids": nonterminal_ids,
            "infra_error_ids": infra_error_ids,
        }.items()
        if value
    }
    return {
        "passed": not blocking_reasons,
        "quality_warning": bool(quality_warning_ids or semantic_rejection_ids),
        "expected_count": len(expected_ids),
        "observed_count": len(observed_ids),
        "terminal_count": sum(
            str(row.get("status") or "") in TERMINAL_RESULT_STATUSES
            for row in scoped_rows.values()
        ),
        "status_counts": dict(sorted(Counter(
            str(row.get("status") or "missing") for row in scoped_rows.values()
        ).items())),
        "semantic_rejection_ids": semantic_rejection_ids,
        "quality_warning_ids": quality_warning_ids,
        "blocking_reasons": blocking_reasons,
        "results_path": str(results_path),
        "validated_at": utc_now(),
    }


def validate_cot_to_blueprint_results(
    arm_root: Path,
    expected_ids: set[str],
) -> dict[str, Any]:
    """Validate prepare, RobustPA, and export as one resumable stage."""
    result_validation = validate_blueprint_results(arm_root, expected_ids)
    generation_path = arm_root / "prepared/generation_inputs.jsonl"
    generation, generation_errors = _jsonl_index(generation_path, key="name")
    context_path = arm_root / "blueprint_contexts/blueprint_contexts.jsonl"
    contexts, context_errors = _jsonl_index(context_path, key="ID")
    result_rows, _result_errors = _jsonl_index(
        arm_root / "robustpa/blueprint/results.jsonl",
        key="source_id",
        allow_duplicates=True,
    )

    generation_blockers = _identity_blockers(
        generation, generation_errors, expected_ids,
    )
    context_blockers = _identity_blockers(contexts, context_errors, expected_ids)
    scoped_contexts = {
        row_id: contexts[row_id] for row_id in sorted(expected_ids & set(contexts))
    }
    noninfra_terminal_context_ids = sorted(
        row_id for row_id, row in scoped_contexts.items()
        if str(row.get("context_quality") or "") == "INFRA_ERROR"
        and str(result_rows.get(row_id, {}).get("status") or "") == "error"
        and not _result_is_infrastructure_failure(result_rows.get(row_id, {}))
    )
    infra_context_ids = sorted(
        row_id for row_id, row in scoped_contexts.items()
        if str(row.get("context_quality") or "") == "INFRA_ERROR"
        and row_id not in set(noninfra_terminal_context_ids)
    )
    unknown_context_ids = sorted(
        row_id for row_id, row in scoped_contexts.items()
        if str(row.get("context_quality") or "")
        not in {"VERIFIED", "INVALID_BLUEPRINT_CANDIDATE", "INFRA_ERROR"}
    )
    invalid_candidate_ids = sorted(
        row_id for row_id, row in scoped_contexts.items()
        if str(row.get("context_quality") or "") == "INVALID_BLUEPRINT_CANDIDATE"
    )
    if infra_context_ids:
        context_blockers["infra_context_ids"] = infra_context_ids
    if unknown_context_ids:
        context_blockers["unknown_context_quality_ids"] = unknown_context_ids

    blocking_reasons: dict[str, Any] = {}
    if result_validation["blocking_reasons"]:
        blocking_reasons["robustpa"] = result_validation["blocking_reasons"]
    if generation_blockers:
        blocking_reasons["prepare"] = generation_blockers
    if context_blockers:
        blocking_reasons["export"] = context_blockers
    return {
        **result_validation,
        "passed": not blocking_reasons,
        "quality_warning": bool(
            result_validation["quality_warning"]
            or invalid_candidate_ids
            or noninfra_terminal_context_ids
        ),
        "blocking_reasons": blocking_reasons,
        "prepared_count": len(generation),
        "export_count": len(contexts),
        "invalid_blueprint_candidate_ids": invalid_candidate_ids,
        "noninfra_terminal_context_ids": noninfra_terminal_context_ids,
        "prepared_path": str(generation_path),
        "export_path": str(context_path),
        "validated_at": utc_now(),
    }


def enabled_refine_variants(config: DictConfig) -> list[str]:
    variants = config.refine.get("variants")
    if variants is None:
        return ["blueprint"]
    enabled = [
        str(name) for name, value in variants.items()
        if bool(value.get("enabled", True))
    ]
    if not enabled:
        raise ValueError("at least one refinement variant must be enabled")
    return enabled


def validate_refine_results(
    arm_root: Path,
    expected_ids: set[str],
    variants: Iterable[str],
) -> dict[str, Any]:
    blocking_reasons: dict[str, Any] = {}
    variant_summaries: dict[str, Any] = {}
    for variant in variants:
        output_path = arm_root / f"refinement/{variant}/refined_predictions.jsonl"
        rows, errors = _jsonl_index(output_path, key="ID")
        blockers = _identity_blockers(rows, errors, expected_ids)
        scoped = {row_id: rows[row_id] for row_id in expected_ids & set(rows)}
        non_ok_ids = sorted(
            row_id for row_id, row in scoped.items()
            if str(row.get("status") or "") not in REFINE_TERMINAL_STATUSES
        )
        missing_conversation_ids = sorted(
            row_id for row_id, row in scoped.items()
            if str(row.get("status") or "") == "ok"
            and (
                not str(row.get("conversation_path") or "")
                or not Path(str(row.get("conversation_path") or "")).is_file()
            )
        )
        if non_ok_ids:
            blockers["non_ok_ids"] = non_ok_ids
        if missing_conversation_ids:
            blockers["missing_conversation_ids"] = missing_conversation_ids

        metrics_path = arm_root / f"refinement/{variant}/refinement_metrics.json"
        try:
            metrics = _load_json(metrics_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            metrics = {}
            blockers["metrics_error"] = f"{type(exc).__name__}: {exc}"
        if metrics and int(metrics.get("rows") or -1) != len(expected_ids):
            blockers["metrics_row_count"] = {
                "actual": metrics.get("rows"), "expected": len(expected_ids),
            }
        variant_summaries[str(variant)] = {
            "rows": len(rows),
            "status_counts": dict(sorted(Counter(
                str(row.get("status") or "missing") for row in scoped.values()
            ).items())),
            "output_path": str(output_path),
            "metrics_path": str(metrics_path),
        }
        if blockers:
            blocking_reasons[str(variant)] = blockers
    return {
        "passed": not blocking_reasons,
        "expected_count": len(expected_ids),
        "variants": variant_summaries,
        "blocking_reasons": blocking_reasons,
        "validated_at": utc_now(),
    }


def validate_evaluation_results(
    arm_root: Path,
    expected_ids: set[str],
    variants: Iterable[str],
    *,
    judge_enabled: bool,
) -> dict[str, Any]:
    blocking_reasons: dict[str, Any] = {}
    variant_summaries: dict[str, Any] = {}
    variant_list = [str(value) for value in variants]
    for variant in variant_list:
        comparison_path = arm_root / f"evaluation/{variant}/comparison.jsonl"
        rows, errors = _jsonl_index(comparison_path, key="ID")
        blockers = _identity_blockers(rows, errors, expected_ids)
        scoped = {row_id: rows[row_id] for row_id in expected_ids & set(rows)}
        bad_refine_ids = sorted(
            row_id for row_id, row in scoped.items()
            if str(row.get("refine_status") or "") != "ok"
        )
        if bad_refine_ids:
            blockers["non_ok_refine_ids"] = bad_refine_ids
        if judge_enabled:
            judge_error_ids = sorted(
                row_id for row_id, row in scoped.items()
                if str(row.get("before_judge_status") or "") == "error"
                or str(row.get("after_judge_status") or "") == "error"
            )
            if judge_error_ids:
                blockers["judge_error_ids"] = judge_error_ids

        metrics_path = arm_root / f"evaluation/{variant}/metrics.json"
        try:
            metrics = _load_json(metrics_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            metrics = {}
            blockers["metrics_error"] = f"{type(exc).__name__}: {exc}"
        metric_rows = (
            metrics.get("efficiency", {}).get("rows")
            if isinstance(metrics.get("efficiency"), dict) else None
        )
        if metrics and int(metric_rows or -1) != len(expected_ids):
            blockers["metrics_row_count"] = {
                "actual": metric_rows, "expected": len(expected_ids),
            }
        variant_summaries[variant] = {
            "rows": len(rows),
            "comparison_path": str(comparison_path),
            "metrics_path": str(metrics_path),
        }
        if blockers:
            blocking_reasons[variant] = blockers

    aggregate_path = arm_root / "evaluation/metrics.json"
    try:
        aggregate = _load_json(aggregate_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        aggregate = {}
        blocking_reasons["aggregate"] = {
            "metrics_error": f"{type(exc).__name__}: {exc}",
        }
    if aggregate:
        aggregate_variants = aggregate.get("variants")
        missing_variants = sorted(
            set(variant_list) - set(aggregate_variants or {})
            if isinstance(aggregate_variants, dict) else set(variant_list)
        )
        artifacts = (
            aggregate.get("analysis_artifacts")
            if isinstance(aggregate.get("analysis_artifacts"), dict) else {}
        )
        expected_analysis_rows = len(expected_ids) * len(variant_list)
        artifact_errors: dict[str, Any] = {}
        if missing_variants:
            artifact_errors["missing_variants"] = missing_variants
        if int(artifacts.get("parquet_rows") or -1) != expected_analysis_rows:
            artifact_errors["analysis_row_count"] = {
                "actual": artifacts.get("parquet_rows"),
                "expected": expected_analysis_rows,
            }
        parquet_path = str(artifacts.get("parquet") or "")
        if not parquet_path or not Path(parquet_path).is_file():
            artifact_errors["missing_analysis_parquet"] = parquet_path
        if artifact_errors:
            blocking_reasons["aggregate"] = {
                **blocking_reasons.get("aggregate", {}), **artifact_errors,
            }
    return {
        "passed": not blocking_reasons,
        "expected_count": len(expected_ids),
        "variants": variant_summaries,
        "blocking_reasons": blocking_reasons,
        "aggregate_path": str(aggregate_path),
        "validated_at": utc_now(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_arm_manifest(
    *,
    arm_root: Path,
    arm: str,
    run_id: str,
    input_path: Path,
    input_sha256: str,
    code_sha256: str,
    config_sha256: str,
    expected_ids: set[str],
    label: str,
    parent: str,
    dry_run: bool,
) -> dict[str, Any]:
    path = arm_root / MANIFEST_NAME
    expected = {
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "arm": arm,
        "run_id": run_id,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        "config_sha256": config_sha256,
        "selected_ids": sorted(expected_ids),
        "selection_sha256": selection_sha256(expected_ids),
        "expected_result_count": len(expected_ids),
    }
    if path.exists():
        manifest = _load_json(path)
        mismatches = {
            key: {"existing": manifest.get(key), "requested": value}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"unsafe resume refused for {arm_root}; fingerprints differ: "
                f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}. "
                "Use a new RUN_ID."
            )
        return manifest
    if arm_root.exists() and any(arm_root.iterdir()):
        raise RuntimeError(
            f"unsafe resume refused: non-empty arm root has no {MANIFEST_NAME}: {arm_root}"
        )
    manifest = {
        **expected,
        "label": label,
        "parent": parent,
        "status": "planned" if dry_run else "created",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "completed_stages": [],
        "failures": [],
    }
    if not dry_run:
        _write_json(path, manifest)
    return manifest


def update_manifest(arm_root: Path, manifest: dict[str, Any], **updates: Any) -> None:
    manifest.update(updates)
    manifest["updated_at"] = utc_now()
    _write_json(arm_root / MANIFEST_NAME, manifest)


def begin_suite_manifest(
    *,
    path: Path,
    identity: dict[str, Any],
    requested_arms: list[str],
    refine_policy: str,
    input_path: Path,
    code_paths: list[str],
    legacy_root: Path,
    available_arms: Iterable[str],
) -> tuple[dict[str, Any], str, list[str]]:
    existing = _load_json(path) if path.exists() else {}
    mismatches = {
        key: {"existing": existing.get(key), "requested": value}
        for key, value in identity.items()
        if existing and existing.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "unsafe matrix resume refused; suite fingerprints differ: "
            f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}. "
            "Use a new RUN_ID."
        )

    now = utc_now()
    executions = [dict(value) for value in (existing.get("executions") or [])]
    for execution in executions:
        if not execution.get("finished_at"):
            execution.update({
                "finished_at": now,
                "status": "interrupted_before_resume",
                "failures": [{
                    "arm": "_suite",
                    "error": "previous controller exited without finalizing this execution",
                }],
            })
    known_set = {
        *[str(value) for value in (existing.get("known_arms") or [])],
        *[str(value) for value in (existing.get("arms") or [])],
        *requested_arms,
    }
    known_arms = [str(arm) for arm in available_arms if str(arm) in known_set]
    execution_id = uuid.uuid4().hex
    executions.append({
        "execution_id": execution_id,
        "started_at": now,
        "requested_arms": requested_arms,
        "refine_policy": refine_policy,
    })
    payload = {
        **existing,
        **identity,
        "status": "starting",
        "created_at": existing.get("created_at") or now,
        "started_at": now,
        "current_execution_id": execution_id,
        "requested_arms": requested_arms,
        # Kept for older report readers; unlike previous behavior this is the
        # accumulated set rather than only the most recent invocation.
        "arms": known_arms,
        "known_arms": known_arms,
        "refine_policy": refine_policy,
        "input_path": str(input_path),
        "code_paths": code_paths,
        "legacy_root": str(legacy_root),
        "executions": executions,
        "failures": [],
    }
    payload.pop("finished_at", None)
    _write_json(path, payload)
    return payload, execution_id, known_arms


def finish_suite_manifest(
    path: Path,
    *,
    execution_id: str,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    suite = _load_json(path)
    finished_at = utc_now()
    status = "failed" if failures else "complete"
    matched = False
    for execution in suite.get("executions") or []:
        if str(execution.get("execution_id") or "") == execution_id:
            execution.update({
                "finished_at": finished_at,
                "status": status,
                "failures": failures,
            })
            matched = True
            break
    if not matched:
        raise RuntimeError(f"matrix execution disappeared from manifest: {execution_id}")
    suite.update({
        "status": status,
        "finished_at": finished_at,
        "current_execution_id": None,
        "failures": failures,
    })
    _write_json(path, suite)
    return suite


def _run_command(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[semantic-matrix-command] " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def child_command(profile: str, stage: str, overrides: Iterable[str]) -> list[str]:
    return [
        sys.executable,
        str(EXPERIMENT_DIR / "run_experiment.py"),
        "--profile",
        profile,
        "--stage",
        stage,
        *overrides,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the wrong-answer-76 semantic-fidelity add/reduce matrix."
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--mode", choices=("add", "reduce", "all"), default="all")
    parser.add_argument("--arms", default="", help="Comma-separated explicit arm list")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--refine", choices=("key", "all", "none"), default="key")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument(
        "--audit-replay-manifest",
        type=Path,
        help=(
            "Run only deterministic semantic-audit replay cases from a JSON "
            "manifest, sharing one owned 397B service."
        ),
    )
    parser.add_argument("override", nargs="*", help="Base OmegaConf dot-list overrides")
    return parser.parse_args()


def _jsonl_row_by_name(path: Path, name: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            if str(value.get("name") or value.get("ID") or "") == name:
                matches.append(value)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one row named {name!r} in {path}, found {len(matches)}"
        )
    return matches[0]


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _execute_audit_replay(
    *,
    config: DictConfig,
    run_id: str,
    matrix_root: Path,
    manifest_path: Path,
) -> None:
    """Replay frozen candidates through the current audit prompt only.

    This separates audit calibration from stochastic Blueprint regeneration,
    while retaining the same owned vLLM lifecycle and request implementation as
    the real pipeline.
    """
    payload = _load_json(manifest_path.resolve())
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("audit replay manifest must contain a non-empty cases list")
    replay_root = matrix_root / "_audit_replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    trace_path = replay_root / "trace.jsonl"
    results_path = replay_root / "results.jsonl"
    summary_path = replay_root / "summary.json"
    existing = [path for path in (trace_path, results_path, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "audit replay output already exists; use a fresh run-id: "
            + ", ".join(str(path) for path in existing)
        )
    suite_config = OmegaConf.merge(
        config,
        OmegaConf.create({
            "exp_name": f"{config.matrix.output_prefix}/{run_id}/_audit_replay",
            "vllm": {"auto_start": True, "auto_destroy": True},
        }),
    )
    model = str(config.blueprint.model)
    base_url = str(config.blueprint.openai_base_url)
    output_rows: list[dict[str, Any]] = []
    tracer = JsonlTracer(trace_path)
    started = time.monotonic()
    try:
        with PersistentVLLMRuntime(suite_config) as runtime:
            runtime.ensure(
                stage="semantic_matrix/audit_replay",
                client_model=model,
                base_url=base_url,
                service=config.blueprint.vllm,
            )
            os.environ["GOEDEL_OPENAI_BASE_URL"] = base_url
            os.environ.setdefault("GOEDEL_OPENAI_API_KEY", "dummy")
            client = make_client(model)
            expanded_cases: list[dict[str, Any]] = []
            for raw_case in cases:
                if not isinstance(raw_case, dict):
                    raise ValueError("every audit replay case must be an object")
                repeat = int(raw_case.get("repeat", 1))
                if repeat < 1 or repeat > 20:
                    raise ValueError("audit replay repeat must be between 1 and 20")
                for repeat_index in range(1, repeat + 1):
                    expanded_cases.append({
                        **raw_case,
                        "_repeat_index": repeat_index,
                        "_repeat_count": repeat,
                    })
            for index, raw_case in enumerate(expanded_cases, start=1):
                if not isinstance(raw_case, dict):
                    raise ValueError(f"audit replay case {index} must be an object")
                base_case_id = str(raw_case.get("case_id") or f"case_{index:03d}")
                repeat_count = int(raw_case["_repeat_count"])
                repeat_index = int(raw_case["_repeat_index"])
                case_id = (
                    f"{base_case_id}__repeat_{repeat_index}"
                    if repeat_count > 1 else base_case_id
                )
                name = str(raw_case.get("name") or "")
                expected = str(raw_case.get("expected") or "").upper()
                if expected not in {"PASS", "FAIL"}:
                    raise ValueError(f"{case_id}: expected must be PASS or FAIL")
                input_path = Path(str(raw_case.get("input_path") or "")).resolve()
                blueprint_path = Path(str(raw_case.get("blueprint_path") or "")).resolve()
                if not input_path.is_file() or not blueprint_path.is_file() or not name:
                    raise ValueError(
                        f"{case_id}: name, input_path and blueprint_path must identify files"
                    )
                source = _jsonl_row_by_name(input_path, name)
                numbered_cot = _render_step_grounded_proof(
                    str(source.get("cot_manifest_json") or ""),
                    include_ir=bool(raw_case.get("include_ir", True)),
                )
                blueprint_lean = blueprint_path.read_text(encoding="utf-8")
                case_started = time.monotonic()
                try:
                    audit = run_semantic_audit(
                        model,
                        numbered_cot,
                        blueprint_lean,
                        mode=str(raw_case.get("mode") or "full"),
                        informal_statement=str(source.get("informal_statement") or ""),
                        claimed_answer=str(source.get("claimed_answer") or ""),
                        client=client,
                        tracer=tracer,
                        thm_name=name,
                        phase="semantic_audit_replay",
                        max_tokens=int(raw_case.get(
                            "max_tokens", config.blueprint.semantic_audit_max_tokens,
                        )),
                    )
                    row = {
                        "case_id": case_id,
                        "name": name,
                        "expected": expected,
                        "actual": audit.flag,
                        "matched_expected": audit.flag == expected,
                        "repeat_index": repeat_index,
                        "repeat_count": repeat_count,
                        "input_path": str(input_path),
                        "blueprint_path": str(blueprint_path),
                        "input_sha256": sha256_file(input_path),
                        "blueprint_sha256": sha256_file(blueprint_path),
                        "numbered_cot_sha256": hashlib.sha256(
                            numbered_cot.encode("utf-8")
                        ).hexdigest(),
                        "duration_ms": (time.monotonic() - case_started) * 1000,
                        "audit": asdict(audit),
                    }
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "case_id": case_id,
                        "name": name,
                        "expected": expected,
                        "actual": "ERROR",
                        "matched_expected": False,
                        "repeat_index": repeat_index,
                        "repeat_count": repeat_count,
                        "input_path": str(input_path),
                        "blueprint_path": str(blueprint_path),
                        "duration_ms": (time.monotonic() - case_started) * 1000,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                output_rows.append(row)
                _write_jsonl_atomic(results_path, output_rows)
                print(
                    f"[audit-replay] case={case_id} expected={expected} "
                    f"actual={row['actual']} matched={row['matched_expected']}",
                    flush=True,
                )
    finally:
        tracer.close()
    matched = sum(bool(row["matched_expected"]) for row in output_rows)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path.resolve()),
        "cases": len(output_rows),
        "matched": matched,
        "all_matched": matched == len(output_rows),
        "duration_s": time.monotonic() - started,
        "results_path": str(results_path),
        "trace_path": str(trace_path),
    }
    _write_json(summary_path, summary)
    print(f"[audit-replay-summary] {summary_path} {matched}/{len(output_rows)}", flush=True)
    if not summary["all_matched"]:
        raise RuntimeError("semantic audit replay did not match all expected labels")


def _execute_matrix(
    *,
    args: argparse.Namespace,
    config: DictConfig,
    run_id: str,
    arms: list[str],
    matrix_root: Path,
    legacy_root: Path,
    subset_metrics: Path,
    input_path: Path,
    expected_ids: set[str],
    input_sha256: str,
    code_sha256: str,
    code_paths: list[str],
) -> None:
    plans: list[tuple[str, list[str], Path, dict[str, Any]]] = []
    for arm in arms:
        overrides = [*args.override, *arm_overrides(config, arm, run_id=run_id)]
        config_sha256 = effective_config_fingerprint(args.profile, overrides)
        arm_root = matrix_root / arm
        arm_config = config.matrix.arms[arm]
        manifest = prepare_arm_manifest(
            arm_root=arm_root,
            arm=arm,
            run_id=run_id,
            input_path=input_path,
            input_sha256=input_sha256,
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            expected_ids=expected_ids,
            label=str(arm_config.label),
            parent=str(arm_config.parent),
            dry_run=args.dry_run,
        )
        plans.append((arm, overrides, arm_root, manifest))

    if args.dry_run:
        for arm, overrides, _root, _manifest in plans:
            print(f"[semantic-matrix-plan] arm={arm}", flush=True)
            print(" ".join(child_command(args.profile, "cot-to-blueprint", overrides)))
            if should_refine(config, arm, args.refine):
                print(" ".join(child_command(args.profile, "refine", overrides)))
                print(" ".join(child_command(args.profile, "evaluate", overrides)))
        return

    matrix_root.mkdir(parents=True, exist_ok=True)
    suite_config = OmegaConf.merge(
        config,
        OmegaConf.create({
            "exp_name": f"{config.matrix.output_prefix}/{run_id}/_suite",
            "vllm": {"auto_start": True, "auto_destroy": True},
        }),
    )
    suite_manifest_path = matrix_root / "matrix_manifest.json"
    suite_identity = {
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "run_id": run_id,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        "expected_result_count": len(expected_ids),
        "selected_ids": sorted(expected_ids),
        "selection_sha256": selection_sha256(expected_ids),
    }
    _suite, execution_id, known_arms = begin_suite_manifest(
        path=suite_manifest_path,
        identity=suite_identity,
        requested_arms=arms,
        refine_policy=args.refine,
        input_path=input_path,
        code_paths=code_paths,
        legacy_root=legacy_root,
        available_arms=config.matrix.arms.keys(),
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(REPO_ROOT / "src"),
            str(REPO_ROOT / "experiments"),
            env.get("PYTHONPATH", ""),
        )
        if value
    )
    variants = enabled_refine_variants(config)
    failures: list[dict[str, str]] = []
    fatal_error: BaseException | None = None
    try:
        with PersistentVLLMRuntime(suite_config) as runtime:
            def ensure_model(stage: str) -> None:
                section = (
                    config.blueprint if stage == "cot-to-blueprint"
                    else config.refine if stage == "refine"
                    else config.judge
                )
                runtime.ensure(
                    stage=f"semantic_matrix/{stage}",
                    client_model=str(section.model),
                    base_url=str(section.openai_base_url),
                    service=section.vllm,
                )

            for arm, overrides, arm_root, manifest in plans:
                completed = set(str(value) for value in manifest.get("completed_stages", []))
                try:
                    update_manifest(arm_root, manifest, status="running")
                    if "cot-to-blueprint" in completed:
                        prior = validate_cot_to_blueprint_results(arm_root, expected_ids)
                        update_manifest(
                            arm_root, manifest,
                            blueprint_validation=prior,
                            quality_warning=prior["quality_warning"],
                        )
                        if not prior["passed"]:
                            completed.difference_update(
                                {"cot-to-blueprint", "refine", "evaluate"}
                            )
                            update_manifest(
                                arm_root, manifest,
                                status="blueprint_validation_failed",
                                completed_stages=sorted(completed),
                            )
                    if "cot-to-blueprint" not in completed:
                        # Any rerun can change exported context, so downstream
                        # completion markers are invalid even if files exist.
                        completed.difference_update({"refine", "evaluate"})
                        update_manifest(
                            arm_root, manifest, completed_stages=sorted(completed),
                        )
                        ensure_model("cot-to-blueprint")
                        _run_command(
                            child_command(args.profile, "cot-to-blueprint", overrides), env, False,
                        )
                        validation = validate_cot_to_blueprint_results(
                            arm_root, expected_ids,
                        )
                        update_manifest(
                            arm_root, manifest,
                            blueprint_validation=validation,
                            quality_warning=validation["quality_warning"],
                        )
                        if not validation["passed"]:
                            raise RuntimeError(
                                "cot-to-blueprint validation failed: "
                                + json.dumps(
                                    validation["blocking_reasons"],
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                            )
                        completed.add("cot-to-blueprint")
                        update_manifest(
                            arm_root, manifest,
                            status=(
                                "blueprint_complete_with_quality_warning"
                                if validation["quality_warning"]
                                else "blueprint_complete"
                            ),
                            completed_stages=sorted(completed),
                        )

                    if should_refine(config, arm, args.refine):
                        if "refine" in completed:
                            prior_refine = validate_refine_results(
                                arm_root, expected_ids, variants,
                            )
                            update_manifest(
                                arm_root, manifest, refine_validation=prior_refine,
                            )
                            if not prior_refine["passed"]:
                                completed.difference_update({"refine", "evaluate"})
                                update_manifest(
                                    arm_root, manifest,
                                    status="refine_validation_failed",
                                    completed_stages=sorted(completed),
                                )
                        if "refine" not in completed:
                            completed.discard("evaluate")
                            update_manifest(
                                arm_root, manifest, completed_stages=sorted(completed),
                            )
                            ensure_model("refine")
                            _run_command(
                                child_command(args.profile, "refine", overrides), env, False,
                            )
                            refine_validation = validate_refine_results(
                                arm_root, expected_ids, variants,
                            )
                            update_manifest(
                                arm_root, manifest,
                                refine_validation=refine_validation,
                            )
                            if not refine_validation["passed"]:
                                raise RuntimeError(
                                    "refine validation failed: "
                                    + json.dumps(
                                        refine_validation["blocking_reasons"],
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    )
                                )
                            completed.add("refine")
                            update_manifest(
                                arm_root, manifest, status="refine_complete",
                                completed_stages=sorted(completed),
                            )
                        if "evaluate" in completed:
                            prior_evaluate = validate_evaluation_results(
                                arm_root, expected_ids, variants,
                                judge_enabled=bool(config.judge.enabled),
                            )
                            update_manifest(
                                arm_root, manifest,
                                evaluate_validation=prior_evaluate,
                            )
                            if not prior_evaluate["passed"]:
                                completed.discard("evaluate")
                                update_manifest(
                                    arm_root, manifest,
                                    status="evaluate_validation_failed",
                                    completed_stages=sorted(completed),
                                )
                        if "evaluate" not in completed:
                            if bool(config.judge.enabled):
                                ensure_model("evaluate")
                            _run_command(
                                child_command(args.profile, "evaluate", overrides), env, False,
                            )
                            evaluate_validation = validate_evaluation_results(
                                arm_root, expected_ids, variants,
                                judge_enabled=bool(config.judge.enabled),
                            )
                            update_manifest(
                                arm_root, manifest,
                                evaluate_validation=evaluate_validation,
                            )
                            if not evaluate_validation["passed"]:
                                raise RuntimeError(
                                    "evaluate validation failed: "
                                    + json.dumps(
                                        evaluate_validation["blocking_reasons"],
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    )
                                )
                            completed.add("evaluate")
                    update_manifest(
                        arm_root, manifest,
                        status=(
                            "complete_with_quality_warning"
                            if bool(manifest.get("quality_warning"))
                            else "complete"
                        ),
                        completed_at=utc_now(), completed_stages=sorted(completed),
                    )
                except Exception as exc:  # noqa: BLE001
                    failure = {"arm": arm, "error": f"{type(exc).__name__}: {exc}"}
                    failures.append(failure)
                    history = list(manifest.get("failures") or [])
                    history.append({**failure, "at": utc_now()})
                    update_manifest(arm_root, manifest, status="failed", failures=history)
                    if not args.continue_on_error:
                        raise
    except BaseException as exc:
        fatal_error = exc
        if not failures:
            failures.append({
                "arm": "_suite",
                "error": f"{type(exc).__name__}: {exc}",
            })
    finally:
        finish_suite_manifest(
            suite_manifest_path,
            execution_id=execution_id,
            failures=failures,
        )

    report_path = write_matrix_report(
        matrix_root=matrix_root,
        arms=known_arms,
        legacy_root=legacy_root,
        subset_metrics_path=subset_metrics,
        selected_ids=expected_ids,
    )
    print(f"[semantic-matrix-report] {report_path}", flush=True)
    if fatal_error is not None:
        raise fatal_error
    if failures:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    config = load_config(args.profile, args.override)
    run_id = args.run_id or str(config.matrix.default_run_id)
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run-id {run_id!r}; expected {RUN_ID_RE.pattern}")
    arms = select_arms(config, args.mode, args.arms)
    matrix_root = (
        Path(str(config.output_base)).expanduser().resolve()
        / str(config.matrix.output_prefix)
        / run_id
    )
    legacy_root = (args.legacy_root or Path(str(config.matrix.legacy_root))).resolve()
    subset_metrics = Path(str(config.matrix.subset_metrics)).expanduser().resolve()
    input_path = Path(str(config.input_predictions)).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not subset_metrics.exists():
        raise FileNotFoundError(subset_metrics)

    input_ids = _jsonl_source_ids(input_path, key="ID")
    subset_payload = _load_json(subset_metrics)
    selected_values = subset_payload.get("selected_ids")
    if not isinstance(selected_values, list) or not selected_values:
        raise ValueError(f"no selected_ids in {subset_metrics}")
    subset_ids = {str(value) for value in selected_values if str(value)}
    if len(subset_ids) != len(selected_values):
        raise ValueError(f"blank or duplicate selected_ids in {subset_metrics}")
    if input_ids != subset_ids:
        raise ValueError(
            "input predictions and subset metrics identify different samples: "
            f"missing_from_input={sorted(subset_ids - input_ids)} "
            f"unexpected_in_input={sorted(input_ids - subset_ids)}"
        )
    configured_ids = [str(value) for value in (config.include_ids or []) if str(value)]
    if len(configured_ids) != len(set(configured_ids)):
        raise ValueError("include_ids contains duplicates")
    unknown_configured = sorted(set(configured_ids) - subset_ids)
    if unknown_configured:
        raise ValueError(f"include_ids are outside the frozen wrong-76 subset: {unknown_configured}")
    expected_ids = set(configured_ids) if configured_ids else subset_ids

    input_sha256 = sha256_file(input_path)
    code_sha256, code_paths = code_fingerprint(config)
    print(
        f"[semantic-matrix] run_id={run_id} arms={arms} refine={args.refine} "
        f"selected={len(expected_ids)} selection_sha256={selection_sha256(expected_ids)} "
        f"input_sha256={input_sha256} code_sha256={code_sha256}",
        flush=True,
    )

    if args.report_only:
        output = write_matrix_report(
            matrix_root=matrix_root,
            arms=arms,
            legacy_root=legacy_root,
            subset_metrics_path=subset_metrics,
            selected_ids=expected_ids,
        )
        print(output, flush=True)
        return

    previous_handlers: dict[signal.Signals, Any] = {}

    def terminate(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, terminate)
    try:
        if args.audit_replay_manifest is not None:
            with MatrixRunLock(matrix_root):
                _execute_audit_replay(
                    config=config,
                    run_id=run_id,
                    matrix_root=matrix_root,
                    manifest_path=args.audit_replay_manifest,
                )
            return
        if args.dry_run:
            _execute_matrix(
                args=args, config=config, run_id=run_id, arms=arms,
                matrix_root=matrix_root, legacy_root=legacy_root,
                subset_metrics=subset_metrics, input_path=input_path,
                expected_ids=expected_ids, input_sha256=input_sha256,
                code_sha256=code_sha256, code_paths=code_paths,
            )
        else:
            with MatrixRunLock(matrix_root):
                _execute_matrix(
                    args=args, config=config, run_id=run_id, arms=arms,
                    matrix_root=matrix_root, legacy_root=legacy_root,
                    subset_metrics=subset_metrics, input_path=input_path,
                    expected_ids=expected_ids, input_sha256=input_sha256,
                    code_sha256=code_sha256, code_paths=code_paths,
                )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
