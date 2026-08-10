from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from blueprint import _parse_blueprint  # noqa: E402
from llm_client import make_client  # noqa: E402
from semantic_audit import (  # noqa: E402
    SemanticAuditFormatError,
    build_formal_view,
    comparator_defects,
    run_formal_decompiler,
    run_strict_comparator,
)
from semantic_fidelity import parse_cot_manifest, validate_blueprint_fidelity  # noqa: E402
from tracer import JsonlTracer  # noqa: E402


DEFAULT_V5_ROOT = Path(
    "/ssd/czx/czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_step_v5_phase1_ab_semantic_judge"
)
DEFAULT_OUTPUT = Path(
    "/ssd/czx/czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_step_v6_semantic_audit44"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepared_rows(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "prepared" / "data").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            rows[str(row.get("name") or "")] = row
    return rows


def _audit_one(
    row: dict[str, Any],
    prepared: dict[str, Any],
    *,
    model: str,
    output: Path,
    decompiler_max_tokens: int,
    comparator_max_tokens: int,
    format_attempts: int,
) -> dict[str, Any]:
    source_id = str(row["source_id"])
    trace_path = output / "traces" / str(row["subset"]) / str(row["split"]) / f"{row['record_id']}.jsonl"
    artifact_path = output / "audits" / str(row["subset"]) / str(row["split"]) / f"{row['record_id']}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    tracer = JsonlTracer(trace_path)
    try:
        lean_path = Path(str(row["blueprint_dir"])) / "phase1b_final.lean"
        lean_code = lean_path.read_text(encoding="utf-8")
        blueprint = _parse_blueprint(lean_code, str(row["theorem_name"]))
        manifest = parse_cot_manifest(str(prepared["cot_manifest_json"]))
        static_issues = validate_blueprint_fidelity(
            blueprint,
            manifest,
            claimed_answer=str(prepared.get("claimed_answer") or ""),
            require_step_bindings=True,
            allow_pending_claims=False,
        )
        client = make_client(model)
        view = build_formal_view(blueprint)
        decompiler = run_formal_decompiler(
            client,
            model,
            view=view,
            max_tokens=decompiler_max_tokens,
            max_attempts=format_attempts,
            tracer=tracer,
            thm_name=source_id,
            round_index=0,
        )
        comparator = run_strict_comparator(
            client,
            model,
            informal_statement=str(prepared.get("informal_statement") or ""),
            claimed_answer=str(prepared.get("claimed_answer") or ""),
            manifest=manifest,
            view=view,
            decompiler=decompiler,
            open_obligations=[],
            max_tokens=comparator_max_tokens,
            max_attempts=format_attempts,
            tracer=tracer,
            thm_name=source_id,
            round_index=0,
        )
        static_errors = [issue.to_dict() for issue in static_issues if issue.severity == "error"]
        warnings = [issue.to_dict() for issue in static_issues if issue.severity == "warning"]
        passed = not static_errors and comparator.passed
        status = (
            "strictAccepted" if passed and not warnings else
            "acceptedWithJustifiedSideBranches" if passed else
            "semanticRejected"
        )
        artifact = {
            "source_id": source_id,
            "status": status,
            "formalView": view.to_dict(),
            "formalDecompiler": decompiler.to_dict(),
            "strictComparator": comparator.to_dict(),
            "defects": comparator_defects(comparator),
            "staticErrors": static_errors,
            "staticWarnings": warnings,
        }
        _write_json(artifact_path, artifact)
        return {
            "source_id": source_id,
            "record_id": row["record_id"],
            "subset": row["subset"],
            "split": row["split"],
            "v5_status": row["status"],
            "status": status,
            "static_error_count": len(static_errors),
            "static_warning_count": len(warnings),
            "vacuous_node_count": len(decompiler.vacuous_nodes),
            "comparator_defect_count": len(comparator_defects(comparator)),
            "prompt_tokens": decompiler.prompt_tokens + comparator.prompt_tokens,
            "completion_tokens": decompiler.completion_tokens + comparator.completion_tokens,
            "total_tokens": decompiler.total_tokens + comparator.total_tokens,
            "decompiler_finish_reason": decompiler.finish_reason,
            "comparator_finish_reason": comparator.finish_reason,
            "artifact_path": str(artifact_path),
            "trace_path": str(trace_path),
            "error": "",
        }
    except SemanticAuditFormatError as exc:
        return {
            "source_id": source_id, "record_id": row["record_id"],
            "subset": row["subset"], "split": row["split"],
            "v5_status": row["status"], "status": "infraError",
            "error": f"semantic audit format: {exc.reason}",
            "attempts": list(exc.attempts), "trace_path": str(trace_path),
        }
    except Exception as exc:
        return {
            "source_id": source_id, "record_id": row["record_id"],
            "subset": row["subset"], "split": row["split"],
            "v5_status": row["status"], "status": "infraError",
            "error": f"{type(exc).__name__}: {exc}", "trace_path": str(trace_path),
        }
    finally:
        tracer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-root", type=Path, default=DEFAULT_V5_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--formal-decompiler-max-tokens", type=int, default=4096)
    parser.add_argument("--strict-comparator-max-tokens", type=int, default=4096)
    parser.add_argument("--format-max-attempts", type=int, default=2)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--retry-infra", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    import os
    os.environ["GOEDEL_OPENAI_BASE_URL"] = args.base_url.rstrip("/")
    os.environ.setdefault("GOEDEL_OPENAI_API_KEY", "dummy")

    result_rows = _read_jsonl(args.v5_root / "robustpa" / "blueprint" / "results.jsonl")
    accepted = [row for row in result_rows if row.get("status") == "phase1_accepted"]
    existing_results = _read_jsonl(args.output / "results.jsonl") if (
        args.retry_infra and (args.output / "results.jsonl").exists()
    ) else []
    if args.source_id:
        selected = set(args.source_id)
        accepted = [row for row in accepted if row.get("source_id") in selected]
        missing = selected - {str(row.get("source_id")) for row in accepted}
        if missing:
            raise RuntimeError(f"requested source IDs are not v5 accepted: {sorted(missing)}")
    elif len(accepted) != 44:
        raise RuntimeError(f"expected 44 v5 accepted rows, found {len(accepted)}")
    if args.retry_infra:
        retry_ids = {
            str(row["source_id"]) for row in existing_results
            if row.get("status") == "infraError"
        }
        accepted = [row for row in accepted if row.get("source_id") in retry_ids]
    prepared = _prepared_rows(args.v5_root)
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = [
        row for row in existing_results
        if str(row.get("source_id")) not in {str(item.get("source_id")) for item in accepted}
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _audit_one,
                row,
                prepared[str(row["source_id"])],
                model=args.model,
                output=args.output,
                decompiler_max_tokens=args.formal_decompiler_max_tokens,
                comparator_max_tokens=args.strict_comparator_max_tokens,
                format_attempts=args.format_max_attempts,
            ): row
            for row in accepted
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[audit-{result['status']}] {result['source_id']}", flush=True)
    results.sort(key=lambda item: item["source_id"])
    results_path = args.output / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {"total": len(results), "statusCounts": counts,
               "infraErrors": [row["source_id"] for row in results if row["status"] == "infraError"]}
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
