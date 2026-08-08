from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "experiments")]

from blueprint import _render_step_grounded_proof  # noqa: E402
from semantic_audit import run_semantic_audit  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _latest(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["source_id"]): row for row in _read_jsonl(path)}


def _candidate_path(result: dict[str, Any]) -> Path:
    directory = Path(str(result["blueprint_dir"]))
    repaired = directory / "phase1_iter1.lean"
    return repaired if repaired.exists() else directory / "phase1_iter0.lean"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a full 397B Step-fidelity audit over every final Phase-1 candidate."
    )
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--model", default="Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.max_tokens <= 0:
        raise ValueError("concurrency and max-tokens must be positive")

    root = args.experiment_root.resolve()
    generation_rows = {
        str(row["name"]): row
        for row in _read_jsonl(root / "prepared" / "generation_inputs.jsonl")
    }
    result_rows = _read_jsonl(root / "robustpa" / "blueprint" / "results.jsonl")
    if len(generation_rows) != len(result_rows):
        raise RuntimeError(
            f"experiment is incomplete: inputs={len(generation_rows)} results={len(result_rows)}"
        )

    output = root / "analysis" / "semantic_audit_all.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    cached = _latest(output)
    pending = [
        row for row in result_rows
        if cached.get(str(row["source_id"]), {}).get("status") != "ok"
    ]
    os.environ["GOEDEL_OPENAI_BASE_URL"] = args.base_url.rstrip("/")
    os.environ.setdefault("GOEDEL_OPENAI_API_KEY", "dummy")
    lock = threading.Lock()

    def audit(result: dict[str, Any]) -> dict[str, Any]:
        source_id = str(result["source_id"])
        source = generation_rows[source_id]
        candidate_path = _candidate_path(result)
        candidate = candidate_path.read_text(encoding="utf-8")
        try:
            audited = run_semantic_audit(
                args.model,
                _render_step_grounded_proof(
                    str(source["cot_manifest_json"]), include_ir=False
                ),
                candidate,
                mode="full",
                informal_statement=str(source["informal_statement"]),
                claimed_answer=str(source["claimed_answer"]),
                max_tokens=args.max_tokens,
                thm_name=str(result["id"]),
                phase="offline_semantic_audit_all",
            )
            return {
                "source_id": source_id,
                "status": "ok",
                "candidate_path": str(candidate_path),
                "pipeline_status": result["status"],
                "pipeline_failure_stage": result.get("failed_blueprint_failure_stage", ""),
                **asdict(audited),
            }
        except Exception as exc:  # Preserve format/API failures for explicit retry.
            return {
                "source_id": source_id,
                "status": "error",
                "candidate_path": str(candidate_path),
                "pipeline_status": result["status"],
                "pipeline_failure_stage": result.get("failed_blueprint_failure_stage", ""),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_content": str(getattr(exc, "raw_content", "")),
            }

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(audit, row): row for row in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            completed.append(row)
            with lock, output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[audit {index}/{len(pending)}] {row['source_id']} "
                f"status={row['status']} flag={row.get('flag', '')}",
                flush=True,
            )

    latest = _latest(output)
    rows = [latest[source_id] for source_id in sorted(generation_rows) if source_id in latest]
    summary = {
        "rows": len(rows),
        "ok": sum(row.get("status") == "ok" for row in rows),
        "format_or_api_errors": sum(row.get("status") != "ok" for row in rows),
        "passed": sum(row.get("passed") is True for row in rows),
        "failed": sum(row.get("passed") is False for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
