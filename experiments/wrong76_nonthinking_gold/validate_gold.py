from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT / "src", REPO_ROOT / "experiments/stepfun_blueprint_prover"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blueprint import (  # noqa: E402
    _parse_blueprint,
    canonicalize_blueprint,
    phase2_contract_errors,
    phase2_standalone_contract_report,
)
from kimina_lean_compiler import CompileRequest, KiminaLeanCompiler  # noqa: E402
from orchestrator import active_node_names  # noqa: E402


OUTPUT_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold"
RECORDS_ROOT = OUTPUT_ROOT / "records"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _metadata_issues(record: dict[str, Any], parsed) -> list[str]:
    issues: list[str] = []
    step_ids = [str(item["step_id"]) for item in record["steps"]]
    if step_ids != [f"S{index:02d}" for index in range(1, len(step_ids) + 1)]:
        issues.append("nonContiguousStepIds")
    cot = str(record["source"]["nonthinking_cot"])
    for step in record["steps"]:
        start, end = int(step["char_start"]), int(step["char_end"])
        if cot[start:end] != step["source_span"]:
            issues.append(f"sourceSpanDrift:{step['step_id']}")
    parsed_names = [node.name for node in parsed.nodes]
    if parsed_names != list(record["nodes"]):
        issues.append("nodeOrderOrInventoryDrift")
    if set(record["labels"]) != set(parsed_names):
        issues.append("labelInventoryDrift")
    mapped: set[str] = set()
    for node in parsed.nodes:
        meta = record["nodes"].get(node.name) or {}
        label = record["labels"].get(node.name) or {}
        if meta.get("kind") != node.kind:
            issues.append(f"kindDrift:{node.name}")
        if meta.get("dependencies") != node.dependencies:
            issues.append(f"dependencyDrift:{node.name}")
        if meta.get("declaration_sha256") != _sha(node.lean_declaration):
            issues.append(f"declarationHashDrift:{node.name}")
        role = meta.get("node_role")
        source_steps = set(meta.get("source_step_ids") or [])
        if role == "problem_grounding" and not meta.get("problem_source_span"):
            issues.append(f"ungroundedProblemNode:{node.name}")
        if role in {"cot_claim", "formal_bridge"} and not source_steps:
            issues.append(f"unanchoredCotNode:{node.name}")
        unknown = source_steps - set(step_ids)
        if unknown:
            issues.append(f"unknownSourceStep:{node.name}:{','.join(sorted(unknown))}")
        mapped.update(source_steps)
        if label.get("node_role") != role:
            issues.append(f"roleLabelDrift:{node.name}")
        if label.get("source_step_ids") != meta.get("source_step_ids"):
            issues.append(f"sourceLabelDrift:{node.name}")
        if node.kind == "definition" and label.get("label") != "definition_valid":
            issues.append(f"badDefinitionLabel:{node.name}")
        if node.kind != "definition" and label.get("label") not in {
            "proved", "disproved", "blocked_by_dependency",
        }:
            issues.append(f"badProofLabel:{node.name}")
    missing_steps = set(step_ids) - mapped
    if missing_steps:
        issues.append(f"unmappedSteps:{','.join(sorted(missing_steps))}")
    target = str(record["target_theorem"])
    if target not in parsed_names or not record["nodes"].get(target, {}).get("is_target"):
        issues.append("targetMetadataDrift")
    for node in parsed.nodes:
        label = record["labels"][node.name]
        if label["label"] == "blocked_by_dependency":
            blockers = label.get("blocked_by") or []
            if not blockers:
                issues.append(f"emptyBlockedBy:{node.name}")
            for blocker in blockers:
                if blocker not in node.dependencies:
                    issues.append(f"nonDirectBlocker:{node.name}:{blocker}")
                elif record["labels"].get(blocker, {}).get("label") not in {
                    "disproved", "blocked_by_dependency",
                }:
                    issues.append(f"invalidBlockerLabel:{node.name}:{blocker}")
    return issues


def _compile_record(compiler: KiminaLeanCompiler, record_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    gold_dir = record_path.parent
    blueprint_path = gold_dir / "blueprint.lean"
    lean_code = blueprint_path.read_text(encoding="utf-8")
    target = str(record["target_theorem"])
    parsed = _parse_blueprint(lean_code, target)
    metadata_issues = _metadata_issues(record, parsed)
    contract_issues = phase2_contract_errors(parsed)
    whole = compiler.check_blueprint(lean_code, target)
    canonical_issues: list[str] = []
    canonical_result = None
    standalone = None
    try:
        canonical = canonicalize_blueprint(parsed, list(parsed.nodes))
    except Exception as exc:
        canonical = None
        canonical_issues.append(str(exc))
    if canonical is not None:
        canonical_result = compiler.check_blueprint(canonical.lean_file, target)
        if not contract_issues and canonical_result.success:
            standalone = phase2_standalone_contract_report(
                canonical, compiler, concurrency=8,
                thm_name=target, trace_phase="gold_validation",
            )
    definition_code = parsed.phase2_header.rstrip() + "\n\n" + "\n\n".join(
        node.full_declaration() for node in parsed.nodes if node.kind == "definition"
    ) + "\n"
    definition_result = compiler.check(definition_code)
    replay_items: list[tuple[str, str, Path]] = []
    requests: list[CompileRequest] = []
    for node_name, label in record["labels"].items():
        if label["label"] not in {"proved", "disproved"}:
            continue
        path = Path(str(label["complete_lean_path"]))
        code = path.read_text(encoding="utf-8")
        if label.get("lean_code_sha256") != _sha(code):
            metadata_issues.append(f"replayHashDrift:{node_name}")
        requests.append(CompileRequest(code, allow_sorry=False, request_id=f"gold-{record['record_id']}-{node_name}"))
        replay_items.append((node_name, str(label["label"]), path))
    replay_results = compiler.check_many(requests, batch_concurrency=8)
    replay_failures: list[dict[str, Any]] = []
    for (node_name, label_name, path), result in zip(replay_items, replay_results, strict=True):
        label = record["labels"][node_name]
        label["lean_verified"] = bool(result.success and not result.has_sorry)
        label["lean_warnings"] = list(result.warnings)
        if not label["lean_verified"]:
            replay_failures.append({
                "node_name": node_name,
                "label": label_name,
                "path": str(path),
                "diagnostics": result.diagnostics,
                "warnings": result.warnings,
            })
    active = active_node_names(parsed)
    active_set = set(active)
    if active_set != set(record["nodes"]):
        metadata_issues.append("notAllNodesInRootClosure")
    deterministic = {
        "passed": False,
        "parse_basic": {"passed": bool(parsed.nodes), "issues": [] if parsed.nodes else ["no nodes"]},
        "whole_file_lean": {"passed": whole.success, "issues": whole.diagnostics},
        "canonical_rebuild": {"passed": canonical is not None, "issues": canonical_issues},
        "phase2_contract": {"passed": not contract_issues, "issues": contract_issues},
        "canonical_lean": {
            "passed": canonical_result is not None and canonical_result.success,
            "issues": canonical_result.diagnostics if canonical_result is not None else ["not run"],
        },
        "phase2_standalone": {
            "passed": standalone is not None and not standalone.issues and not standalone.not_run_reason,
            "checked_nodes": len([node for node in parsed.nodes if node.kind != "definition"]),
            "issues": [issue.to_dict() for issue in standalone.issues] if standalone is not None else ["not run"],
            "not_run_reason": standalone.not_run_reason if standalone is not None else "prerequisite failed",
        },
        "definition_bundle": {
            "passed": definition_result.success and not definition_result.has_sorry,
            "issues": definition_result.diagnostics,
            "warnings": definition_result.warnings,
        },
        "metadata_contract": {"passed": not metadata_issues, "issues": metadata_issues},
    }
    deterministic["passed"] = all(
        value.get("passed") is True for key, value in deterministic.items()
        if key != "passed" and isinstance(value, dict)
    )
    record["deterministic_validation"] = deterministic
    replay_ok = not replay_failures
    step_coverage_ok = not any(issue.startswith("unmappedSteps:") for issue in metadata_issues)
    record["gold_fidelity_review"].update({
        "passed": bool(step_coverage_ok and not metadata_issues),
        "status": "author_reviewed" if step_coverage_ok and not metadata_issues else "failed_metadata_review",
        "cot_step_count": len(record["steps"]),
        "cot_steps_covered": len(record["steps"]) if step_coverage_ok else None,
    })
    counts = Counter(label["label"] for label in record["labels"].values())
    complete = deterministic["passed"] and replay_ok and record["gold_fidelity_review"]["passed"]
    record["completion"] = {
        "status": "gold_complete" if complete else "validation_failed",
        "active_node_count": len(active),
        "definition_valid_count": counts["definition_valid"],
        "proved_count": counts["proved"],
        "disproved_count": counts["disproved"],
        "blocked_count": counts["blocked_by_dependency"],
        "unlabeled_count": len(active) - sum(counts.values()),
        "all_active_nodes_labeled": len(active) == sum(counts.values()),
        "all_proof_artifacts_verified": replay_ok,
        "replay_failures": replay_failures,
        "deterministic_validation_passed": deterministic["passed"],
        "gold_fidelity_review_passed": record["gold_fidelity_review"]["passed"],
        "blueprint_path": str(blueprint_path),
        "blueprint_sha256": _sha(lean_code),
    }
    _atomic_json(record_path, record)
    return {
        "record_id": record["record_id"],
        "source_id": record["source_id"],
        "status": record["completion"]["status"],
        "metadata_issues": metadata_issues,
        "contract_issues": contract_issues,
        "whole_lean": whole.success,
        "definition_bundle": definition_result.success and not definition_result.has_sorry,
        "canonical_lean": canonical_result.success if canonical_result is not None else False,
        "standalone": standalone is not None and not standalone.issues,
        "replay_failures": replay_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-id")
    args = parser.parse_args()
    paths = sorted(RECORDS_ROOT.glob("*/gold/record.json"))
    if args.record_id:
        paths = [path for path in paths if path.parent.parent.name == args.record_id]
    compiler = KiminaLeanCompiler(
        api_url=os.environ.get("WRONG76_GOLD_LEAN_API", "http://127.0.0.1:8000"),
        timeout_s=86400, reuse=True, max_inflight_snippets=48,
        batch_size=8, global_batching=True, parallel_batches=6, batch_wait_ms=10,
    )
    try:
        results = [_compile_record(compiler, path) for path in paths]
    finally:
        compiler.close()
    summary = {
        "records": len(results),
        "gold_complete": sum(row["status"] == "gold_complete" for row in results),
        "failed": [row for row in results if row["status"] != "gold_complete"],
    }
    summary_path = OUTPUT_ROOT / "validation_summary.json"
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
