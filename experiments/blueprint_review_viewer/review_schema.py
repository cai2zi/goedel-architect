"""Versioned, read-only review artifact generation.

The viewer only consumes these artifacts.  The schema intentionally keeps
candidate snapshots and edit events generic so a later Repair Bundle/Subgraph
pipeline can add operation types without a front-end redesign.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from . import REVIEW_SCHEMA_VERSION


_CANDIDATE_RE = re.compile(
    r"^(?P<kind>phase1a_attempt|phase1_iter|phase1b_round|phase1b_seed|phase1b_final|phase1_failed_last)(?:_(?P<round>\d+))?\.lean$"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return value if value is not None else default


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return ""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _cot_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _json(row.get("cot_manifest_json"), {})
    steps = raw.get("steps", []) if isinstance(raw, dict) else []
    result: list[dict[str, Any]] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        result.append({
            "stepId": str(item.get("step_id", "")),
            "sourceStart": item.get("source_start"),
            "sourceEnd": item.get("source_end"),
            "sourceText": str(item.get("source_text", "")),
            "sourceSha256": str(item.get("source_sha256", "")),
        })
    return result


def _declaration_nodes(lean: str, cot_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse native Blueprint nodes if available; fall back gracefully.

    Importing the parser here means emitted artifacts carry exact source lines
    and dependencies, while the viewer itself still works for old/corrupt
    artifacts without requiring Lean or a model server.
    """
    step_by_id = {str(step["stepId"]): step for step in cot_steps}
    try:
        from blueprint import _parse_blueprint  # type: ignore

        blueprint = _parse_blueprint(lean, "root")
        result = []
        for node in blueprint.nodes:
            step = step_by_id.get(node.source_step_id, {})
            declaration = node.lean_declaration or node.full_declaration()
            result.append({
                "nodeName": node.name,
                "kind": node.kind,
                "stepId": node.source_step_id,
                "dependencies": list(node.dependencies),
                "declaration": declaration,
                "declarationSha256": _sha256(declaration),
                "leanRange": {"startLine": node.lean_start_line, "endLine": node.lean_end_line},
                "cotSource": step,
            })
        return result
    except Exception as exc:  # A broken candidate should remain reviewable.
        pattern = re.compile(r"(?m)^\s*(?:def|lemma|theorem)\s+([A-Za-z_][\w']*)")
        matches = list(pattern.finditer(lean))
        result = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(lean)
            declaration = lean[match.start():end].strip()
            start_line = lean.count("\n", 0, match.start()) + 1
            end_line = lean.count("\n", 0, end) + 1
            result.append({
                "nodeName": match.group(1), "kind": "unknown", "stepId": "",
                "dependencies": [], "declaration": declaration,
                "declarationSha256": _sha256(declaration),
                "leanRange": {"startLine": start_line, "endLine": end_line},
                "cotSource": {}, "parseWarning": str(exc),
            })
        return result


def _candidate_files(blueprint_dir: Path) -> list[Path]:
    if not blueprint_dir.is_dir():
        return []
    files = [path for path in blueprint_dir.glob("*.lean") if _CANDIDATE_RE.match(path.name)]
    rank = {"phase1b_seed": 0, "phase1a_attempt": 1, "phase1_iter": 2,
            "phase1b_round": 3, "phase1b_final": 4, "phase1_failed_last": 5}
    def sort_key(path: Path) -> tuple[int, int, str]:
        match = _CANDIDATE_RE.match(path.name)
        assert match is not None
        return (rank.get(match.group("kind"), 99), int(match.group("round") or 0), path.name)
    return sorted(files, key=sort_key)


def _candidate_id(path: Path) -> str:
    return path.stem.replace("_", "-")


def _trace_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(row.get("trace_path") or ""))
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    events.append(payload)
    except (OSError, ValueError):
        return []
    return events


def _edits(row: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge compact result history with full trace tool calls where present."""
    result: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") == "tool_call" and event.get("tool_name") == "editBlueprintNode":
            calls.append(event)
    for index, event in enumerate(calls, start=1):
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        result.append({
            "operationId": f"trace-edit-{index}",
            "operationType": "nodeEdit",
            "scope": "node",
            "round": event.get("round") or event.get("iteration"),
            "nodeName": args.get("node_name") or args.get("nodeName"),
            "action": args.get("action"),
            "reason": args.get("reason", ""),
            "expectedNodeHash": args.get("expected_node_hash", ""),
            "replacement": args.get("replacement", ""),
            "lifecycle": [{"stage": "proposed", "at": event.get("ts") or event.get("wall_time_ns")}],
            "raw": {"toolCall": event},
        })
    for history in row.get("phase1b_edit_history") or []:
        if not isinstance(history, dict):
            continue
        round_number = history.get("round")
        for bucket, stage in (("accepted", "atomicallyApplied"), ("rejected", "rejected"), ("identical", "identical")):
            for item in history.get(bucket) or []:
                if not isinstance(item, dict):
                    continue
                name = item.get("nodeName") or item.get("node_name")
                matching = next((edit for edit in result if edit.get("round") == round_number and edit.get("nodeName") == name), None)
                lifecycle = {"stage": stage, "details": item}
                if matching:
                    matching["lifecycle"].append(lifecycle)
                else:
                    result.append({"operationId": f"history-{round_number}-{bucket}-{len(result)}",
                                   "operationType": "nodeEdit", "scope": "node", "round": round_number,
                                   "nodeName": name, "action": item.get("action"), "reason": "",
                                   "lifecycle": [lifecycle], "raw": {"history": item}})
    return result


def _validation(row: dict[str, Any]) -> dict[str, Any]:
    validation = row.get("phase1b_validation")
    if isinstance(validation, dict):
        return validation
    return {}


def build_review_artifact(experiment_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    blueprint_dir = Path(str(row.get("blueprint_dir") or ""))
    cot_steps = _cot_steps(row)
    candidates: list[dict[str, Any]] = []
    for path in _candidate_files(blueprint_dir):
        text = path.read_text(encoding="utf-8")
        match = _CANDIDATE_RE.match(path.name)
        candidates.append({
            "candidateId": _candidate_id(path), "kind": match.group("kind") if match else "snapshot",
            "round": int(match.group("round")) if match and match.group("round") else None,
            "availability": "available", "leanPath": _safe_relative(path, experiment_root),
            "leanSha256": _sha256(text), "lean": text,
            "nodes": _declaration_nodes(text, cot_steps),
        })
    events = _trace_events(row)
    validation = _validation(row)
    audit = validation.get("semanticAudit") if isinstance(validation, dict) else {}
    return {
        "schemaVersion": REVIEW_SCHEMA_VERSION,
        "source": {key: row.get(key, "") for key in ("id", "record_id", "source_id", "subset", "split", "theorem_name", "claimed_answer")},
        "result": {key: row.get(key, "") for key in ("status", "phase", "success", "root_proved", "error", "semantic_status")},
        "cotSteps": cot_steps,
        "candidates": candidates,
        "edits": _edits(row, events),
        "validation": validation,
        "semanticAudit": audit if isinstance(audit, dict) else {},
        "traceSummary": {"eventCount": len(events), "tracePath": _safe_relative(Path(str(row.get("trace_path") or "")), experiment_root)},
        "futureCompatibility": {"operationTypes": ["nodeEdit", "dependencyEdit", "repairBundle", "subgraphEdit"], "readOnly": True},
    }


def write_review_artifact(experiment_root: Path, row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    artifact = build_review_artifact(experiment_root, row)
    raw_blueprint_dir = str(row.get("blueprint_dir") or "")
    if not raw_blueprint_dir:
        raise ValueError("Cannot write review artifact without blueprint_dir")
    blueprint_dir = Path(raw_blueprint_dir)
    path = blueprint_dir / "review.json"
    _atomic_json(path, artifact)
    return path, artifact


def index_entry(artifact_path: Path, experiment_root: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    source = artifact.get("source", {})
    result = artifact.get("result", {})
    return {"schemaVersion": REVIEW_SCHEMA_VERSION, "artifactPath": _safe_relative(artifact_path, experiment_root),
            "id": source.get("id", ""), "sourceId": source.get("source_id", ""),
            "subset": source.get("subset", ""), "status": result.get("status", ""),
            "candidateCount": len(artifact.get("candidates") or []),
            "updatedAtNs": os.stat(artifact_path).st_mtime_ns}
