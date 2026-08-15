from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint import Blueprint, BlueprintNode, render_solved_declaration  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from prover import _build_negation_node_decl  # noqa: E402


DEFAULT_GOLD_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold"
DEFAULT_SEED_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold_phase2_seed"
SIGNAL_TO_LABEL = {
    "solved": "proved",
    "formally_negated": "disproved",
    "blocked_by_dependency": "blocked_by_dependency",
    "proof_too_hard": "proof_too_hard",
    "protocol_error": "protocol_error",
    "infra_error": "infra_error",
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return result or "node"


def _latest_by_source(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["source_id"]): row
        for row in rows
        if str(row.get("source_id") or "")
    }


def _transitive_dependencies(node: BlueprintNode, blueprint: Blueprint) -> set[str]:
    seen: set[str] = set()
    stack = list(node.dependencies)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        dependency = blueprint.node_by_name(name)
        if dependency is not None:
            stack.extend(dependency.dependencies)
    return seen


def _proof_artifact(
    *,
    blueprint: Blueprint,
    node: BlueprintNode,
    signal: str,
    proof_body: str,
    node_results: dict[str, dict[str, Any]],
) -> str:
    definitions = [
        candidate.full_declaration()
        for candidate in blueprint.nodes
        if candidate.kind == "definition"
    ]
    ancestors = _transitive_dependencies(node, blueprint)
    proved_ancestors = [
        candidate
        for candidate in blueprint.dependency_order()
        if candidate.kind != "definition"
        and candidate.name in ancestors
        and str((node_results.get(candidate.name) or {}).get("signal")) == "solved"
    ]
    context = definitions + [
        render_solved_declaration(
            ancestor,
            str(node_results[ancestor.name]["proof_body"]),
        )
        for ancestor in proved_ancestors
    ]
    if signal == "solved":
        current = render_solved_declaration(node, proof_body)
    elif signal == "formally_negated":
        template = _build_negation_node_decl(node.lean_declaration, node.name)
        current, replacements = re.subn(
            r"by\s+sorry_using\s*\[\s*\]\s*$", proof_body, template, count=1
        )
        if replacements != 1:
            raise RuntimeError(f"could not materialize negative proof for {node.name}")
    else:
        raise ValueError(signal)
    return "\n\n".join([
        blueprint.phase2_header.strip(),
        *context,
        current,
    ]).strip() + "\n"


def _actual_label(kind: str, signal: str) -> str:
    if kind == "definition" and signal == "solved":
        return "definition_valid"
    return SIGNAL_TO_LABEL.get(signal, signal or "missing_result")


def evaluate(experiment_root: Path, gold_root: Path, seed_root: Path) -> dict[str, Any]:
    robustpa_root = experiment_root / "robustpa" / "blueprint"
    results_path = robustpa_root / "results.jsonl"
    mapping_path = seed_root / "target_mapping.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)

    results = _latest_by_source(_read_jsonl(results_path))
    mappings = {
        str(row["source_id"]): row
        for row in _read_json(mapping_path)["records"]
    }
    unknown = sorted(set(results) - set(mappings))
    if unknown:
        raise RuntimeError(f"Phase 2 results contain unknown source IDs: {unknown}")

    evaluation_root = experiment_root / "gold_evaluation"
    model_lean_root = evaluation_root / "model_lean"
    node_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    actual_signal_counts: Counter[str] = Counter()
    result_status_counts: Counter[str] = Counter()

    for source_id, result_row in sorted(results.items()):
        mapping = mappings[source_id]
        record_id = str(mapping["record_id"])
        split = str(mapping["split"])
        subset = str(mapping["subset"])
        checkpoint_path = (
            robustpa_root / "checkpoints" / subset / split / f"{record_id}.json"
        )
        trace_path = robustpa_root / "traces" / subset / split / f"{record_id}.jsonl"
        gold_record_path = gold_root / "records" / record_id / "gold" / "record.json"
        gold_record = _read_json(gold_record_path)
        result_status_counts[str(result_row.get("status") or "missing")] += 1

        if not checkpoint_path.is_file():
            state = None
            blueprint = None
            runtime_node_results: dict[str, dict[str, Any]] = {}
        else:
            state = CheckpointState.load(checkpoint_path)
            blueprint = state.get_blueprint()
            if blueprint is None:
                raise RuntimeError(f"Phase 2 checkpoint has no Blueprint: {checkpoint_path}")
            runtime_node_results = state.node_results

        local_total = 0
        local_correct = 0
        root_correct = False
        for gold_name, metadata in sorted(
            gold_record["nodes"].items(), key=lambda item: int(item[1]["order"])
        ):
            expected = str(gold_record["labels"][gold_name]["label"])
            runtime_name = (
                str(mapping["runtime_target"])
                if gold_name == str(mapping["gold_target"])
                else gold_name
            )
            result = runtime_node_results.get(runtime_name) or {}
            signal = str(result.get("signal") or "missing_result")
            kind = str(metadata["kind"])
            actual = _actual_label(kind, signal)
            correct = actual == expected
            confusion[expected][actual] += 1
            actual_signal_counts[signal] += 1
            local_total += 1
            local_correct += int(correct)

            proof_body = str(result.get("proof_body") or "")
            model_lean_path = ""
            lean_verified = False
            lean_code_sha256 = ""
            if blueprint is not None and signal in {"solved", "formally_negated"} and proof_body:
                node = blueprint.node_by_name(runtime_name)
                if node is None:
                    raise RuntimeError(f"runtime Blueprint is missing node {runtime_name}")
                lean_code = _proof_artifact(
                    blueprint=blueprint,
                    node=node,
                    signal=signal,
                    proof_body=proof_body,
                    node_results=runtime_node_results,
                )
                suffix = "positive" if signal == "solved" else "negative"
                path = model_lean_root / record_id / f"{_safe_name(gold_name)}.{suffix}.lean"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(lean_code, encoding="utf-8")
                model_lean_path = str(path)
                lean_verified = True
                lean_code_sha256 = _sha_text(lean_code)

            is_target = gold_name == str(gold_record["target_theorem"])
            if is_target:
                root_correct = correct
            node_rows.append({
                "source_id": source_id,
                "record_id": record_id,
                "node_name": gold_name,
                "runtime_node_name": runtime_name,
                "kind": kind,
                "node_role": str(metadata["node_role"]),
                "source_step_ids": list(metadata["source_step_ids"]),
                "dependencies": list(metadata["dependencies"]),
                "is_target": is_target,
                "expected_label": expected,
                "actual_signal": signal,
                "actual_label": actual,
                "correct": correct,
                "proof_body": proof_body,
                "lean_errors": list(result.get("lean_errors") or []),
                "lean_verified_by_phase2": lean_verified,
                "model_lean_path": model_lean_path,
                "lean_code_sha256": lean_code_sha256,
                "checkpoint_path": str(checkpoint_path),
                "trace_path": str(trace_path),
                "gold_record_path": str(gold_record_path),
            })

        record_rows.append({
            "source_id": source_id,
            "record_id": record_id,
            "phase2_status": str(result_row.get("status") or "missing"),
            "nodes": local_total,
            "correct_nodes": local_correct,
            "all_nodes_correct": local_correct == local_total,
            "root_correct": root_correct,
            "checkpoint_path": str(checkpoint_path),
            "trace_path": str(trace_path),
        })

    expected_counts = Counter(row["expected_label"] for row in node_rows)
    correct_counts = Counter(
        row["expected_label"] for row in node_rows if row["correct"]
    )

    def metric(label: str) -> dict[str, Any]:
        denominator = expected_counts[label]
        numerator = correct_counts[label]
        return {
            "correct": numerator,
            "total": denominator,
            "rate": numerator / denominator if denominator else None,
        }

    total_correct = sum(bool(row["correct"]) for row in node_rows)
    root_total = sum(bool(row["is_target"]) for row in node_rows)
    root_correct = sum(bool(row["correct"]) for row in node_rows if row["is_target"])
    summary = {
        "schema_version": "wrong76_gold_phase2_evaluation_v1",
        "experiment_root": str(experiment_root),
        "seed_root": str(seed_root),
        "gold_root": str(gold_root),
        "record_count": len(record_rows),
        "node_count": len(node_rows),
        "phase2_status_counts": dict(sorted(result_status_counts.items())),
        "actual_signal_counts": dict(sorted(actual_signal_counts.items())),
        "node_label_accuracy": {
            "correct": total_correct,
            "total": len(node_rows),
            "rate": total_correct / len(node_rows) if node_rows else None,
        },
        "positive_proof_accuracy": metric("proved"),
        "negative_proof_accuracy": metric("disproved"),
        "blocked_dependency_accuracy": metric("blocked_by_dependency"),
        "definition_accuracy": metric("definition_valid"),
        "root_label_accuracy": {
            "correct": root_correct,
            "total": root_total,
            "rate": root_correct / root_total if root_total else None,
        },
        "exact_record_accuracy": {
            "correct": sum(bool(row["all_nodes_correct"]) for row in record_rows),
            "total": len(record_rows),
        },
        "confusion": {
            expected: dict(sorted(actual.items()))
            for expected, actual in sorted(confusion.items())
        },
        "infrastructure_or_protocol_nodes": sum(
            actual_signal_counts[name]
            for name in ("infra_error", "protocol_error", "missing_result")
        ),
        "node_results_path": str(evaluation_root / "node_results.jsonl"),
        "record_results_path": str(evaluation_root / "record_results.jsonl"),
        "model_lean_root": str(model_lean_root),
    }
    _write_jsonl(evaluation_root / "node_results.jsonl", node_rows)
    _write_jsonl(evaluation_root / "record_results.jsonl", record_rows)
    _write_json(evaluation_root / "summary.json", summary)
    print(
        "[gold-eval] "
        f"records={summary['record_count']} nodes={summary['node_count']} "
        f"correct={total_correct}/{len(node_rows)} "
        f"root={root_correct}/{root_total} output={evaluation_root}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Phase 2 node signals against Wrong76 Gold.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        args.experiment_root.expanduser().resolve(),
        args.gold_root.expanduser().resolve(),
        args.seed_root.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
