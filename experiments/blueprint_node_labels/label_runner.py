from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "experiments/stepfun_blueprint_prover"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blueprint import render_solved_declaration  # noqa: E402
from blueprint_text import BLUEPRINT_PROOF_RE, extract_current_node_decl  # noqa: E402
from input_loader import load_accepted_blueprints  # noqa: E402
from kimina_lean_compiler import CompileRequest, KiminaLeanCompiler  # noqa: E402
from node_context import build_node_problem  # noqa: E402
from orchestrator import active_node_names  # noqa: E402
from prover import _build_negation_node_decl  # noqa: E402

SOURCE_ROOT = Path(
    "/ssd/czx/czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge/"
    "robustpa/blueprint"
)
OUTPUT_ROOT = Path("/ssd/czx/czx_work/blueprint_node_labels/codex_closed_negation")
FINAL_LABELS = {"definition_valid", "proved", "disproved", "blocked_by_dependency"}

POSITIVE_CANDIDATES = [
    ("rfl", "by\n  rfl"),
    ("norm_num", "by\n  norm_num"),
    ("simp", "by\n  simp"),
    ("omega", "by\n  omega"),
    ("ring", "by\n  ring"),
    ("linarith", "by\n  linarith"),
    ("nlinarith", "by\n  nlinarith"),
    ("high_decide", "by\n  set_option maxRecDepth 100000 in\n    decide"),
    ("high_norm_num", "by\n  set_option maxRecDepth 100000 in\n    norm_num"),
]

NEGATIVE_CANDIDATES = [
    ("norm_num", "by\n  norm_num"),
    ("simp", "by\n  simp"),
    ("omega", "by\n  omega"),
    ("high_decide", "by\n  set_option maxRecDepth 100000 in\n    decide"),
    ("high_norm_num", "by\n  set_option maxRecDepth 100000 in\n    norm_num"),
]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "record"


def record_path(source) -> Path:
    return OUTPUT_ROOT / "checkpoints" / source.subset / source.split / f"{safe_name(source.record_id)}.json"


def new_record(source) -> dict[str, Any]:
    active = active_node_names(source.blueprint)
    return {
        "record_id": source.record_id,
        "source_id": source.source_id,
        "subset": source.subset,
        "split": source.split,
        "source_checkpoint": str(source.checkpoint_path),
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "target_theorem": source.blueprint.target_theorem,
        "active_nodes": [n.name for n in source.blueprint.dependency_order() if n.name in active],
        "labels": {},
    }


def load_records():
    rows = []
    for source in load_accepted_blueprints(SOURCE_ROOT):
        path = record_path(source)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["source_checkpoint_sha256"] != source.checkpoint_sha256:
                raise RuntimeError(f"source drift: {source.record_id}")
        else:
            data = new_record(source)
            atomic_json(path, data)
        rows.append((source, path, data))
    return rows


def definitions(source) -> list:
    active = set(active_node_names(source.blueprint))
    return [n for n in source.blueprint.dependency_order() if n.name in active and n.kind == "definition"]


def proof_nodes(source) -> list:
    active = set(active_node_names(source.blueprint))
    return [n for n in source.blueprint.dependency_order() if n.name in active and n.kind in {"lemma", "theorem"}]


def assemble_attempt(problem, proof_body: str, prefix_code: str = "") -> str:
    decl = extract_current_node_decl(problem.node_decl)
    decl, count = BLUEPRINT_PROOF_RE.subn(lambda _: f":= {proof_body}", decl, count=1)
    if count != 1:
        raise ValueError(f"missing proof placeholder: {problem.node_name}")
    return "\n\n".join(
        value for value in (
            problem.header.rstrip(), problem.parent_lemma_decls.strip(),
            prefix_code.strip(), decl.strip(),
        ) if value
    ) + "\n"


def verified_proofs(data: dict[str, Any]) -> dict[str, str]:
    return {
        name: row["proof_body"] for name, row in data["labels"].items()
        if row.get("label") == "proved"
    }


def direct_blockers(source, data, node) -> list[str]:
    active = set(data["active_nodes"])
    blockers = []
    for name in node.dependencies:
        dep = source.blueprint.node_by_name(name)
        if name in active and dep is not None and dep.kind in {"lemma", "theorem"}:
            label = data["labels"].get(name, {}).get("label")
            if label in {"disproved", "blocked_by_dependency"}:
                blockers.append(name)
    return blockers


def ready(source, data, node) -> bool:
    active = set(data["active_nodes"])
    for name in node.dependencies:
        dep = source.blueprint.node_by_name(name)
        if name in active and dep is not None and dep.kind in {"lemma", "theorem"}:
            if data["labels"].get(name, {}).get("label") != "proved":
                return False
    return True


def save_verified(source, path, data, node, label, proof_body, code, method, result) -> None:
    suffix = "positive" if label == "proved" else "negative"
    lean_path = OUTPUT_ROOT / "lean" / source.subset / source.split / safe_name(source.record_id) / f"{safe_name(node.name)}.{suffix}.lean"
    lean_path.parent.mkdir(parents=True, exist_ok=True)
    lean_path.write_text(code, encoding="utf-8")
    data["labels"][node.name] = {
        "kind": node.kind,
        "dependencies": list(node.dependencies),
        "label": label,
        "proof_body": proof_body,
        "negated_declaration": _build_negation_node_decl(node.lean_declaration, node.name) if label == "disproved" else "",
        "proof_method": method,
        "proof_author": "codex",
        "lean_verified": True,
        "complete_lean_path": str(lean_path),
        "lean_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "lean_warnings": result.warnings,
    }
    atomic_json(path, data)


def initialize_definitions(compiler, records) -> None:
    requests = []
    owners = []
    for source, path, data in records:
        defs = definitions(source)
        # Parsed definitions may retain the doc comment preceding the next
        # node. Remove only that trailing inter-node comment, then compile the
        # complete definition bundle with no proof placeholders or sorry.
        declarations = [
            node.full_declaration().split("\n\n/--", 1)[0].strip()
            for node in defs
        ]
        code = "\n\n".join([source.blueprint.phase2_header.rstrip(), *declarations]) + "\n"
        requests.append(CompileRequest(code))
        owners.append((source, path, data, defs, code))
    results = compiler.check_many(requests, batch_concurrency=6)
    for (source, path, data, defs, code), result in zip(owners, results, strict=True):
        if not result.success or result.has_sorry:
            raise RuntimeError(f"definition validation failed {source.record_id}: {result.diagnostics}")
        digest = hashlib.sha256(code.encode()).hexdigest()
        for node in defs:
            data["labels"][node.name] = {
                "kind": "definition", "dependencies": list(node.dependencies),
                "label": "definition_valid", "lean_verified": True,
                "definition_bundle_sha256": digest,
            }
        atomic_json(path, data)


def propagate_blocks(records) -> int:
    changed = 0
    again = True
    while again:
        again = False
        for source, path, data in records:
            for node in proof_nodes(source):
                if node.name in data["labels"]:
                    continue
                blockers = direct_blockers(source, data, node)
                if blockers:
                    data["labels"][node.name] = {
                        "kind": node.kind, "dependencies": list(node.dependencies),
                        "label": "blocked_by_dependency", "blocked_by": blockers,
                        "lean_verified": False,
                    }
                    changed += 1
                    again = True
            atomic_json(path, data)
    return changed


def attempt_candidates(compiler, records, stage: str) -> int:
    candidates = POSITIVE_CANDIDATES if stage == "positive" else NEGATIVE_CANDIDATES
    pending = []
    for source, path, data in records:
        cache = verified_proofs(data)
        for node in proof_nodes(source):
            if node.name in data["labels"] or not ready(source, data, node):
                continue
            problem = build_node_problem(source.blueprint, node.name, cache, stage=stage)
            pending.append((source, path, data, node, problem))
    solved = 0
    remaining = pending
    for method, proof in candidates:
        if not remaining:
            break
        requests = [CompileRequest(assemble_attempt(item[4], proof)) for item in remaining]
        results = compiler.check_many(requests, batch_concurrency=6)
        next_remaining = []
        for item, request, result in zip(remaining, requests, results, strict=True):
            source, path, data, node, _ = item
            if result.success and not result.has_sorry:
                save_verified(
                    source, path, data, node,
                    "proved" if stage == "positive" else "disproved",
                    proof, request.lean_code, f"verified_{method}", result,
                )
                solved += 1
            else:
                next_remaining.append(item)
        remaining = next_remaining
    return solved


def apply_manual_proofs(
    compiler, records, record_id: str | None = None, node_name: str | None = None,
    reasoning: str | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    from manual_proofs import MANUAL_PROOFS

    by_id = {source.record_id: (source, path, data) for source, path, data in records}
    accepted = 0
    failures = []
    for entry in MANUAL_PROOFS:
        if record_id is not None and entry["record_id"] != record_id:
            continue
        if node_name is not None and entry["node_name"] != node_name:
            continue
        if reasoning is not None and entry["reasoning"] != reasoning:
            continue
        source, path, data = by_id[entry["record_id"]]
        node = source.blueprint.node_by_name(entry["node_name"])
        if node is None:
            raise KeyError(f"missing node: {entry}")
        existing = data["labels"].get(node.name)
        if existing:
            continue
        if not ready(source, data, node):
            failures.append({**entry, "error": "not_ready"})
            continue
        problem = build_node_problem(
            source.blueprint, node.name, verified_proofs(data), stage=entry["stage"],
        )
        code = assemble_attempt(problem, entry["proof_body"], entry.get("prefix_code", ""))
        result = compiler.check(code)
        if result.success and not result.has_sorry:
            save_verified(
                source, path, data, node,
                "proved" if entry["stage"] == "positive" else "disproved",
                entry["proof_body"], code, entry["reasoning"], result,
            )
            accepted += 1
        else:
            failures.append({
                **entry, "error": "lean_rejected", "diagnostics": result.diagnostics,
                "failure_kind": result.failure_kind,
            })
    atomic_json(OUTPUT_ROOT / "manual_failures.json", failures)
    return accepted, failures


def summary(records) -> dict[str, Any]:
    counts = Counter()
    missing = []
    for source, _, data in records:
        for name in data["active_nodes"]:
            label = data["labels"].get(name, {}).get("label")
            if label:
                counts[label] += 1
            else:
                missing.append(f"{source.record_id}::{name}")
    value = {"blueprints": len(records), "active_nodes": sum(counts.values()) + len(missing), "counts": dict(counts), "pending_manual": missing}
    atomic_json(OUTPUT_ROOT / "progress.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--propagate", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--record-id")
    parser.add_argument("--node-name")
    parser.add_argument("--reasoning")
    args = parser.parse_args()
    records = load_records()
    compiler = KiminaLeanCompiler(
        api_url="http://127.0.0.1:8000", timeout_s=86400, reuse=True,
        max_inflight_snippets=48, batch_size=8, global_batching=True,
        parallel_batches=6, batch_wait_ms=10,
    )
    try:
        if args.initialize:
            initialize_definitions(compiler, records)
        if args.propagate:
            print({"blocked": propagate_blocks(records)}, flush=True)
        if args.manual:
            accepted, failures = apply_manual_proofs(
                compiler, records, args.record_id, args.node_name,
                args.reasoning,
            )
            print({"manual_accepted": accepted, "manual_failures": len(failures)}, flush=True)
            print({"blocked": propagate_blocks(records)}, flush=True)
        if args.sweep:
            while True:
                changed = propagate_blocks(records)
                positive = attempt_candidates(compiler, records, "positive")
                negative = attempt_candidates(compiler, records, "negative")
                print({"blocked": changed, "proved": positive, "disproved": negative}, flush=True)
                if not (changed or positive or negative):
                    break
    finally:
        compiler.close()
    print(json.dumps(summary(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
