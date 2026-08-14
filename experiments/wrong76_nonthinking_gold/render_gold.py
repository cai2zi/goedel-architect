from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blueprint import _parse_blueprint, render_solved_declaration  # noqa: E402
from blueprint_text import BLUEPRINT_PROOF_RE, extract_current_node_decl  # noqa: E402
from gold_dsl import Case, Node  # noqa: E402
from prover import _build_negation_node_decl  # noqa: E402


OUTPUT_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold"
BLIND_PATH = OUTPUT_ROOT / "blind_inputs.jsonl"
RECORDS_ROOT = OUTPUT_ROOT / "records"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _blind_index() -> dict[str, dict[str, Any]]:
    rows = [json.loads(raw) for raw in BLIND_PATH.read_text(encoding="utf-8").splitlines() if raw]
    return {str(row["source_id"]): row for row in rows}


def _escape_comment(value: str) -> str:
    return value.replace("-/", "- / ").strip()


def _skeleton_node(node: Node) -> str:
    statement = _escape_comment(node.statement or node.name.replace("_", " "))
    attrs = [f"(statement := /-- {statement} -/)" ]
    if node.kind != "definition":
        proof = _escape_comment(node.proof_sketch or "Gold proof is stored in the replay artifact")
        attrs.append(f"(proof := /-- {proof} -/)")
    attr = "@[blueprint " + " ".join(attrs) + "]"
    declaration = node.declaration.strip()
    if node.kind == "definition":
        return f"{attr}\n{declaration}"
    deps = ", ".join(node.dependencies)
    return f"{attr}\n{declaration} := by sorry_using [{deps}]"


def _skeleton(case: Case) -> str:
    return case.header.rstrip() + "\n\n" + "\n\n".join(_skeleton_node(node) for node in case.nodes) + "\n"


def _replace_proof(declaration: str, proof: str) -> str:
    current = extract_current_node_decl(declaration)
    value, count = BLUEPRINT_PROOF_RE.subn(lambda _: f":= {proof.strip()}", current, count=1)
    if count != 1:
        raise ValueError("could not replace proof")
    return value.strip()


def _replay_code(case: Case, node: Node, parsed, proved: dict[str, str]) -> tuple[str, str]:
    node_map = {item.name: item for item in parsed.nodes}
    declarations: list[str] = []
    for spec in case.nodes:
        parsed_node = node_map[spec.name]
        if spec.kind == "definition":
            declarations.append(parsed_node.full_declaration())
    needed: set[str] = set()

    def visit(name: str) -> None:
        if name in needed:
            return
        spec = next(item for item in case.nodes if item.name == name)
        for dep in spec.dependencies:
            dep_spec = next(item for item in case.nodes if item.name == dep)
            if dep_spec.kind != "definition":
                visit(dep)
        if spec.kind != "definition":
            needed.add(name)

    for dep in node.dependencies:
        dep_spec = next(item for item in case.nodes if item.name == dep)
        if dep_spec.kind != "definition":
            visit(dep)
    for spec in case.nodes:
        if spec.name in needed:
            if spec.label != "proved":
                raise ValueError(f"non-proved dependency {spec.name} used by {node.name}")
            declarations.append(render_solved_declaration(node_map[spec.name], proved[spec.name]))
    parsed_node = node_map[node.name]
    if node.label == "proved":
        target_decl = f"{parsed_node.signature()} := {node.proof.strip()}"
        negated = ""
    elif node.label == "disproved":
        raw_negated = _build_negation_node_decl(parsed_node.lean_declaration, node.name)
        target_decl = _replace_proof(raw_negated, node.proof)
        negated = extract_current_node_decl(raw_negated).strip()
    else:
        raise ValueError(f"no replay for label {node.label}")
    code = case.header.rstrip() + "\n\n" + "\n\n".join([*declarations, target_decl]) + "\n"
    return code, negated


def _exact_steps(case: Case, cot: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    cursor = 0
    for index, step in enumerate(case.steps, start=1):
        start = cot.find(step.source_span, cursor)
        if start < 0:
            raise ValueError(f"source span S{index:02d} not found in {case.source_id}")
        end = start + len(step.source_span)
        steps.append({
            "step_id": f"S{index:02d}",
            "ordinal": index,
            "source_span": step.source_span,
            "char_start": start,
            "char_end": end,
        })
        cursor = end
    return steps


def render_case(case: Case, blind: dict[str, Any]) -> Path:
    record_dir = RECORDS_ROOT / _safe(str(blind["record_id"]))
    gold_dir = record_dir / "gold"
    lean_dir = gold_dir / "lean"
    lean_dir.mkdir(parents=True, exist_ok=True)
    skeleton = _skeleton(case)
    blueprint_path = gold_dir / "blueprint.lean"
    blueprint_path.write_text(skeleton, encoding="utf-8")
    parsed = _parse_blueprint(skeleton, case.target_name)
    if [node.name for node in parsed.nodes] != [node.name for node in case.nodes]:
        raise ValueError(f"parsed nodes drift for {case.source_id}")
    proved = {node.name: node.proof for node in case.nodes if node.label == "proved"}
    metadata: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    for order, (spec, node) in enumerate(zip(case.nodes, parsed.nodes, strict=True)):
        source_ids = [f"S{index:02d}" for index in spec.source_steps]
        metadata[spec.name] = {
            "order": order,
            "kind": spec.kind,
            "node_role": spec.role,
            "source_step_ids": source_ids,
            "primary_source_step_id": source_ids[0] if source_ids else "",
            "problem_source_span": spec.problem_source_span,
            "dependencies": list(spec.dependencies),
            "is_target": spec.name == case.target_name,
            "in_root_closure": True,
            "lean_declaration": node.lean_declaration,
            "declaration_sha256": _sha(node.lean_declaration),
        }
        label: dict[str, Any] = {
            "kind": spec.kind,
            "node_role": spec.role,
            "source_step_ids": source_ids,
            "dependencies": list(spec.dependencies),
            "label": spec.label,
            "lean_verified": False,
        }
        if spec.label in {"proved", "disproved"}:
            code, negated = _replay_code(case, spec, parsed, proved)
            suffix = "positive" if spec.label == "proved" else "negative"
            path = lean_dir / f"{_safe(spec.name)}.{suffix}.lean"
            path.write_text(code, encoding="utf-8")
            label.update({
                "proof_body": spec.proof,
                "negated_declaration": negated if spec.label == "disproved" else "",
                "proof_method": spec.proof_method,
                "complete_lean_path": str(path),
                "lean_code_sha256": _sha(code),
            })
        elif spec.label == "blocked_by_dependency":
            label["blocked_by"] = [
                dep for dep in spec.dependencies
                if next(item for item in case.nodes if item.name == dep).label
                in {"disproved", "blocked_by_dependency"}
            ]
            if not label["blocked_by"]:
                raise ValueError(f"blocked node has no blocker: {spec.name}")
        labels[spec.name] = label
    source = {
        key: blind[key]
        for key in (
            "problem", "nonthinking_cot", "claimed_answer", "extraction_rule",
            "problem_sha256", "nonthinking_cot_sha256", "source_line_sha256",
        )
    }
    record = {
        "schema_version": "wrong76_gold_v1",
        "record_id": blind["record_id"],
        "source_id": blind["source_id"],
        "subset": blind["subset"],
        "split": blind["split"],
        "source": source,
        "steps": _exact_steps(case, str(blind["nonthinking_cot"])),
        "target_theorem": case.target_name,
        "nodes": metadata,
        "labels": labels,
        "deterministic_validation": {"passed": False, "status": "not_run"},
        "gold_fidelity_review": {
            "passed": False,
            "status": "author_asserted_pending_independent_replay",
            "review_notes": list(case.fidelity_notes),
        },
        "completion": {"status": "authored_pending_validation"},
    }
    record_path = gold_dir / "record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", action="append", required=True)
    args = parser.parse_args()
    blind = _blind_index()
    rendered = 0
    for module_name in args.module:
        module = importlib.import_module(module_name)
        cases: tuple[Case, ...] = module.CASES
        for case in cases:
            render_case(case, blind[case.source_id])
            rendered += 1
    print(f"rendered={rendered}")


if __name__ == "__main__":
    main()
