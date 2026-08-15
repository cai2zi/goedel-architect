from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_BASE_DIR = Path("/ssd/czx/czx_work/cot_blueprint_refine")
DEFAULT_STATUSES = ("semanticRejected",)
SCHEMA_VERSION = "semantic-review-bundle-v1"


CHATGPT_PROMPT = r"""# Task: independently audit COT-to-Blueprint semantic fidelity

I attached `semantic_review_inputs.parquet`. Load the complete Parquet file and
audit every row. Do not treat `pipeline_status` as proof that the candidate is
wrong, and do not reuse or assume any taxonomy from a previous comparator. You
must infer the failure taxonomy from the supplied data itself.

Each row contains:

- `source_id`: stable sample identifier;
- `problem`: original problem statement;
- `claimed_answer`: answer claimed by the source response;
- `cot`: complete source chain-of-thought to be preserved;
- `final_blueprint`: the final mechanically valid Lean Blueprint submitted to
  semantic audit;
- provenance columns identifying the experiment, audit mode, Blueprint file,
  hash, generation attempt, and semantic-audit ordinal.

## Audit semantics

This is a semantic-translation audit, not a truth judgment. A mathematically
incorrect COT should still be judged faithful when the Blueprint expresses that
same incorrect reasoning and conclusion. Conversely, Lean compilation, a
plausible final number, matching identifier names, and comments do not establish
semantic fidelity.

Read Lean declarations literally. Treat definitions as global context available
to all proof nodes. Treat lemma/theorem declarations as formal propositions;
`sorry` hides proofs but does not change their statement. `sorry_using [...]`
records the explicit proof-to-proof dependency graph. A definition does not need
to occur in `sorry_using`. Identifier names and comments carry no semantic credit.

For every row, independently determine:

1. What target object, source conditions, quantifier/answer scope, and material
   reasoning chain the COT requires.
2. What the final Blueprint literally defines and claims.
3. Whether any source clause is omitted, weakened, strengthened, attached to the
   wrong object, represented by the wrong relation/direction, or supplemented by
   an unsupported extra condition.
4. Whether every important source object is formally bound to the intended
   object rather than merely given a suggestive name.
5. Whether the root preserves the requested target object and whether its answer
   is grounded in source conditions instead of being fixed in a definition,
   assumed in a hypothesis, reduced to a tautology, or verified after the fact.
6. Whether a materially required proposition exists but its explicit proof
   dependency chain fails to reach the root. Do not reject harmless verification,
   abandoned branches, or explanatory side branches merely for being unreachable.

## Derive the taxonomy from the data

First finish the per-row audits. Then induce a compact set of root-cause
categories from the observed failures. Do not start from a predefined list of
category names. Categories must be mutually distinguishable by a practical
counterfactual test: explain what edit would fix that category and why that edit
would not by itself fix the neighboring categories.

One root cause may affect several nodes. One sample may contain several
independent root causes and may therefore have one primary category plus
secondary categories. Merge duplicate symptoms caused by the same underlying
defect. If the evidence is ambiguous, use `uncertain` and state what is missing.

## Required deliverables

Produce both `semantic_audit_analysis.md` and `semantic_audit_analysis.json` as
downloadable files, and summarize their key findings in the chat.

The Markdown report must contain:

1. Dataset integrity: row count, unique `source_id` count, experiment name(s),
   status distribution, and confirmation that every Blueprint hash matches the
   loaded text.
2. Induced taxonomy: category name, precise definition, decision boundary,
   minimal repair, number of affected samples, and all affected IDs.
3. Per-sample audit: `source_id`, independent verdict (`faithful`, `unfaithful`,
   or `uncertain`), concise source contract, literal formal behavior, primary
   category, secondary categories, affected node names, and evidence-backed
   reason.
4. For every category, three detailed representative examples when at least
   three exist; otherwise analyze all available examples. Quote short Lean
   fragments and explain the semantic direction of the error.
5. Overlap analysis: which categories co-occur, which apparent overlaps are one
   root cause, and which are genuinely independent.
6. Possible false rejects: candidates that appear faithful under independent
   inspection, with reasons.
7. Builder-oriented conclusions: the smallest prompt or representation changes
   suggested by the recurring root causes. Keep these recommendations separate
   from the audit evidence.

The JSON file must have this shape (category names are yours to induce):

```json
{
  "dataset": {
    "rows": 0,
    "unique_source_ids": 0,
    "experiments": [],
    "blueprint_hashes_verified": true
  },
  "categories": [
    {
      "category": "induced name",
      "definition": "...",
      "decision_boundary": "...",
      "minimal_repair": "...",
      "sample_count": 0,
      "source_ids": []
    }
  ],
  "samples": [
    {
      "source_id": "...",
      "verdict": "faithful|unfaithful|uncertain",
      "source_contract": "...",
      "formal_behavior": "...",
      "primary_category": "...",
      "secondary_categories": [],
      "node_names": [],
      "reason": "...",
      "confidence": "high|medium|low"
    }
  ],
  "possible_false_reject_ids": [],
  "cross_sample_conclusions": []
}
```

Do not silently sample rows. If you cannot inspect every row, report exactly
which IDs remain unaudited and do not present partial counts as complete.
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _latest_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            result[value] = row
    return result


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _last_completed_semantic_attempt(row: dict[str, Any]) -> dict[str, Any] | None:
    for item in reversed(row.get("generation_history") or []):
        if not isinstance(item, dict):
            continue
        validation = item.get("validation") or {}
        invoked = bool(
            item.get("semanticAuditInvoked")
            or (isinstance(validation, dict) and validation.get("semanticAuditInvoked"))
        )
        if not invoked:
            continue
        if isinstance(validation, dict) and validation.get("semanticAuditError"):
            continue
        return item
    return None


def _candidate_paths(
    experiment_root: Path,
    row: dict[str, Any],
    semantic_attempt: dict[str, Any] | None,
) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []

    explicit = str(row.get("failed_blueprint_candidate_path") or "")
    if explicit:
        candidates.append((Path(explicit), "failed_blueprint_candidate_path"))

    blueprint_dir_text = str(row.get("blueprint_dir") or "")
    blueprint_dir = Path(blueprint_dir_text) if blueprint_dir_text else None
    if blueprint_dir is not None:
        candidates.append((blueprint_dir / "phase1_failed_last.lean", "phase1_failed_last"))
        if semantic_attempt is not None:
            attempt_index = int(semantic_attempt.get("round") or 0)
            if attempt_index > 0:
                candidates.extend([
                    (
                        blueprint_dir / f"generation_round_{attempt_index}_canonical.lean",
                        "semantic_attempt_canonical",
                    ),
                    (
                        blueprint_dir / f"generation_round_{attempt_index}.lean",
                        "semantic_attempt_compat",
                    ),
                ])
        candidates.extend([
            (blueprint_dir / "round_00_phase1.lean", "accepted_phase1"),
            (blueprint_dir / "blueprint.lean", "blueprint_default"),
        ])

    for value in reversed(row.get("phase1_candidate_paths") or []):
        candidates.append((Path(str(value)), "phase1_candidate_paths"))

    unique: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path, source in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        if not _is_within(resolved, experiment_root):
            raise ValueError(
                f"candidate path escapes experiment root: {resolved} not under {experiment_root}"
            )
        seen.add(resolved)
        unique.append((resolved, source))
    return unique


def _load_final_blueprint(
    experiment_root: Path,
    row: dict[str, Any],
) -> tuple[str, Path, str, str, int | None, int | None]:
    semantic_attempt = _last_completed_semantic_attempt(row)
    expected_hash = ""
    generation_attempt: int | None = None
    semantic_ordinal: int | None = None
    if semantic_attempt is not None:
        expected_hash = str(
            semantic_attempt.get("canonicalCandidateHash")
            or semantic_attempt.get("candidateHash")
            or ""
        )
        generation_attempt = int(semantic_attempt.get("round") or 0) or None
        semantic_ordinal = int(semantic_attempt.get("semanticAuditOrdinal") or 0) or None

    existing = [
        (path, source)
        for path, source in _candidate_paths(experiment_root, row, semantic_attempt)
        if path.is_file()
    ]
    if not existing:
        raise FileNotFoundError(
            f"no final Blueprint artifact for source_id={row.get('source_id')}"
        )

    mismatches: list[str] = []
    for path, source in existing:
        blueprint = path.read_text(encoding="utf-8")
        actual_hash = _sha256(blueprint)
        if expected_hash and actual_hash != expected_hash:
            mismatches.append(f"{path}={actual_hash}")
            continue
        return (
            blueprint,
            path,
            source,
            actual_hash,
            generation_attempt,
            semantic_ordinal,
        )

    raise ValueError(
        f"no candidate matches final semantic hash {expected_hash} for "
        f"source_id={row.get('source_id')}; candidates: {mismatches}"
    )


def build_rows(
    experiment_root: Path,
    *,
    experiment_name: str,
    statuses: set[str] | None,
) -> list[dict[str, Any]]:
    results_path = experiment_root / "robustpa" / "blueprint" / "results.jsonl"
    generations_path = experiment_root / "prepared" / "generation_inputs.jsonl"
    results = _latest_by(_read_jsonl(results_path), "source_id")
    generations = _latest_by(_read_jsonl(generations_path), "name")

    selected = [
        row for row in results.values()
        if statuses is None or str(row.get("status") or "") in statuses
    ]
    selected.sort(key=lambda row: str(row.get("source_id") or ""))
    if not selected:
        requested = "all" if statuses is None else ",".join(sorted(statuses))
        raise ValueError(f"no result rows matched statuses={requested}")

    output: list[dict[str, Any]] = []
    for result in selected:
        source_id = str(result.get("source_id") or "")
        generation = generations.get(source_id)
        if generation is None:
            raise ValueError(f"missing generation input for source_id={source_id}")
        problem = str(generation.get("problem") or "").strip()
        cot = str(generation.get("post_think_cot") or "").strip()
        if not problem or not cot:
            raise ValueError(f"empty problem or COT for source_id={source_id}")

        (
            blueprint,
            blueprint_path,
            blueprint_source,
            blueprint_hash,
            generation_attempt,
            semantic_ordinal,
        ) = _load_final_blueprint(experiment_root, result)
        if not blueprint.strip():
            raise ValueError(f"empty final Blueprint for source_id={source_id}")

        output.append({
            "schema_version": SCHEMA_VERSION,
            "experiment_name": experiment_name,
            "source_id": source_id,
            "record_id": str(result.get("record_id") or ""),
            "source": str(generation.get("source") or result.get("split") or ""),
            "row_index": int(generation.get("row_index", result.get("row_index", -1))),
            "pipeline_status": str(result.get("status") or ""),
            "semantic_audit_mode": str(result.get("semantic_audit_mode") or ""),
            "semantic_comparator_protocol": str(
                result.get("semantic_comparator_protocol") or ""
            ),
            "claimed_answer": str(
                generation.get("claimed_answer") or result.get("claimed_answer") or ""
            ),
            "problem": problem,
            "cot": cot,
            "final_blueprint": blueprint,
            "final_blueprint_sha256": blueprint_hash,
            "final_blueprint_path": str(blueprint_path),
            "final_blueprint_source": blueprint_source,
            "final_generation_attempt": generation_attempt,
            "final_semantic_audit_ordinal": semantic_ordinal,
            "trace_path": str(result.get("trace_path") or ""),
        })
    return output


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".parquet"
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_bundle(
    experiment_root: Path,
    *,
    experiment_name: str,
    output_dir: Path,
    statuses: set[str] | None,
) -> dict[str, Any]:
    rows = build_rows(
        experiment_root,
        experiment_name=experiment_name,
        statuses=statuses,
    )
    parquet_path = output_dir / "semantic_review_inputs.parquet"
    prompt_path = output_dir / "CHATGPT_PROMPT.md"
    manifest_path = output_dir / "manifest.json"
    _atomic_write_parquet(parquet_path, rows)
    _atomic_write_text(prompt_path, CHATGPT_PROMPT.rstrip() + "\n")

    status_counts = Counter(str(row["pipeline_status"]) for row in rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "experiment_root": str(experiment_root),
        "rows": len(rows),
        "unique_source_ids": len({str(row["source_id"]) for row in rows}),
        "status_filter": "all" if statuses is None else sorted(statuses),
        "status_counts": dict(sorted(status_counts.items())),
        "parquet_path": str(parquet_path),
        "prompt_path": str(prompt_path),
        "source_ids": [str(row["source_id"]) for row in rows],
        "blueprints": [
            {
                "source_id": row["source_id"],
                "path": row["final_blueprint_path"],
                "sha256": row["final_blueprint_sha256"],
                "generation_attempt": row["final_generation_attempt"],
                "semantic_audit_ordinal": row["final_semantic_audit_ordinal"],
            }
            for row in rows
        ],
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def _statuses(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    statuses = {item.strip() for item in value.split(",") if item.strip()}
    if not statuses:
        raise argparse.ArgumentTypeError("statuses must be a comma-separated list or 'all'")
    return statuses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bundle problem, COT, and the hash-verified final Blueprint into a "
            "Parquet file for independent ChatGPT semantic review."
        )
    )
    parser.add_argument("experiment_name", help="experiment directory name under --base-dir")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"experiment base directory (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--statuses",
        type=_statuses,
        default=set(DEFAULT_STATUSES),
        help="comma-separated pipeline statuses, or 'all' (default: semanticRejected)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: <experiment>/semantic_review_bundle)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    experiment_root = (base_dir / args.experiment_name).resolve()
    if experiment_root.parent != base_dir:
        raise ValueError("experiment_name must be one directory name, not a path")
    if not experiment_root.is_dir():
        raise FileNotFoundError(experiment_root)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else experiment_root / "semantic_review_bundle"
    )
    manifest = write_bundle(
        experiment_root,
        experiment_name=args.experiment_name,
        output_dir=output_dir,
        statuses=args.statuses,
    )
    print(
        f"[semantic-review-bundle] rows={manifest['rows']} "
        f"unique_source_ids={manifest['unique_source_ids']} "
        f"parquet={manifest['parquet_path']} prompt={manifest['prompt_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
