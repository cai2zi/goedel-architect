from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
SOURCE_PATH = (
    WORKSPACE_ROOT
    / "czx_work/cot_blueprint_refine/subsets/"
    "qwen3_8b_original_answer_incorrect/predictions.jsonl"
)
RESULTS_PATH = (
    WORKSPACE_ROOT
    / "czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_whole_cot_blueprint_generation/"
    "robustpa/blueprint/results.jsonl"
)
OUTPUT_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold"
BLIND_PATH = OUTPUT_ROOT / "blind_inputs.jsonl"
MANIFEST_PATH = OUTPUT_ROOT / "blind_manifest.json"


FORBIDDEN_OUTPUT_KEYS = {
    "gold",
    "extracted_gold",
    "is_correct",
    "math_verify_parse_ok",
    "reasoning_content",
    "raw_cot",
    "raw_response",
    "string_match_correct",
    "subset_selection",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            value["_source_line_number"] = line_number
            value["_source_line_sha256"] = _sha256_text(raw.rstrip("\n"))
            rows.append(value)
    return rows


def _nonthinking(raw_cot: str) -> str:
    marker = "</think>"
    if marker not in raw_cot:
        raise ValueError("raw_cot has no </think> delimiter")
    value = raw_cot.rsplit(marker, 1)[1].strip()
    if not value:
        raise ValueError("empty final answer after last </think>")
    if "<think>" in value or "</think>" in value:
        raise ValueError("thinking delimiter leaked into final answer")
    return value


def _claimed_answer(row: dict[str, Any]) -> str:
    values = row.get("extracted_pred")
    if isinstance(values, list) and values:
        return str(values[0])
    if values not in (None, ""):
        return str(values)
    raise ValueError(f"missing claimed answer for {row.get('ID')}")


def _result_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in _jsonl(RESULTS_PATH):
        source_id = str(row.get("source_id") or "")
        if not source_id:
            raise ValueError("result row has no source_id")
        if source_id in index:
            raise ValueError(f"duplicate result source_id: {source_id}")
        index[source_id] = row
    return index


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = _jsonl(SOURCE_PATH)
    if len(source_rows) != 76:
        raise ValueError(f"expected 76 source rows, found {len(source_rows)}")
    result_index = _result_index()
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(source_rows):
        source_id = str(row.get("ID") or "")
        if not source_id or source_id in seen_ids:
            raise ValueError(f"invalid or duplicate source ID: {source_id!r}")
        result = result_index.get(source_id)
        if result is None:
            raise ValueError(f"missing result identity for {source_id}")
        problem = str(row.get("problem") or "").strip()
        final_cot = _nonthinking(str(row.get("raw_cot") or ""))
        item = {
            "schema_version": "wrong76_blind_input_v1",
            "record_id": str(result["record_id"]),
            "source_id": source_id,
            "subset": str(result.get("subset") or "unknown"),
            "split": str(result.get("split") or "unknown"),
            "source_row_index": row_index,
            "source_line_number": int(row["_source_line_number"]),
            "source_line_sha256": str(row["_source_line_sha256"]),
            "problem": problem,
            "nonthinking_cot": final_cot,
            "claimed_answer": _claimed_answer(row),
            "extraction_rule": "after_last_think_close",
            "problem_sha256": _sha256_text(problem),
            "nonthinking_cot_sha256": _sha256_text(final_cot),
        }
        leaked = FORBIDDEN_OUTPUT_KEYS.intersection(item)
        if leaked:
            raise AssertionError(f"forbidden blind fields leaked: {sorted(leaked)}")
        output.append(item)
        seen_ids.add(source_id)
    payload = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in output
    )
    manifest = {
        "schema_version": "wrong76_blind_manifest_v1",
        "records": len(output),
        "source_path": str(SOURCE_PATH),
        "source_file_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "blind_inputs_sha256": _sha256_text(payload),
        "extraction_rule": "after_last_think_close",
        "forbidden_fields": sorted(FORBIDDEN_OUTPUT_KEYS),
        "official_gold_exposed": False,
    }
    return output, manifest


def _atomic_freeze(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise RuntimeError(f"frozen artifact drift: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    rows, manifest = build()
    blind_content = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in rows
    )
    manifest_content = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    _atomic_freeze(BLIND_PATH, blind_content)
    _atomic_freeze(MANIFEST_PATH, manifest_content)
    print(f"frozen records={len(rows)} path={BLIND_PATH}")
    print(f"sha256={manifest['blind_inputs_sha256']}")


if __name__ == "__main__":
    main()
