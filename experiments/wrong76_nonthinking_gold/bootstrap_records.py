from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUT_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold"
BLIND_PATH = OUTPUT_ROOT / "blind_inputs.jsonl"
RECORDS_ROOT = OUTPUT_ROOT / "records"

EXPERIMENT_ROOTS = (
    WORKSPACE_ROOT
    / "czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge/"
    "robustpa/blueprint",
    WORKSPACE_ROOT
    / "czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_whole_cot_blueprint_generation/robustpa/blueprint",
    WORKSPACE_ROOT
    / "czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_blueprint_generation/robustpa/blueprint",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw]


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _result_candidates() -> dict[str, list[tuple[int, Path]]]:
    found: dict[str, list[tuple[int, Path]]] = {}
    accepted = {"strictAccepted", "acceptedWithWarnings", "solved"}
    for rank, root in enumerate(EXPERIMENT_ROOTS):
        results_path = root / "results.jsonl"
        if not results_path.is_file():
            continue
        for row in _rows(results_path):
            source_id = str(row.get("source_id") or "")
            record_id = str(row.get("record_id") or "")
            if not source_id or not record_id:
                continue
            status_rank = 0 if row.get("status") in accepted else 10
            checkpoint = Path(str(row.get("checkpoint_path") or ""))
            blueprint_dir = (
                root / "blueprints" / str(row.get("subset") or "unknown")
                / str(row.get("split") or "unknown") / record_id
            )
            paths: list[Path] = []
            if checkpoint.is_file() and row.get("status") in accepted:
                round_zero = blueprint_dir / "round_00_phase1.lean"
                if round_zero.is_file():
                    paths.append(round_zero)
            for name in ("phase1_failed_last.lean", "generation_round_8.lean"):
                path = blueprint_dir / name
                if path.is_file():
                    paths.append(path)
            for path in paths:
                found.setdefault(source_id, []).append((status_rank + rank, path))
    return found


def _paragraph_spans(text: str) -> list[dict[str, Any]]:
    """Create an authoring draft only; Gold steps require manual review."""
    spans: list[dict[str, Any]] = []
    for match in re.finditer(r"(?s)(?:\A|\n\s*\n)(.*?)(?=\n\s*\n|\Z)", text):
        raw = match.group(1)
        stripped = raw.strip()
        if not stripped or re.fullmatch(r"[-#*\s]+", stripped):
            continue
        if stripped.startswith("###") or stripped.startswith("---"):
            continue
        # Keep paragraphs with an equation/number or an explicit mathematical
        # consequence.  False positives are expected and manually removed.
        mathematical = bool(
            "$" in stripped
            or re.search(r"\d", stripped)
            or re.search(
                r"\b(?:therefore|thus|hence|implies|equals|solve|compute|"
                r"probability|area|distance|radius|count|sum|product)\b",
                stripped,
                flags=re.IGNORECASE,
            )
        )
        if not mathematical:
            continue
        leading = len(raw) - len(raw.lstrip())
        start = match.start(1) + leading
        end = start + len(stripped)
        spans.append({
            "step_id": f"S{len(spans) + 1:02d}",
            "ordinal": len(spans) + 1,
            "source_span": stripped,
            "char_start": start,
            "char_end": end,
            "draft_requires_manual_review": True,
        })
    return spans


def _freeze(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"frozen record drift: {path}")
        return
    path.write_text(text, encoding="utf-8")


def main() -> None:
    blind = _rows(BLIND_PATH)
    candidates = _result_candidates()
    seeded = 0
    for item in blind:
        record_dir = RECORDS_ROOT / _safe(str(item["record_id"]))
        source_path = record_dir / "source.json"
        _freeze(source_path, json.dumps(item, ensure_ascii=False, indent=2) + "\n")
        draft = {
            "schema_version": "wrong76_step_draft_v1",
            "record_id": item["record_id"],
            "nonthinking_cot_sha256": item["nonthinking_cot_sha256"],
            "steps": _paragraph_spans(str(item["nonthinking_cot"])),
            "review_status": "draft",
        }
        _freeze(
            record_dir / "steps.draft.json",
            json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
        )
        ranked = sorted(candidates.get(str(item["source_id"]), []), key=lambda pair: pair[0])
        if ranked:
            seed_path = ranked[0][1]
            _freeze(record_dir / "seed_blueprint.lean", seed_path.read_text(encoding="utf-8"))
            _freeze(
                record_dir / "seed_provenance.json",
                json.dumps({
                    "source_path": str(seed_path),
                    "status": "syntax_reference_only",
                    "promoted_to_gold": False,
                }, ensure_ascii=False, indent=2) + "\n",
            )
            seeded += 1
    print(f"records={len(blind)} syntax_seeds={seeded} root={RECORDS_ROOT}")


if __name__ == "__main__":
    main()
