from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from blueprint import _parse_blueprint  # noqa: E402
from llm_client import make_client  # noqa: E402
from semantic_audit import (  # noqa: E402
    build_formal_view,
    run_canonical_compact_whole_cot_comparator,
    run_canonical_direct_whole_cot_comparator,
    run_formal_decompiler,
)
from tracer import JsonlTracer  # noqa: E402


PREPARED = Path(
    "/ssd/czx/czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_global_defs_direct_named_t00/prepared/"
    "generation_inputs.jsonl"
)
BASELINE = Path(
    "/ssd/czx/czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_global_defs_compact_separate_named_t00_rerun1/"
    "robustpa/blueprint/blueprints/qwen3_8b_math_verify"
)
REPAIR = Path(
    "/ssd/czx/czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_global_defs_compact_separate_repair_v1_semrej11_t00/"
    "robustpa/blueprint/blueprints/qwen3_8b_math_verify"
)
MODEL = "Qwen3.5-397B-A17B-FP8"


CASES = {
    "MATH-500/test/counting_and_probability/765.json": (
        REPAIR / "MATH_500/robustpa_MATH_500_test_counting_and_probability_765_json/round_00_phase1.lean",
        "known V1 false positive; universal intermediate overclaim",
    ),
    "cmimc_2025/11": (
        REPAIR / "cmimc_2025/robustpa_cmimc_2025_11/round_00_phase1.lean",
        "known V1 false positive; preassigned material intermediate",
    ),
    "MATH-500/test/geometry/434.json": (
        BASELINE / "MATH_500/robustpa_MATH_500_test_geometry_434_json/phase1_failed_last.lean",
        "target derivation shortcut",
    ),
    "aime_2025/20": (
        BASELINE / "aime_2025/robustpa_aime_2025_20/phase1_failed_last.lean",
        "vacuous or hard-coded geometry",
    ),
    "MATH-500/test/precalculus/768.json": (
        BASELINE / "MATH_500/robustpa_MATH_500_test_precalculus_768_json/round_00_phase1.lean",
        "exhaustive answer-scope mismatch",
    ),
    "aime_2025/11": (
        BASELINE / "aime_2025/robustpa_aime_2025_11/round_00_phase1.lean",
        "exhaustive answer-scope mismatch",
    ),
    "hmmt_feb_2025/16": (
        BASELINE / "hmmt_feb_2025/robustpa_hmmt_feb_2025_16/phase1_failed_last.lean",
        "wrong semantic relation",
    ),
    "cmimc_2025/39": (
        BASELINE / "cmimc_2025/robustpa_cmimc_2025_39/phase1_failed_last.lean",
        "material dependency break",
    ),
}


def _load_records() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in PREPARED.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["name"] in CASES:
            rows[row["name"]] = row
    missing = set(CASES) - set(rows)
    if missing:
        raise RuntimeError(f"missing prepared records: {sorted(missing)}")
    return rows


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ids", nargs="*", choices=tuple(CASES))
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to reuse probe output: {args.output}")
    args.output.mkdir(parents=True)
    trace_path = args.output / "trace.jsonl"
    os.environ.setdefault("GOEDEL_OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
    os.environ.setdefault("GOEDEL_OPENAI_API_KEY", "dummy")
    tracer = JsonlTracer(trace_path)
    client = make_client(MODEL)
    selected_cases = {
        case_id: CASES[case_id] for case_id in (args.ids or tuple(CASES))
    }
    records = _load_records()
    prepared: dict[str, dict[str, Any]] = {}
    for case_id, (path, expectation) in selected_cases.items():
        if not path.is_file():
            raise RuntimeError(f"missing frozen candidate: {path}")
        code = path.read_text(encoding="utf-8")
        blueprint = _parse_blueprint(code, "n_final")
        prepared[case_id] = {
            "record": records[case_id], "view": build_formal_view(blueprint),
            "candidate_path": str(path), "candidate_hash": _sha256(code),
            "expectation": expectation,
        }
    manifest = [{
        "id": case_id,
        "candidatePath": item["candidate_path"],
        "candidateHash": item["candidate_hash"],
        "formalViewHash": item["view"].sha256,
        "expectation": item["expectation"],
    } for case_id, item in prepared.items()]
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    def run(case_id: str, mode: str) -> dict[str, Any]:
        item = prepared[case_id]
        row, view = item["record"], item["view"]
        common = dict(
            informal_statement=row["informal_statement"],
            informal_proof=row["informal_proof"],
            claimed_answer=row["claimed_answer"], view=view,
            max_tokens=16384, max_attempts=2, enable_thinking=True,
            temperature=0.0, top_p=0.95, top_k=20, min_p=0.0,
            presence_penalty=0.0, repetition_penalty=1.0,
            tracer=tracer, thm_name=case_id, round_index=0,
        )
        decompiler = None
        if mode == "compact_separate":
            decompiler = run_formal_decompiler(
                client, MODEL, view=view, max_tokens=16384, max_attempts=2,
                enable_thinking=True, temperature=0.0, top_p=0.95,
                top_k=20, min_p=0.0, presence_penalty=0.0,
                repetition_penalty=1.0, tracer=tracer, thm_name=case_id,
                round_index=0, compact=True,
            )
            result = run_canonical_compact_whole_cot_comparator(
                client, MODEL, decompiler=decompiler, **common,
            )
        else:
            result = run_canonical_direct_whole_cot_comparator(
                client, MODEL, **common,
            )
        return {
            "id": case_id, "mode": mode,
            "candidatePath": item["candidate_path"],
            "candidateHash": item["candidate_hash"],
            "formalViewHash": view.sha256,
            "expectation": item["expectation"],
            "decompiler": decompiler.to_dict() if decompiler else None,
            "comparator": result.to_dict(),
        }

    outputs: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run, case_id, mode): (case_id, mode)
                for case_id in selected_cases
                for mode in ("direct", "compact_separate")
            }
            for future in as_completed(futures):
                case_id, mode = futures[future]
                result = future.result()
                outputs.append(result)
                print(
                    f"[probe] {case_id} {mode} passed={result['comparator']['passed']} "
                    f"issues={len(result['comparator']['issues'])}",
                    flush=True,
                )
    finally:
        tracer.close()
    outputs.sort(key=lambda item: (item["id"], item["mode"]))
    (args.output / "results.json").write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for item in outputs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[probe-complete] results={len(outputs)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
