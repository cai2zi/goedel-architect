from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from shared.io_utils import safe_stem, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_BASE = REPO_ROOT.parent / "czx_work" / "robustpa_eval"
_GOEDEL_HELPERS: tuple[Any, Any, Any, Any] | None = None


def _load_goedel_helpers() -> tuple[Any, Any, Any, Any]:
    """Load Goedel internals lazily so --help does not require full deps."""
    global _GOEDEL_HELPERS
    if _GOEDEL_HELPERS is None:
        from checkpoint import CheckpointState
        from pipeline import _assemble_partial_file, _orch_result_from_checkpoint
        from robustpa_refine.run_robustpa_refine import _read_parquet_rows

        _GOEDEL_HELPERS = (
            CheckpointState,
            _assemble_partial_file,
            _orch_result_from_checkpoint,
            _read_parquet_rows,
        )
    return _GOEDEL_HELPERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Goedel robustpa_refine experiment into the RobustPABench "
            "LLM_Output#1 JSONL shape used by StmtSC evaluation."
        )
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Finished robustpa_refine experiment directory containing results.jsonl.",
    )
    parser.add_argument(
        "--exp-name",
        default="",
        help="Name under --output-base. Defaults to basename of --exp-dir.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Base directory for exported evaluation artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output JSONL path. Defaults to <output-base>/<exp-name>/stmt_sc_inputs.jsonl.",
    )
    parser.add_argument(
        "--lean-output-dir",
        type=Path,
        default=None,
        help=(
            "Override directory for assembled Lean files. Defaults to "
            "<output-base>/<exp-name>/lean_outputs."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Export only the first N final result rows after de-duplication (0 = all).",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _latest_result_rows(results_path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(results_path):
        row_id = str(row.get("id") or "")
        if row_id:
            latest[row_id] = row
    return list(latest.values())


class ParquetRowCache:
    def __init__(self, read_parquet_rows) -> None:
        self._read_parquet_rows = read_parquet_rows
        self._cache: dict[Path, list[dict[str, Any]]] = {}

    def get_row(self, parquet_path: Path, row_index_1based: int) -> dict[str, Any]:
        if parquet_path not in self._cache:
            self._cache[parquet_path] = self._read_parquet_rows(parquet_path)
        rows = self._cache[parquet_path]
        idx = row_index_1based - 1
        if idx < 0 or idx >= len(rows):
            raise IndexError(f"row_index out of range: {parquet_path}:{row_index_1based}")
        return rows[idx]


def _result_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _assemble_lean_from_checkpoint(checkpoint_path: Path) -> tuple[str, str, dict[str, Any]]:
    """Return (lean_text_or_error, assembly_kind, diagnostics)."""
    if not checkpoint_path.exists():
        return (
            f"ERROR [Goedel checkpoint missing: {checkpoint_path}]",
            "error",
            {"reason": "missing_checkpoint"},
        )
    try:
        CheckpointState, assemble_partial_file, orch_result_from_checkpoint, _ = _load_goedel_helpers()
        state = CheckpointState.load(checkpoint_path)
        blueprint = state.get_blueprint()
        if blueprint is None:
            return (
                f"ERROR [Goedel checkpoint has no blueprint: {checkpoint_path}]",
                "error",
                {"reason": "missing_blueprint"},
            )
        orch_result = orch_result_from_checkpoint(state, blueprint)
        lean = assemble_partial_file(blueprint, orch_result, dict(state.proved_cache))
        root_proved = blueprint.target_theorem in state.proved_cache
        kind = "final" if state.success and root_proved else "partial"
        diagnostics = {
            "state_done": state.done,
            "state_success": state.success,
            "state_iteration": state.iteration,
            "blueprint_target": blueprint.target_theorem,
            "proved_cache_count": len(state.proved_cache),
            "node_results_count": len(state.node_results),
            "root_in_proved_cache": root_proved,
        }
        return lean, kind, diagnostics
    except BaseException as exc:
        return (
            f"ERROR [Goedel Lean assembly failed: {_result_text(exc)}]",
            "error",
            {"reason": "assembly_exception", "exception": _result_text(exc)},
        )


def _write_lean_file(
    lean_output_dir: Path,
    row: dict[str, Any],
    lean_text: str,
    assembly_kind: str,
) -> tuple[str, str, int]:
    if assembly_kind == "error" or lean_text.startswith("ERROR"):
        return "", "", 0
    subset = safe_stem(str(row.get("subset") or "unknown_subset"))
    split = safe_stem(str(row.get("split") or "unknown_split"))
    record_id = safe_stem(str(row.get("record_id") or row.get("source_id") or row.get("id") or "row"))
    path = lean_output_dir / subset / split / f"{record_id}.lean"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lean_text, encoding="utf-8")
    digest = hashlib.sha256(lean_text.encode("utf-8")).hexdigest()
    return str(path), digest, len(lean_text)


def _make_export_row(
    result_row: dict[str, Any],
    source_row: dict[str, Any],
    lean_text: str,
    assembly_kind: str,
    assembly_diag: dict[str, Any],
    lean_path: str,
    lean_hash: str,
    lean_chars: int,
) -> dict[str, Any]:
    source_id = str(result_row.get("source_id") or source_row.get("name") or source_row.get("id") or "")
    return {
        "name": source_id,
        "id": result_row.get("id", ""),
        "record_id": result_row.get("record_id", ""),
        "source_id": source_id,
        "subset": result_row.get("subset", ""),
        "split": result_row.get("split", ""),
        "row_index": result_row.get("row_index", ""),
        "parquet_path": result_row.get("parquet_path", ""),
        "theorem_name": result_row.get("theorem_name", ""),
        "informal_statement": str(source_row.get("informal_statement") or result_row.get("theorem_stmt") or ""),
        "informal_proof": str(source_row.get("informal_proof") or ""),
        "LLM_Output#1": lean_text,
        "INFERENCE_DONE": "yes",
        "GOEDEL_status": result_row.get("status", ""),
        "GOEDEL_phase": result_row.get("phase", ""),
        "GOEDEL_success": bool(result_row.get("success")),
        "GOEDEL_root_proved": bool(result_row.get("root_proved")),
        "GOEDEL_all_nodes_proved": bool(result_row.get("all_nodes_proved")),
        "GOEDEL_iterations": result_row.get("iterations", 0),
        "GOEDEL_total_nodes": result_row.get("total_nodes", 0),
        "GOEDEL_proved_node_count": result_row.get("proved_node_count", 0),
        "GOEDEL_failed_nodes": result_row.get("failed_nodes", []),
        "GOEDEL_checkpoint_path": result_row.get("checkpoint_path", ""),
        "GOEDEL_trace_path": result_row.get("trace_path", ""),
        "GOEDEL_lean_output_path": lean_path,
        "GOEDEL_lean_output_hash": lean_hash,
        "GOEDEL_lean_output_chars": lean_chars,
        "GOEDEL_lean_assembly": assembly_kind,
        "GOEDEL_lean_assembly_details": assembly_diag,
    }


def export(args: argparse.Namespace) -> Path:
    _, _, _, read_parquet_rows = _load_goedel_helpers()
    exp_dir = args.exp_dir.resolve()
    exp_name = args.exp_name or exp_dir.name
    eval_dir = args.output_base / exp_name
    output_path = args.output or (eval_dir / "stmt_sc_inputs.jsonl")
    lean_output_dir = args.lean_output_dir or (eval_dir / "lean_outputs")

    results_path = exp_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"results.jsonl not found: {results_path}")

    result_rows = _latest_result_rows(results_path)
    if args.limit > 0:
        result_rows = result_rows[: args.limit]

    parquet_cache = ParquetRowCache(read_parquet_rows)
    exported_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for result_row in tqdm(result_rows, desc="export-stmt-sc", unit="row"):
        try:
            source_row = parquet_cache.get_row(
                Path(str(result_row.get("parquet_path") or "")),
                int(result_row.get("row_index") or 0),
            )
        except BaseException as exc:
            source_row = {}
            counts["source_row_error"] += 1
            source_error = _result_text(exc)
        else:
            source_error = ""

        lean_text, assembly_kind, assembly_diag = _assemble_lean_from_checkpoint(
            Path(str(result_row.get("checkpoint_path") or ""))
        )
        if source_error:
            assembly_diag = dict(assembly_diag)
            assembly_diag["source_row_error"] = source_error

        lean_path, lean_hash, lean_chars = _write_lean_file(
            lean_output_dir, result_row, lean_text, assembly_kind
        )
        counts[assembly_kind] += 1
        exported_rows.append(
            _make_export_row(
                result_row,
                source_row,
                lean_text,
                assembly_kind,
                assembly_diag,
                lean_path,
                lean_hash,
                lean_chars,
            )
        )

    write_jsonl(output_path, exported_rows)
    print(f"[export] rows={len(exported_rows)} output={output_path}")
    print(f"[export] lean_outputs={lean_output_dir}")
    print(f"[export] assembly_counts={dict(counts)}")
    return output_path


def main() -> None:
    export(parse_args())


if __name__ == "__main__":
    main()
