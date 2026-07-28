from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_BASE = REPO_ROOT.parent / "czx_work" / "robustpa_eval"
DEFAULT_ROBUST_PA_ROOT = REPO_ROOT.parent / "robust-proof-autoformalization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RobustPABench-aligned full correctness evaluation "
            "(TC + StmtSC + ProofSC + FullyCorrect) "
            "on Goedel robustpa_refine exported outputs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Exported JSONL containing informal_statement, informal_proof, "
            "GOEDEL_root_proved, and LLM_Output#1. "
            "Defaults to <eval-base>/<exp-name>/stmt_sc_inputs.jsonl."
        ),
    )
    parser.add_argument(
        "--exp-name",
        default="",
        help="Experiment name under --eval-base, used when --input is omitted.",
    )
    parser.add_argument("--eval-base", type=Path, default=DEFAULT_EVAL_BASE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Scored JSONL. Defaults to <input dir>/stmt_sc_scored.jsonl.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary JSON. Defaults to <input dir>/stmt_sc_summary.json.",
    )
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument(
        "--robust-pa-root",
        type=Path,
        default=DEFAULT_ROBUST_PA_ROOT,
        help="Path to robust-proof-autoformalization, whose StmtSC prompt/parser are reused.",
    )
    parser.add_argument("--sample-key", default="LLM_Output#1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("FULL_SC_WORKERS", os.environ.get("STMT_SC_WORKERS", "8"))),
        help="Concurrent Gemini judge calls.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.input is None:
        if not args.exp_name:
            raise SystemExit("Pass --input or --exp-name")
        input_path = args.eval_base / args.exp_name / "stmt_sc_inputs.jsonl"
    else:
        input_path = args.input
    output_path = args.output or (input_path.parent / "stmt_sc_scored.jsonl")
    summary_path = args.summary or (input_path.parent / "stmt_sc_summary.json")
    return input_path, output_path, summary_path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_robustpa_modules(root: Path):
    sc_path = root / "sc.py"
    utils_path = root / "utils.py"
    if not sc_path.exists():
        raise FileNotFoundError(f"RobustPA sc.py not found: {sc_path}")
    if not utils_path.exists():
        raise FileNotFoundError(f"RobustPA utils.py not found: {utils_path}")
    sc_mod = _load_module("robustpa_sc_full", sc_path)
    utils_mod = _load_module("robustpa_utils_full", utils_path)
    return sc_mod, utils_mod


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _empty_sc_result(reason: str) -> dict[str, Any]:
    return {
        "LLM_StmtSC?": "no",
        "LLM_ProofSC?": "no",
        "LLM_BothSC?": "no",
        "LLM_ValidProofSC?": "no",
        "LLM_FullyCorrect?": "no",
        "LLM_StmtSC_score": -1,
        "LLM_ProofSC_score": -1,
        "LLM_Semantics?": "no",
        "LLM_SC_details": {
            "version": "v3_decoupled",
            "early_return_reason": reason,
        },
    }


def _score_one(row: dict[str, Any], *, sample_key: str, model, sc_mod) -> dict[str, Any]:
    idx = sample_key.split("#")[-1] if "#" in sample_key else "1"
    generated_fl = str(row.get(sample_key, "") or "")
    enriched = dict(row)

    if not generated_fl:
        result = _empty_sc_result("empty_output")
    elif generated_fl.startswith("ERROR"):
        result = _empty_sc_result("error_output")
    else:
        tc_passes = _truthy_metric(row.get("GOEDEL_root_proved"))
        result = sc_mod.run_sc_v3(
            informal_statement=str(row.get("informal_statement") or ""),
            informal_proof=str(row.get("informal_proof") or ""),
            generated_fl=generated_fl,
            tc_passes=tc_passes,
            model=model,
        )
        details = result.get("LLM_SC_details") or {}
        stmt_call = details.get("stmt_call") or {}
        proof_call = details.get("proof_call") or {}
        result["LLM_StmtSC_score"] = stmt_call.get("stmt_score")
        result["LLM_ProofSC_score"] = proof_call.get("proof_score")

    for key, value in result.items():
        enriched[f"{key}#{idx}"] = value
    return enriched


def _bool_group_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no", ""}:
        return "false"
    return str(value)


def _truthy_metric(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "passed", "pass"}


def _yes_metric(value: Any) -> bool:
    return str(value).strip().lower() == "yes"


def _sample_idx(sample_key: str) -> str:
    return sample_key.split("#")[-1] if "#" in sample_key else "1"


def _metric_row(
    scope: str,
    rows: list[dict[str, Any]],
    *,
    sample_idx: str,
    **labels: str,
) -> dict[str, Any]:
    total = len(rows)
    stmt_key = f"LLM_StmtSC?#{sample_idx}"
    proof_key = f"LLM_ProofSC?#{sample_idx}"
    both_key = f"LLM_BothSC?#{sample_idx}"
    valid_proof_key = f"LLM_ValidProofSC?#{sample_idx}"
    fully_correct_key = f"LLM_FullyCorrect?#{sample_idx}"
    details_key = f"LLM_SC_details#{sample_idx}"
    stmt_yes = sum(1 for row in rows if _yes_metric(row.get(stmt_key)))
    tc_yes = sum(1 for row in rows if _truthy_metric(row.get("GOEDEL_root_proved")))
    tc_stmt_yes = sum(
        1
        for row in rows
        if _truthy_metric(row.get("GOEDEL_root_proved")) and _yes_metric(row.get(stmt_key))
    )
    proof_yes = sum(1 for row in rows if _yes_metric(row.get(proof_key)))
    both_yes = sum(1 for row in rows if _yes_metric(row.get(both_key)))
    valid_proof_yes = sum(1 for row in rows if _yes_metric(row.get(valid_proof_key)))
    fully_correct_yes = sum(1 for row in rows if _yes_metric(row.get(fully_correct_key)))
    if not any(fully_correct_key in row for row in rows):
        fully_correct_yes = sum(
            1
            for row in rows
            if _truthy_metric(row.get("GOEDEL_root_proved"))
            and _yes_metric(row.get(stmt_key))
            and _yes_metric(row.get(proof_key))
        )
    details = [row.get(details_key) or {} for row in rows]
    early = Counter(str(d.get("early_return_reason") or "") for d in details)
    stmt_parse_failed = sum(
        1
        for d in details
        if isinstance(d.get("stmt_call"), dict) and d["stmt_call"].get("_parse_failed")
    )
    proof_parse_failed = sum(
        1
        for d in details
        if isinstance(d.get("proof_call"), dict) and d["proof_call"].get("_parse_failed")
    )
    out = {
        "scope": scope,
        "total": total,
        "tc_source": "GOEDEL_root_proved",
        "tc_yes": tc_yes,
        "tc_no": total - tc_yes,
        "tc_acc": round(tc_yes / total, 6) if total else 0.0,
        "stmt_sc_yes": stmt_yes,
        "stmt_sc_no": total - stmt_yes,
        "stmt_sc_acc": round(stmt_yes / total, 6) if total else 0.0,
        "tc_and_stmt_sc_yes": tc_stmt_yes,
        "tc_and_stmt_sc_acc": round(tc_stmt_yes / total, 6) if total else 0.0,
        "proof_sc_available": True,
        "proof_sc_yes": proof_yes,
        "proof_sc_no": total - proof_yes,
        "proof_sc_acc": round(proof_yes / total, 6) if total else 0.0,
        "both_sc_yes": both_yes,
        "both_sc_no": total - both_yes,
        "both_sc_acc": round(both_yes / total, 6) if total else 0.0,
        "valid_proof_sc_yes": valid_proof_yes,
        "valid_proof_sc_no": total - valid_proof_yes,
        "valid_proof_sc_acc": round(valid_proof_yes / total, 6) if total else 0.0,
        "fully_correct_available": True,
        "fully_correct_yes": fully_correct_yes,
        "fully_correct_no": total - fully_correct_yes,
        "fully_correct_acc": round(fully_correct_yes / total, 6) if total else 0.0,
        "error_output": early.get("error_output", 0),
        "empty_output": early.get("empty_output", 0),
        "no_theorem": early.get("no_theorem", 0),
        "stmt_parse_failed": stmt_parse_failed,
        "proof_parse_failed": proof_parse_failed,
        "parse_failed": stmt_parse_failed + proof_parse_failed,
    }
    out.update(labels)
    return out


def _summary(scored_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    sample_idx = _sample_idx(args.sample_key)
    groups: list[dict[str, Any]] = [
        _metric_row("overall", scored_rows, sample_idx=sample_idx)
    ]

    def add_group(scope: str, key_fn, label_names: tuple[str, ...]) -> None:
        buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in scored_rows:
            buckets[key_fn(row)].append(row)
        for key, rows in sorted(buckets.items()):
            labels = {name: value for name, value in zip(label_names, key)}
            groups.append(_metric_row(scope, rows, sample_idx=sample_idx, **labels))

    add_group("subset", lambda r: (str(r.get("subset") or ""),), ("subset",))
    add_group("split", lambda r: (str(r.get("split") or ""),), ("split",))
    add_group(
        "subset_split",
        lambda r: (str(r.get("subset") or ""), str(r.get("split") or "")),
        ("subset", "split"),
    )
    add_group("goedel_status", lambda r: (str(r.get("GOEDEL_status") or ""),), ("status",))
    add_group(
        "goedel_root_proved",
        lambda r: (_bool_group_value(r.get("GOEDEL_root_proved")),),
        ("root_proved",),
    )

    return {
        "primary_metric": "fully_correct_acc",
        "paper_metric_note": (
            "TC is GOEDEL_root_proved. StmtSC, ProofSC, BothSC, ValidProofSC, "
            "and FullyCorrect are produced by RobustPA sc.run_sc_v3; "
            "FullyCorrect = TC and StmtSC and ProofSC."
        ),
        "gemini_model": args.gemini_model,
        "sample_key": args.sample_key,
        "groups": groups,
    }


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path, output_path, summary_path = _resolve_paths(args)
    rows = _read_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[full-sc] rows={len(rows)} input={input_path}")

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY before running full SC.")

    sc_mod, utils_mod = _load_robustpa_modules(args.robust_pa_root)
    model = utils_mod.make_gemini_model(args.gemini_model)

    scored: list[dict[str, Any] | None] = [None] * len(rows)
    max_workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _score_one,
                row,
                sample_key=args.sample_key,
                model=model,
                sc_mod=sc_mod,
            ): idx
            for idx, row in enumerate(rows)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="full-sc", unit="row"):
            idx = futures[fut]
            scored[idx] = fut.result()

    scored_rows = [row for row in scored if row is not None]
    _write_jsonl(output_path, scored_rows)
    summary = _summary(scored_rows, args)
    _write_json(summary_path, summary)
    overall = summary["groups"][0] if summary["groups"] else {}
    print(
        "[correctness] "
        f"TC={overall.get('tc_yes')}/{overall.get('total')} "
        f"({overall.get('tc_acc')}) "
        f"StmtSC={overall.get('stmt_sc_yes')}/{overall.get('total')} "
        f"({overall.get('stmt_sc_acc')}) "
        f"TC&StmtSC={overall.get('tc_and_stmt_sc_yes')}/{overall.get('total')} "
        f"({overall.get('tc_and_stmt_sc_acc')}) "
        f"ProofSC={overall.get('proof_sc_yes')}/{overall.get('total')} "
        f"({overall.get('proof_sc_acc')}) "
        f"BothSC={overall.get('both_sc_yes')}/{overall.get('total')} "
        f"({overall.get('both_sc_acc')}) "
        f"ValidProofSC={overall.get('valid_proof_sc_yes')}/{overall.get('total')} "
        f"({overall.get('valid_proof_sc_acc')}) "
        f"FullyCorrect={overall.get('fully_correct_yes')}/{overall.get('total')} "
        f"({overall.get('fully_correct_acc')})"
    )
    print(f"[full-sc] scored={output_path}")
    print(f"[full-sc] summary={summary_path}")
    return output_path, summary_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
