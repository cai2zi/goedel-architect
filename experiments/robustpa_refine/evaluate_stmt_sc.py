from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_BASE = REPO_ROOT.parent / "czx_work" / "robustpa_eval"
DEFAULT_ROBUST_PA_ROOT = REPO_ROOT.parent / "robust-proof-autoformalization"
PROOF_SC_SCOPE_FINAL_PROOF = "final-proof"
PROOF_SC_SCOPE_CONTEXT_NO_PROOF = "context-no-proof"
PROOF_SC_SCOPE_CONTEXT_WITH_PROOF = "context-with-proof"
PROOF_SC_SCOPES = {
    PROOF_SC_SCOPE_FINAL_PROOF,
    PROOF_SC_SCOPE_CONTEXT_NO_PROOF,
    PROOF_SC_SCOPE_CONTEXT_WITH_PROOF,
}

_BY_MARKER_RE = re.compile(r":=\s*by\b")
_DECL_RE = re.compile(r"(?m)^\s*(?:def|theorem|lemma|example)\s+\w*")
_DECL_NAME_RE = re.compile(r"^\s*(def|theorem|lemma|example)\s+([^\s:]+)?")
_PROOF_DECL_KINDS = {"theorem", "lemma", "example"}


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
    parser.add_argument(
        "--proof-sc-scope",
        choices=sorted(PROOF_SC_SCOPES),
        default=os.environ.get("PROOF_SC_SCOPE", PROOF_SC_SCOPE_FINAL_PROOF),
        help=(
            "Lean content shown to ProofSC: final-proof judges only the final "
            "declaration proof body; context-no-proof prepends preceding def and "
            "lemma/theorem/example context with proofs elided; context-with-proof "
            "prepends preceding def and lemma/theorem/example context including proofs."
        ),
    )
    parser.add_argument(
        "--include-subset",
        action="append",
        default=[],
        help=(
            "Only evaluate rows from this subset. May be repeated or comma-separated. "
            "When omitted, all subsets are evaluated."
        ),
    )
    parser.add_argument(
        "--include-split",
        action="append",
        default=[],
        help=(
            "Only evaluate rows from this split. May be repeated or comma-separated. "
            "When omitted, all splits are evaluated."
        ),
    )
    args = parser.parse_args()
    args.include_subset = _parse_filter_values(args.include_subset)
    args.include_split = _parse_filter_values(args.include_split)
    return args


def _parse_filter_values(raw_values: list[str]) -> list[str]:
    values: list[str] = []
    for raw_value in raw_values:
        for value in str(raw_value).split(","):
            value = value.strip()
            if value:
                values.append(value)
    return values


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


def _filter_rows_by_scope(
    rows: list[dict[str, Any]],
    *,
    include_subsets: list[str],
    include_splits: list[str],
) -> list[dict[str, Any]]:
    subset_filter = set(include_subsets)
    split_filter = set(include_splits)
    if not subset_filter and not split_filter:
        return rows
    return [
        row
        for row in rows
        if (not subset_filter or str(row.get("subset") or "") in subset_filter)
        and (not split_filter or str(row.get("split") or "") in split_filter)
    ]


def _scope_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        f"{row.get('subset') or ''}/{row.get('split') or ''}"
        for row in rows
    )


def _print_scope(prefix: str, rows: list[dict[str, Any]]) -> None:
    subsets = sorted({str(row.get("subset") or "") for row in rows})
    splits = sorted({str(row.get("split") or "") for row in rows})
    print(f"[{prefix}] adopted_subsets={subsets}", flush=True)
    print(f"[{prefix}] adopted_splits={splits}", flush=True)
    for subset_split, count in sorted(_scope_counts(rows).items()):
        print(f"[{prefix}] adopted_subset_split={subset_split} rows={count}", flush=True)


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


def _decl_label(statement: str) -> str:
    match = _DECL_NAME_RE.match(statement)
    if not match:
        return "declaration"
    kind = match.group(1)
    name = match.group(2) or ""
    return f"{kind} {name}".strip()


def _decl_kind(block: str) -> str:
    match = _DECL_NAME_RE.match(block)
    return match.group(1) if match else ""


def _comment_block(title: str, text: str) -> str:
    lines = [f"-- {title}"]
    for line in text.rstrip().splitlines():
        lines.append(f"-- {line}" if line else "--")
    return "\n".join(lines)


def _elide_proof(block: str) -> str:
    by_match = _BY_MARKER_RE.search(block)
    if not by_match:
        return block.strip()
    return f"{block[: by_match.end()].rstrip()}\n  -- proof elided for ProofSC context"


def _extract_final_proof_with_context(
    generated_fl: str,
    *,
    include_context: bool,
    include_context_proofs: bool,
) -> tuple[str | None, str | None, dict[str, Any]]:
    if not generated_fl:
        return None, None, {
            "proof_context_declaration_count": 0,
            "proof_context_proofs_included": include_context_proofs,
        }

    decls = list(_DECL_RE.finditer(generated_fl))
    declaration_blocks: list[tuple[str, str]] = []
    for idx, decl in enumerate(decls):
        start = decl.start()
        end = decls[idx + 1].start() if idx + 1 < len(decls) else len(generated_fl)
        block = generated_fl[start:end].strip()
        kind = _decl_kind(block)
        if kind:
            declaration_blocks.append((kind, block))

    final_index = None
    final_statement = None
    final_proof_body = None
    for idx in range(len(declaration_blocks) - 1, -1, -1):
        kind, block = declaration_blocks[idx]
        if kind not in _PROOF_DECL_KINDS:
            continue
        by_match = _BY_MARKER_RE.search(block)
        if not by_match:
            continue
        final_index = idx
        final_statement = block[: by_match.start()].rstrip()
        final_proof_body = block[by_match.end() :].strip()
        break

    details = {
        "proof_context_declaration_count": 0,
        "proof_context_proofs_included": include_context_proofs,
    }
    if final_index is None or not final_statement:
        return None, None, details

    proof_sections: list[str] = []
    if include_context:
        context_blocks: list[str] = []
        for kind, block in declaration_blocks[:final_index]:
            if kind == "def" or include_context_proofs:
                context_blocks.append(block.strip())
            else:
                context_blocks.append(_elide_proof(block))
        if context_blocks:
            details["proof_context_declaration_count"] = len(context_blocks)
            context_label = (
                "Lean context before final theorem (definitions and lemma proofs included)"
                if include_context_proofs
                else "Lean context before final theorem (lemma proofs elided)"
            )
            proof_sections.append(_comment_block(context_label, "\n\n".join(context_blocks)))

    if final_proof_body:
        proof_sections.append(final_proof_body)

    return final_statement, "\n\n".join(proof_sections).strip(), details


def _generated_for_proof_scope(
    generated_fl: str,
    proof_sc_scope: str,
) -> tuple[str, dict[str, Any]]:
    details: dict[str, Any] = {
        "proof_sc_scope": proof_sc_scope,
        "proof_sc_scope_transform": "none",
    }
    if proof_sc_scope == PROOF_SC_SCOPE_FINAL_PROOF:
        return generated_fl, details

    if proof_sc_scope == PROOF_SC_SCOPE_CONTEXT_NO_PROOF:
        final_statement, proof_body, extract_details = _extract_final_proof_with_context(
            generated_fl,
            include_context=True,
            include_context_proofs=False,
        )
    elif proof_sc_scope == PROOF_SC_SCOPE_CONTEXT_WITH_PROOF:
        final_statement, proof_body, extract_details = _extract_final_proof_with_context(
            generated_fl,
            include_context=True,
            include_context_proofs=True,
        )
    else:
        final_statement, proof_body, extract_details = None, None, {}
    details.update(extract_details)
    if not final_statement or not proof_body:
        details["proof_sc_scope_transform"] = "fallback_original"
        return generated_fl, details

    details["proof_sc_scope_transform"] = "context_as_final_body"
    return f"{final_statement} := by\n{proof_body}", details


def _score_one(
    row: dict[str, Any],
    *,
    sample_key: str,
    proof_sc_scope: str,
    model,
    sc_mod,
) -> dict[str, Any]:
    idx = sample_key.split("#")[-1] if "#" in sample_key else "1"
    generated_fl = str(row.get(sample_key, "") or "")
    enriched = dict(row)

    if not generated_fl:
        result = _empty_sc_result("empty_output")
    elif generated_fl.startswith("ERROR"):
        result = _empty_sc_result("error_output")
    else:
        tc_passes = _truthy_metric(row.get("GOEDEL_root_proved"))
        scoped_generated_fl, scope_details = _generated_for_proof_scope(
            generated_fl,
            proof_sc_scope,
        )
        result = sc_mod.run_sc_v3(
            informal_statement=str(row.get("informal_statement") or ""),
            informal_proof=str(row.get("informal_proof") or ""),
            generated_fl=scoped_generated_fl,
            tc_passes=tc_passes,
            model=model,
        )
        details = dict(result.get("LLM_SC_details") or {})
        details.update(scope_details)
        result["LLM_SC_details"] = details
        details = result.get("LLM_SC_details") or {}
        stmt_call = details.get("stmt_call") or {}
        proof_call = details.get("proof_call") or {}
        result["LLM_StmtSC_score"] = stmt_call.get("stmt_score")
        result["LLM_ProofSC_score"] = proof_call.get("proof_score")

    details = dict(result.get("LLM_SC_details") or {})
    details.setdefault("proof_sc_scope", proof_sc_scope)
    result["LLM_SC_details"] = details

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
        "proof_sc_scope": args.proof_sc_scope,
        "groups": groups,
    }


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path, output_path, summary_path = _resolve_paths(args)
    all_rows = _read_jsonl(input_path)
    rows = _filter_rows_by_scope(
        all_rows,
        include_subsets=args.include_subset,
        include_splits=args.include_split,
    )
    if args.limit > 0:
        rows = rows[: args.limit]
    print(
        f"[full-sc] loaded_rows={len(all_rows)} rows={len(rows)} input={input_path} "
        f"proof_sc_scope={args.proof_sc_scope}"
    )
    if args.include_subset or args.include_split:
        print(
            f"[full-sc] include_subsets={args.include_subset or 'ALL'} "
            f"include_splits={args.include_split or 'ALL'}",
            flush=True,
        )
    _print_scope("full-sc", rows)

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
                proof_sc_scope=args.proof_sc_scope,
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
