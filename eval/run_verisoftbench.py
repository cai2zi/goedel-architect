"""
Evaluate Goedel-Architect (full 3-phase pipeline) on VeriSoftBench.

This script is a thin adapter: it loads VeriSoftBench problems, builds the
per-theorem compiler and repo-search instances, delegates to the goedel-arch
src/pipeline.py, then verifies results with VeriSoftBench's LeanREPL.

Usage:
    python eval/run_verisoftbench.py --repo VCV-io --limit 5 --trace
    python eval/run_verisoftbench.py --repo VCV-io --limit 20 --blueprint
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
GOEDEL_ARCH_ROOT = Path(__file__).parent.parent
GOEDEL_ARCH_SRC  = GOEDEL_ARCH_ROOT / "src"
VSB_ROOT         = GOEDEL_ARCH_ROOT.parent / "VeriSoftBench"

# Insert VSB first, then src/ — src/ wins on name conflicts (e.g. prompts.py)
sys.path.insert(0, str(VSB_ROOT))
sys.path.insert(0, str(GOEDEL_ARCH_SRC))

from core.evaluator import _clean_thm_stmt
from core.lean_interface import LeanREPL
from prompts.prompt_builder import PromptBuilder
import utils.utils as utils

from lean_compiler import AbstractLeanCompiler
from mathlib_retrieval import MathlibRetrieval
from pipeline import prove_theorem, ProofResult
from prover import GoedelProver
from repo_retrieval import RepoRetrieval
from tracer import JsonlTracer, NullTracer, TraceEvent
import graph_viz
from vsb_lean_compiler import VSBLeanCompiler

VSB_DATA      = VSB_ROOT / "data" / "verisoftbench.jsonl"
VSB_LEAN_SRC  = VSB_ROOT / "data" / "lean_repos"
VSB_PROMPTS   = VSB_ROOT / "prompts" / "templates"
CACHE_DIR     = GOEDEL_ARCH_ROOT / "results" / "repo_index_cache"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(
    limit: int | None = None,
    repo_filter: str | None = None,
    thm_filter: str | None = None,
) -> list[dict]:
    problems = []
    with open(VSB_DATA) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if repo_filter and entry.get("lean_root") != repo_filter:
                    continue
                if thm_filter and thm_filter not in entry.get("thm_name", ""):
                    continue
                problems.append(entry)
                if limit and len(problems) >= limit:
                    break
    return problems


# ---------------------------------------------------------------------------
# Per-theorem evaluation
# ---------------------------------------------------------------------------

def evaluate_theorem(
    theorem_entry: dict,
    lean_repl: LeanREPL,
    prompt_builder: PromptBuilder,
    retrieval: MathlibRetrieval,
    repo_retrieval: RepoRetrieval | None,
    mode: str,
    use_blueprint: bool,
    model: str,
    tracer,
    verbose: bool,
) -> dict:
    thm_name  = theorem_entry["thm_name"]
    lean_root = theorem_entry["lean_root"]
    rel_path  = theorem_entry["rel_path"]
    imports   = theorem_entry.get("imports", [])
    suffix    = theorem_entry.get("suffix", "")
    local_ctx = theorem_entry.get("local_ctx", "")

    thm_stmt = _clean_thm_stmt(
        theorem_entry["thm_stmt"],
        gt_proof=theorem_entry.get("ground_truth_proof", ""),
    )
    theorem_entry["thm_stmt"] = thm_stmt

    result = {
        "thm_name": thm_name, "lean_root": lean_root, "rel_path": rel_path,
        "thm_stmt": thm_stmt, "success": False,
        "proof": None, "aux_lemmas": None, "error": None,
        "wall_time_s": 0.0, "blueprint_used": use_blueprint,
    }

    t0 = time.monotonic()

    try:
        # Build verification context
        verif_ctx = _build_verif_context(lean_repl, theorem_entry, lean_root,
                                          rel_path, imports, local_ctx, thm_stmt, thm_name)
        theorem_entry["verif_local_ctxs"] = verif_ctx

        # VeriSoftBench prompts
        sys_prompt  = prompt_builder.retrive_sys_prompt()
        user_prompt = prompt_builder.build_user_prompt(theorem_entry, mode=mode)

        # Per-theorem compiler (tracks call count for proof_id)
        def make_compiler() -> AbstractLeanCompiler:
            return VSBLeanCompiler(lean_repl, theorem_entry, call_prefix="ga")

        if use_blueprint:
            # Full 3-phase pipeline
            proof_result: ProofResult = prove_theorem(
                theorem_stmt=thm_stmt,
                model=model,
                compiler_factory=make_compiler,
                retrieval=retrieval,
                repo_retrieval=repo_retrieval,
                tracer=tracer,
                repo_context=verif_ctx,
            )
            # Use only the root-node proof body — final_lean_file is a full Lean
            # file with import statements, not a proof body VSBLeanCompiler can verify.
            raw_output = proof_result.proof_body
        else:
            # Phase 2 only (direct per-theorem proving, no blueprint)
            prover = GoedelProver(model_id=model, retrieval=retrieval, tracer=tracer)
            result_obj = prover.prove_node(
                compiler=make_compiler(),
                node_name=thm_name,
                node_stmt=thm_stmt,
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                repo_retrieval=repo_retrieval,
            )
            raw_output = result_obj.proof_body

        proof     = utils.get_proof_from_llm_output(raw_output) or raw_output.strip()
        aux_lemmas = utils.get_lemmas_from_llm_output(raw_output)

        if not proof:
            result["error"] = "No proof found in prover output."
            result["wall_time_s"] = round(time.monotonic() - t0, 2)
            return result

        name_mapping = utils.find_conflicting_names_from_local_context(verif_ctx, aux_lemmas)
        aux_lemmas, proof = utils.apply_name_replacements(aux_lemmas, proof, name_mapping)
        proof, aux_lemmas = utils.clean_leaked_identifiers(theorem_entry, proof, aux_lemmas)

        if verbose:
            print(f"[{thm_name}] Verifying final proof ...")

        success, error_msg = lean_repl.verify_proof(
            thm_name=thm_name,
            repo_name=lean_root,
            rel_path=rel_path,
            local_context=verif_ctx,
            theorem_stmt=thm_stmt,
            theorem_proof=proof,
            proof_id="ga_final",
            aux_lemmas=aux_lemmas or "",
            suffix=suffix,
        )

        result.update(proof=proof, aux_lemmas=aux_lemmas, success=success,
                      error=None if success else error_msg)

    except Exception as exc:
        result["error"] = f"Exception: {exc}"

    result["wall_time_s"] = round(time.monotonic() - t0, 2)

    tracer.emit(TraceEvent(
        kind="final_verify",
        thm_name=theorem_entry.get("thm_name", ""),
        ok=result["success"],
        args={
            "proof": result.get("proof", ""),
            "error": result.get("error", ""),
            "wall_time_s": result["wall_time_s"],
        },
    ))

    return result


def _build_verif_context(lean_repl, entry, lean_root, rel_path,
                          imports, local_ctx, thm_stmt, thm_name) -> str:
    if lean_root == "iris-lean":
        imports = [imp.replace("import src.", "import ") for imp in imports]
    fallback = "\n".join(imports) + "\n" + local_ctx
    try:
        src_path = lean_repl.lean_src_dir / lean_root / rel_path
        if src_path.exists():
            full = src_path.read_text(encoding="utf-8")
            ctx = utils.get_content_before_theorem(full, thm_stmt, thm_name=thm_name)
            if ctx is not None:
                return ctx
    except Exception:
        pass
    return fallback


def summarise(results: list[dict]) -> dict:
    total  = len(results)
    proved = sum(1 for r in results if r["success"])
    return {
        "total": total,
        "proved": proved,
        "pass_rate": round(proved / total, 4) if total else 0.0,
        "avg_wall_time_s": round(sum(r["wall_time_s"] for r in results) / total, 2) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Goedel-Architect on VeriSoftBench")
    parser.add_argument("--model",    default="gpt-4o")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--repo",    default=None)
    parser.add_argument("--thm",     default=None, help="Run a single theorem by name")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default 1 for local mode — "
                             "`lake build` is not safe to parallelize on a shared repo)")
    parser.add_argument("--output",  default="results/verisoftbench/")
    parser.add_argument("--mode",    default="filtered_context",
                        choices=["filtered_context", "full_context"])
    parser.add_argument("--quick",   action="store_true", help="10 problems, 2 workers")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--blueprint", action="store_true",
                        help="Use full 3-phase pipeline (Phase 1 + 2 + 3)")
    parser.add_argument("--trace", metavar="PATH", nargs="?", const="",
                        help="Write JSONL trace (default: results/verisoftbench/trace.jsonl)")
    args = parser.parse_args()

    if args.quick:
        args.limit, args.workers = 10, 2

    output_dir = GOEDEL_ARCH_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.trace is None:
        trace_path = None
    elif args.trace == "":
        trace_path = output_dir / "trace.jsonl"
    else:
        trace_path = Path(args.trace)
    if trace_path:
        trace_path.unlink(missing_ok=True)
    tracer = JsonlTracer(trace_path) if trace_path else NullTracer()

    print("Loading VeriSoftBench dataset ...")
    problems = load_dataset(limit=args.limit, repo_filter=args.repo, thm_filter=args.thm)
    print(f"  {len(problems)} problems"
          + (f" (repo={args.repo})" if args.repo else "")
          + (f" (thm={args.thm})" if args.thm else "")
          + (f" (limit={args.limit})" if args.limit else ""))
    if trace_path:
        print(f"  Tracing to: {trace_path}")
    if args.blueprint:
        print("  Mode: full 3-phase pipeline (Phase 1 + Phase 2 + Phase 3)")
    else:
        print("  Mode: Phase 2 only (direct per-theorem proving)")

    retrieval    = MathlibRetrieval()
    lean_repl    = LeanREPL(lean_src_dir=VSB_LEAN_SRC)
    prompt_builder = PromptBuilder(VSB_PROMPTS, mode=args.mode)

    # Build repo retrieval index (shared across all theorems in the same repo)
    repo_retrievals: dict[str, RepoRetrieval | None] = {}
    for entry in problems:
        repo = entry.get("lean_root", "")
        if repo not in repo_retrievals:
            repo_root = VSB_LEAN_SRC / repo
            if repo_root.exists():
                repo_retrievals[repo] = RepoRetrieval(repo_root, cache_dir=CACHE_DIR)
            else:
                repo_retrievals[repo] = None

    print(f"Running with model={args.model}, workers={args.workers} ...")

    all_results: list[dict] = []

    def run_one(entry: dict) -> dict:
        return evaluate_theorem(
            theorem_entry=entry,
            lean_repl=lean_repl,
            prompt_builder=prompt_builder,
            retrieval=retrieval,
            repo_retrieval=repo_retrievals.get(entry.get("lean_root", "")),
            mode=args.mode,
            use_blueprint=args.blueprint,
            model=args.model,
            tracer=tracer,
            verbose=args.verbose,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, p): p["thm_name"] for p in problems}
        for i, fut in enumerate(as_completed(futures), 1):
            thm_name = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"thm_name": thm_name, "success": False,
                       "error": str(exc), "wall_time_s": 0.0}
            all_results.append(res)
            status = "PASS" if res["success"] else "FAIL"
            print(f"  [{i}/{len(problems)}] {status} {thm_name} ({res.get('wall_time_s',0):.1f}s)")

    summary = summarise(all_results)
    print(f"\nResults: {summary['proved']}/{summary['total']} proved "
          f"({summary['pass_rate']*100:.1f}%), avg {summary['avg_wall_time_s']:.1f}s/theorem")

    results_file = output_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    with open(output_dir / "summary.json", "w") as f:
        json.dump({**summary, "model": args.model, "mode": args.mode,
                   "blueprint": args.blueprint}, f, indent=2)

    print(f"Results saved to {output_dir}")

    if trace_path and trace_path.exists():
        graph_viz.generate(trace_path)
        print(f"\nView graph:  python eval/serve_viz.py {trace_path}")


if __name__ == "__main__":
    main()
