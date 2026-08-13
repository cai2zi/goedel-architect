from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI
from omegaconf import DictConfig, OmegaConf

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_blueprint_refine.common import latest_rows, load_config, output_root, prepared_dir, robustpa_dir  # noqa: E402
from cot_blueprint_refine.evaluate import evaluate  # noqa: E402
from cot_blueprint_refine.export_blueprint_contexts import export_contexts  # noqa: E402
from cot_blueprint_refine.prepare_inputs import DATASET_SUBSET, prepare  # noqa: E402
from cot_blueprint_refine.run_cot_refinement import refine  # noqa: E402
from cot_blueprint_refine.vllm_runtime import PersistentVLLMRuntime  # noqa: E402
from cot_blueprint_refine.kimina_runtime import PersistentKiminaRuntime  # noqa: E402
from kimina_lean_compiler import KiminaLeanCompiler  # noqa: E402


STAGES = ("prepare", "blueprint", "export", "refine", "evaluate")
STAGE_SEQUENCES = {
    "all": STAGES,
    "cot-to-blueprint": ("prepare", "blueprint", "export"),
    "blueprint-refine": ("prepare", "blueprint", "export", "refine"),
    "phase1-only": ("prepare", "blueprint"),
    "phase2-only": ("blueprint",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blueprint-guided COT refinement experiment")
    parser.add_argument("--profile", default="base", help="Config profile under configs/")
    parser.add_argument(
        "--stage",
        choices=(*STAGES, *STAGE_SEQUENCES),
        default="all",
        help=(
            "A single stage, all stages, cot-to-blueprint "
            "(prepare+blueprint+export), or blueprint-refine "
            "(prepare+blueprint+export+refine)"
        ),
    )
    parser.add_argument("override", nargs="*", help="OmegaConf dot-list overrides")
    return parser.parse_args()


def preflight_kimina(config: DictConfig) -> None:
    compiler = KiminaLeanCompiler(
        api_url=str(config.blueprint.lean_api_url),
        timeout_s=int(config.blueprint.lean_server_timeout),
        reuse=True,
        debug=False,
        max_inflight_snippets=min(2, int(config.blueprint.lean_max_inflight_snippets)),
        batch_size=1,
    )
    try:
        result = compiler.check("import GoedelArch\nexample : True := by trivial\n")
    finally:
        compiler.close()
    if not result.success:
        diagnostics = "\n".join(result.diagnostics) or result.raw_output
        raise RuntimeError(f"Kimina preflight failed ({result.failure_kind}):\n{diagnostics}")
    print("[preflight] Kimina compilation successful", flush=True)


def preflight_model(model: str, base_url: str, api_key: str = "dummy") -> None:
    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=30.0)
    available = {item.id for item in client.models.list().data}
    if model not in available:
        raise RuntimeError(
            f"Model {model!r} is not served at {base_url}; available={sorted(available)}"
        )
    print(f"[preflight] model={model} base_url={base_url} successful", flush=True)


def _blueprint_result_is_terminal(row: dict[str, Any], *, retry_error_results: bool) -> bool:
    status = str(row.get("status") or "")
    if bool(row.get("root_proved")) or status in {
        "solved", "exhausted", "strictAccepted",
        "acceptedWithWarnings",
    }:
        return True
    return status in {
        "error", "semanticRejected", "semanticAuditError", "structuralRejected", "infraError",
    } and not retry_error_results


def blueprint_results_complete(config: DictConfig) -> bool:
    if not bool(config.resume):
        return False
    generation_rows = latest_rows(prepared_dir(config) / "generation_inputs.jsonl", "name")
    if not generation_rows:
        return False
    result_rows = {
        str(row.get("source_id") or ""): row
        for row in latest_rows(robustpa_dir(config) / "results.jsonl", "source_id")
    }
    retry_error_results = bool(config.blueprint.get("retry_error_results", False))
    execution_mode = str(config.blueprint.get("execution_mode", "full"))
    missing_count = 0
    nonterminal_count = 0
    for generation in generation_rows:
        source_id = str(generation.get("name") or "")
        result = result_rows.get(source_id)
        if result is None:
            missing_count += 1
        elif execution_mode == "phase2_only" and str(result.get("status") or "") in {
            "strictAccepted", "acceptedWithWarnings",
        }:
            nonterminal_count += 1
        elif not _blueprint_result_is_terminal(result, retry_error_results=retry_error_results):
            nonterminal_count += 1
    complete = missing_count == 0 and nonterminal_count == 0
    if complete:
        print(
            "[blueprint-skip] all prepared rows already have terminal RobustPA results "
            f"rows={len(generation_rows)} retry_error_results={retry_error_results}",
            flush=True,
        )
    else:
        print(
            "[blueprint-resume] "
            f"prepared={len(generation_rows)} results={len(result_rows)} "
            f"missing={missing_count} nonterminal={nonterminal_count} "
            f"retry_error_results={retry_error_results}",
            flush=True,
        )
    return complete


def run_blueprint(
    config: DictConfig,
    runtime: PersistentVLLMRuntime,
    kimina_runtime: PersistentKiminaRuntime,
) -> None:
    execution_mode = str(config.blueprint.get("execution_mode", "full"))
    if execution_mode != "phase2_only" and blueprint_results_complete(config):
        return
    blueprint = config.blueprint
    runtime.ensure(
        stage="blueprint",
        client_model=str(blueprint.model),
        base_url=str(blueprint.openai_base_url),
        service=blueprint.vllm,
    )
    kimina_runtime.ensure("blueprint")
    preflight_model(str(blueprint.model), str(blueprint.openai_base_url))
    preflight_kimina(config)
    root = output_root(config)
    robustpa_output_base = root / "robustpa"
    phase1_source_root = None
    phase1_seed_root = None
    phase1_source_results_path = None
    if execution_mode == "phase2_only":
        raw_source_root = str(
            blueprint.get("phase1_input_experiment_root", "") or ""
        ).strip()
        if not raw_source_root:
            raise ValueError(
                "phase2_only requires blueprint.phase1_input_experiment_root"
            )
        phase1_source_root = Path(raw_source_root).expanduser().resolve()
        robustpa_data_root = phase1_source_root / "prepared" / "data"
        phase1_seed_root = phase1_source_root / "robustpa" / "blueprint"
        phase1_source_results_path = phase1_seed_root / "results.jsonl"
        for required in (robustpa_data_root, phase1_seed_root, phase1_source_results_path):
            if not required.exists():
                raise FileNotFoundError(f"Phase 1 input artifact is missing: {required}")
    else:
        robustpa_data_root = prepared_dir(config) / "data"
    overrides = [
        "exp_name=blueprint",
        f"data_root={robustpa_data_root}",
        f"output_base={robustpa_output_base}",
        f"model={blueprint.model}",
        f"openai_base_url={blueprint.openai_base_url}",
        f"subset={DATASET_SUBSET}",
        "split=null",
        "limit=null",
        f"problem_id={blueprint.get('problem_id', 'null') or 'null'}",
        "include_source_ids_path=null",
        f"resume={str(bool(config.resume)).lower()}",
        f"retry_error_results={str(bool(blueprint.get('retry_error_results', False))).lower()}",
        f"execution_mode={blueprint.get('execution_mode', 'full')}",
        f"phase1_seed_root={phase1_seed_root or 'null'}",
        f"phase1_source_results_path={phase1_source_results_path or 'null'}",
        f"phase1_source_experiment_root={phase1_source_root or 'null'}",
        f"generation_prompt_profile={blueprint.get('generation_prompt_profile', 'whole_cot_minimal')}",
        f"max_refinement_iterations={blueprint.max_refinement_iterations}",
        f"refinement_max_retries={blueprint.refinement_max_retries}",
        f"generation_max_turns={blueprint.generation_max_turns}",
        f"generation_enable_thinking={str(bool(blueprint.generation_enable_thinking)).lower()}",
        f"generation_temperature={blueprint.generation_temperature}",
        f"generation_top_p={blueprint.generation_top_p}",
        f"generation_top_k={blueprint.generation_top_k}",
        f"generation_min_p={blueprint.generation_min_p}",
        f"generation_presence_penalty={blueprint.generation_presence_penalty}",
        f"generation_repetition_penalty={blueprint.generation_repetition_penalty}",
        f"generation_model_max_context={blueprint.generation_model_max_context}",
        f"generation_context_safety_margin={blueprint.generation_context_safety_margin}",
        f"formal_decompiler_max_tokens={blueprint.formal_decompiler_max_tokens}",
        f"strict_comparator_max_tokens={blueprint.strict_comparator_max_tokens}",
        f"semantic_format_max_attempts={blueprint.semantic_format_max_attempts}",
        f"semantic_audit_mode={blueprint.get('semantic_audit_mode', 'separate')}",
        f"joint_semantic_audit_max_tokens={blueprint.get('joint_semantic_audit_max_tokens', 32768)}",
        f"semantic_audit_enable_thinking={str(bool(blueprint.semantic_audit_enable_thinking)).lower()}",
        f"semantic_audit_temperature={blueprint.semantic_audit_temperature}",
        f"semantic_audit_top_p={blueprint.semantic_audit_top_p}",
        f"semantic_audit_top_k={blueprint.semantic_audit_top_k}",
        f"semantic_audit_min_p={blueprint.semantic_audit_min_p}",
        f"semantic_audit_presence_penalty={blueprint.semantic_audit_presence_penalty}",
        f"semantic_audit_repetition_penalty={blueprint.semantic_audit_repetition_penalty}",
        f"node_max_prove_turns={blueprint.node_max_prove_turns}",
        f"node_max_negation_probe_turns={blueprint.node_max_negation_probe_turns}",
        f"max_tool_calls_per_turn={blueprint.max_tool_calls_per_turn}",
        f"proof_policy={blueprint.get('proof_policy', 'full')}",
        f"critical_negation_max_turns={int(blueprint.get('critical_negation_max_turns', 0))}",
        "node_timeout_s=null",
        "llm_api_timeout_s=null",
        f"phase1_concurrency={blueprint.phase1_concurrency}",
        f"phase2_blueprint_concurrency={blueprint.phase2_blueprint_concurrency}",
        f"phase2_node_concurrency={blueprint.phase2_node_concurrency}",
        f"refine_concurrency={blueprint.refine_concurrency}",
        f"phase2_contract_check_concurrency={blueprint.phase2_contract_check_concurrency}",
        f"lean_api_url={blueprint.lean_api_url}",
        f"lean_server_timeout={blueprint.lean_server_timeout}",
        "lean_server_reuse=true",
        "lean_server_debug=false",
        f"lean_max_inflight_snippets={blueprint.lean_max_inflight_snippets}",
        f"lean_batch_size={blueprint.lean_batch_size}",
        f"lean_global_batching={str(bool(blueprint.lean_global_batching)).lower()}",
        f"lean_parallel_batches={blueprint.lean_parallel_batches}",
        f"lean_batch_wait_ms={blueprint.lean_batch_wait_ms}",
    ]
    env = os.environ.copy()
    env.setdefault("GOEDEL_OPENAI_API_KEY", "dummy")
    env["GOEDEL_PHASE3_MODEL_MAX_CONTEXT"] = str(
        int(blueprint.get("phase3_model_max_context", blueprint.vllm.max_model_len))
    )
    env["GOEDEL_PHASE3_CONTEXT_SAFETY_MARGIN"] = str(
        int(blueprint.get("phase3_context_safety_margin", 512))
    )
    env["GOEDEL_PHASE3_MAX_OUTPUT_CAP"] = str(
        int(blueprint.refinement_max_tokens)
    )
    env["GOEDEL_PHASE3_MIN_OUTPUT_TOKENS"] = str(
        int(blueprint.get("phase3_min_output_tokens", 512))
    )
    env["GOEDEL_TOKENIZER_PATH"] = str(
        blueprint.get("tokenizer_path", blueprint.vllm.model_path)
    )
    env["GOEDEL_PROVER_MAX_TOKENS"] = str(int(blueprint.prover_max_tokens))
    env["GOEDEL_PROVER_LENGTH_RETRY_MAX_TOKENS"] = str(
        int(blueprint.get("prover_length_retry_max_tokens", blueprint.prover_max_tokens))
    )
    command = [
        str(config.python_bin),
        str(REPO_ROOT / "experiments" / "robustpa_refine" / "run_robustpa_refine.py"),
        *overrides,
    ]
    print("[blueprint] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _enabled_refine_variants(config: DictConfig) -> list[tuple[str, DictConfig]]:
    variants = config.refine.get("variants")
    if variants is None:
        return [("blueprint", OmegaConf.create({"enabled": True, "prompt_mode": "blueprint"}))]
    return [
        (str(name), variant)
        for name, variant in variants.items()
        if bool(variant.get("enabled", True))
    ]


def run_stage(
    stage: str,
    config: DictConfig,
    runtime: PersistentVLLMRuntime | None = None,
    kimina_runtime: PersistentKiminaRuntime | None = None,
) -> Any:
    if runtime is None:
        with (
            PersistentVLLMRuntime(config) as owned_runtime,
            PersistentKiminaRuntime(config) as owned_kimina,
        ):
            return run_stage(stage, config, owned_runtime, owned_kimina)
    if kimina_runtime is None:
        with PersistentKiminaRuntime(config) as owned_kimina:
            return run_stage(stage, config, runtime, owned_kimina)
    if stage == "prepare":
        return prepare(config)
    if stage == "blueprint":
        return run_blueprint(config, runtime, kimina_runtime)
    if stage == "export":
        if runtime.server is not None:
            runtime.ensure(
                stage="export",
                client_model=str(config.blueprint.model),
                base_url=str(config.blueprint.openai_base_url),
                service=config.blueprint.vllm,
            )
        kimina_runtime.ensure("export")
        preflight_kimina(config)
        return export_contexts(config)
    if stage == "refine":
        results: dict[str, Any] = {}
        for variant_name, variant_config in _enabled_refine_variants(config):
            runtime.ensure(
                stage=f"refine/{variant_name}",
                client_model=str(config.refine.model),
                base_url=str(config.refine.openai_base_url),
                service=config.refine.vllm,
            )
            preflight_model(
                str(config.refine.model),
                str(config.refine.openai_base_url),
                str(config.refine.api_key),
            )
            results[variant_name] = asyncio.run(
                refine(config, variant_name, variant_config)
            )
        return results
    if stage == "evaluate":
        if bool(config.judge.enabled):
            runtime.ensure(
                stage="evaluate/judge",
                client_model=str(config.judge.model),
                base_url=str(config.judge.openai_base_url),
                service=config.judge.vllm,
            )
            preflight_model(
                str(config.judge.model),
                str(config.judge.openai_base_url),
                str(config.judge.api_key),
            )
        return evaluate(config)
    raise ValueError(f"unknown stage: {stage}")


class ExperimentLock:
    def __init__(self, root: Path) -> None:
        self.path = root / ".run.lock"
        self.handle = None

    def __enter__(self) -> "ExperimentLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.seek(0)
            owner = self.handle.read().strip() or "unknown"
            self.handle.close()
            raise RuntimeError(
                f"Experiment output is already locked: {self.path} owner={owner}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.profile, args.override)
    if (
        args.stage == "phase2-only"
        and str(config.blueprint.get("execution_mode", "full")) != "phase2_only"
    ):
        raise ValueError("phase2-only stage requires execution_mode=phase2_only")
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    previous_handlers: dict[signal.Signals, Any] = {}

    def terminate(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, terminate)
    try:
        with (
            ExperimentLock(root),
            PersistentVLLMRuntime(config) as runtime,
            PersistentKiminaRuntime(config) as kimina_runtime,
        ):
            (root / "config_resolved.yaml").write_text(
                OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
            )
            stages = STAGE_SEQUENCES.get(args.stage, (args.stage,))
            for stage in stages:
                print(f"[stage-start] {stage}", flush=True)
                run_stage(stage, config, runtime, kimina_runtime)
                print(f"[stage-done] {stage}", flush=True)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
