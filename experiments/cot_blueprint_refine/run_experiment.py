from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
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
from kimina_lean_compiler import KiminaLeanCompiler  # noqa: E402


STAGES = ("prepare", "blueprint", "export", "refine", "evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blueprint-guided COT refinement experiment")
    parser.add_argument("--profile", default="base", help="Config profile under configs/")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
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
        result = compiler.check("import Mathlib\nexample : True := by trivial\n")
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
    if bool(row.get("root_proved")) or status in {"solved", "exhausted"}:
        return True
    return status == "error" and not retry_error_results


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
    missing_count = 0
    nonterminal_count = 0
    for generation in generation_rows:
        source_id = str(generation.get("name") or "")
        result = result_rows.get(source_id)
        if result is None:
            missing_count += 1
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


def run_blueprint(config: DictConfig) -> None:
    if blueprint_results_complete(config):
        return
    preflight_model(
        str(config.blueprint.model),
        str(config.blueprint.openai_base_url),
    )
    preflight_kimina(config)
    root = output_root(config)
    robustpa_output_base = root / "robustpa"
    blueprint = config.blueprint
    overrides = [
        "exp_name=blueprint",
        f"data_root={prepared_dir(config) / 'data'}",
        f"output_base={robustpa_output_base}",
        f"model={blueprint.model}",
        f"openai_base_url={blueprint.openai_base_url}",
        f"subset={DATASET_SUBSET}",
        "split=null",
        "limit=null",
        "problem_id=null",
        f"resume={str(bool(config.resume)).lower()}",
        f"max_refinement_iterations={blueprint.max_refinement_iterations}",
        f"blueprint_max_retries={blueprint.blueprint_max_retries}",
        f"node_max_prove_turns={blueprint.node_max_prove_turns}",
        f"max_tool_calls_per_turn={blueprint.max_tool_calls_per_turn}",
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
    ]
    env = os.environ.copy()
    env.setdefault("GOEDEL_OPENAI_API_KEY", "dummy")
    env["GOEDEL_BLUEPRINT_MAX_TOKENS"] = str(int(blueprint.generation_max_tokens))
    env["GOEDEL_PROVER_MAX_TOKENS"] = str(int(blueprint.prover_max_tokens))
    command = [
        str(config.python_bin),
        str(REPO_ROOT / "experiments" / "robustpa_refine" / "run_robustpa_refine.py"),
        *overrides,
    ]
    print("[blueprint] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def run_stage(stage: str, config: DictConfig) -> Any:
    if stage == "prepare":
        return prepare(config)
    if stage == "blueprint":
        return run_blueprint(config)
    if stage == "export":
        preflight_kimina(config)
        return export_contexts(config)
    if stage == "refine":
        preflight_model(
            str(config.refine.model),
            str(config.refine.openai_base_url),
            str(config.refine.api_key),
        )
        return asyncio.run(refine(config))
    if stage == "evaluate":
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
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    with ExperimentLock(root):
        (root / "config_resolved.yaml").write_text(
            OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
        )
        stages = STAGES if args.stage == "all" else (args.stage,)
        for stage in stages:
            print(f"[stage-start] {stage}", flush=True)
            run_stage(stage, config)
            print(f"[stage-done] {stage}", flush=True)


if __name__ == "__main__":
    main()
