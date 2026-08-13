from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from blueprint import Blueprint  # noqa: E402
from kimina_lean_compiler import KiminaLeanCompiler  # noqa: E402
from orchestrator import active_node_names  # noqa: E402
from pipeline import _assemble_final_file  # noqa: E402
from prover import NEGATION_SEMANTICS  # noqa: E402
from input_loader import AcceptedBlueprint, load_accepted_blueprints  # noqa: E402
from node_context import build_node_problem  # noqa: E402
from stepfun_repl_prover import ProverOutcome, StepFunReplProver  # noqa: E402
from goedel_self_correct_prover import GoedelSelfCorrectProver  # noqa: E402


@dataclass
class RecordRuntime:
    source: AcceptedBlueprint
    checkpoint_path: Path
    data: dict[str, Any]

    @property
    def blueprint(self) -> Blueprint:
        return self.source.blueprint

    @property
    def positive(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("positive", {})

    @property
    def negative(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("negative", {})

    @property
    def proved_cache(self) -> dict[str, str]:
        return self.data.setdefault("proved_cache", {})


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "record"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError("config root must be a mapping")
    return value


def new_checkpoint(source: AcceptedBlueprint) -> dict[str, Any]:
    active = active_node_names(source.blueprint)
    return {
        "schema_version": 2,
        "negation_semantics": NEGATION_SEMANTICS,
        "record_id": source.record_id,
        "source_id": source.source_id,
        "subset": source.subset,
        "split": source.split,
        "source_checkpoint": str(source.checkpoint_path),
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "target_theorem": source.blueprint.target_theorem,
        "active_nodes": sorted(active),
        "positive": {},
        "negative": {},
        "proved_cache": {},
        "final": {},
    }


def load_runtime(source: AcceptedBlueprint, output_root: Path, resume: bool) -> RecordRuntime:
    checkpoint = output_root / "checkpoints" / source.subset / source.split / f"{safe_name(source.record_id)}.json"
    if checkpoint.is_file() and resume:
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        if data.get("source_checkpoint_sha256") != source.checkpoint_sha256:
            raise ValueError(f"source checkpoint drift for {source.key}")
        if data.get("negation_semantics") != NEGATION_SEMANTICS:
            raise ValueError(
                f"checkpoint uses legacy negation semantics for {source.key}; "
                "use a fresh output_root or --no-resume"
            )
    else:
        data = new_checkpoint(source)
        atomic_json(checkpoint, data)
    return RecordRuntime(source, checkpoint, data)


def proof_nodes(runtime: RecordRuntime) -> list[str]:
    active = set(runtime.data["active_nodes"])
    return [
        node.name for node in runtime.blueprint.dependency_order()
        if node.name in active and node.kind in {"lemma", "theorem"}
    ]


def unresolved_direct_dependencies(runtime: RecordRuntime, node_name: str) -> list[str]:
    node = runtime.blueprint.node_by_name(node_name)
    assert node is not None
    active = set(runtime.data["active_nodes"])
    unresolved: list[str] = []
    for name in node.dependencies:
        dependency = runtime.blueprint.node_by_name(name)
        if name in active and dependency is not None and dependency.kind != "definition":
            if runtime.positive.get(name, {}).get("status") != "solved":
                unresolved.append(name)
    return unresolved


def trace_path(output_root: Path, runtime: RecordRuntime, stage: str, node_name: str) -> Path:
    return (
        output_root / "conversations" / runtime.source.subset / runtime.source.split
        / safe_name(runtime.source.record_id) / f"{safe_name(node_name)}.{stage}.json"
    )


def persist_outcome(
    output_root: Path,
    runtime: RecordRuntime,
    node_name: str,
    stage: str,
    outcome: ProverOutcome,
) -> None:
    full = outcome.to_dict()
    trajectory = full.pop("trajectory")
    conversation_path = trace_path(output_root, runtime, stage, node_name)
    atomic_json(conversation_path, {
        "record_id": runtime.source.record_id,
        "node_name": node_name,
        "stage": stage,
        "negation_semantics": NEGATION_SEMANTICS if stage == "negative" else None,
        "outcome": full,
        "trajectory": trajectory,
    })
    full["conversation_path"] = str(conversation_path)
    if stage == "positive":
        runtime.positive[node_name] = full
        if outcome.status == "solved":
            runtime.proved_cache[node_name] = outcome.proof_body
    else:
        runtime.negative[node_name] = {
            **full,
            "status": "formally_negated" if outcome.status == "solved" else "not_proved",
            "prover_status": outcome.status,
        }
    atomic_json(runtime.checkpoint_path, runtime.data)


async def run_positive(
    runtimes: list[RecordRuntime],
    prover: Any,
    output_root: Path,
) -> None:
    inflight: dict[asyncio.Task[ProverOutcome], tuple[RecordRuntime, str]] = {}
    scheduled: set[tuple[str, str]] = set()
    total_nodes = sum(len(proof_nodes(runtime)) for runtime in runtimes)
    completed_nodes = sum(len(runtime.positive) for runtime in runtimes)
    statuses = Counter(
        result.get("status", "unknown")
        for runtime in runtimes for result in runtime.positive.values()
    )
    progress = tqdm(
        total=total_nodes,
        initial=completed_nodes,
        desc="positive nodes",
        unit="node",
        dynamic_ncols=True,
    )
    progress.set_postfix(dict(statuses), refresh=False)

    def enqueue_ready() -> int:
        added = 0
        for runtime in runtimes:
            for name in proof_nodes(runtime):
                key = (runtime.source.record_id, name)
                if name in runtime.positive or key in scheduled:
                    continue
                if unresolved_direct_dependencies(runtime, name):
                    continue
                problem = build_node_problem(
                    runtime.blueprint, name, runtime.proved_cache, stage="positive",
                )
                task = asyncio.create_task(prover.prove(problem))
                inflight[task] = (runtime, name)
                scheduled.add(key)
                added += 1
        return added

    try:
        enqueue_ready()
        while inflight:
            done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                runtime, name = inflight.pop(task)
                outcome = task.result()
                persist_outcome(output_root, runtime, name, "positive", outcome)
                statuses[outcome.status] += 1
                progress.update(1)
                progress.set_postfix(dict(statuses), refresh=False)
                tqdm.write(
                    f"[positive] {runtime.source.record_id} node={name} "
                    f"status={outcome.status} turns={outcome.turns}"
                )
            enqueue_ready()

        for runtime in runtimes:
            blocked = 0
            for name in proof_nodes(runtime):
                if name in runtime.positive:
                    continue
                unresolved = unresolved_direct_dependencies(runtime, name)
                runtime.positive[name] = {
                    "status": "blocked_by_dependency",
                    "proof_body": "",
                    "lean_errors": [f"Unresolved dependencies: {', '.join(unresolved)}"],
                    "turns": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed_seconds": 0.0,
                }
                blocked += 1
            if blocked:
                statuses["blocked_by_dependency"] += blocked
                progress.update(blocked)
                progress.set_postfix(dict(statuses), refresh=False)
            atomic_json(runtime.checkpoint_path, runtime.data)
    finally:
        progress.close()


async def run_negative(
    runtimes: list[RecordRuntime],
    prover: Any,
    output_root: Path,
) -> None:
    tasks: dict[asyncio.Task[ProverOutcome], tuple[RecordRuntime, str]] = {}
    excluded = {"solved", "blocked_by_dependency", "infra_error"}
    eligible = [
        (runtime, name)
        for runtime in runtimes
        for name, positive in runtime.positive.items()
        if positive.get("status") not in excluded
    ]
    completed = sum(name in runtime.negative for runtime, name in eligible)
    statuses = Counter(
        runtime.negative[name].get("status", "unknown")
        for runtime, name in eligible if name in runtime.negative
    )
    progress = tqdm(
        total=len(eligible),
        initial=completed,
        desc="negative nodes",
        unit="node",
        dynamic_ncols=True,
    )
    progress.set_postfix(dict(statuses), refresh=False)
    for runtime in runtimes:
        for name, positive in runtime.positive.items():
            if positive.get("status") in excluded or name in runtime.negative:
                continue
            problem = build_node_problem(
                runtime.blueprint, name, runtime.proved_cache, stage="negative",
            )
            tasks[asyncio.create_task(prover.prove(problem))] = (runtime, name)
    try:
        while tasks:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                runtime, name = tasks.pop(task)
                outcome = task.result()
                persist_outcome(output_root, runtime, name, "negative", outcome)
                statuses[runtime.negative[name]["status"]] += 1
                progress.update(1)
                progress.set_postfix(dict(statuses), refresh=False)
                tqdm.write(
                    f"[negative] {runtime.source.record_id} node={name} "
                    f"status={runtime.negative[name]['status']} prover={outcome.status}"
                )
    finally:
        progress.close()


async def verify_finals(
    runtimes: list[RecordRuntime],
    compiler: KiminaLeanCompiler,
    output_root: Path,
) -> None:
    progress = tqdm(
        total=len(runtimes),
        desc="final blueprints",
        unit="blueprint",
        dynamic_ncols=True,
    )
    statuses: Counter[str] = Counter()

    async def verify(runtime: RecordRuntime) -> None:
        try:
            root = runtime.blueprint.target_theorem
            if runtime.positive.get(root, {}).get("status") != "solved":
                runtime.data["final"] = {
                    "status": "root_not_solved",
                    "root_status": runtime.positive.get(root, {}).get("status", "missing"),
                }
                atomic_json(runtime.checkpoint_path, runtime.data)
                return
            try:
                lean = _assemble_final_file(runtime.blueprint, None, runtime.proved_cache)
            except Exception as exc:  # noqa: BLE001
                runtime.data["final"] = {
                    "status": "assembly_error", "errors": [f"{type(exc).__name__}: {exc}"],
                }
                atomic_json(runtime.checkpoint_path, runtime.data)
                return
            result = await asyncio.to_thread(compiler.check, lean, False)
            lean_path = (
                output_root / "final_lean" / runtime.source.subset / runtime.source.split
                / f"{safe_name(runtime.source.record_id)}.lean"
            )
            lean_path.parent.mkdir(parents=True, exist_ok=True)
            lean_path.write_text(lean, encoding="utf-8")
            runtime.data["final"] = {
                "status": "solved" if result.success and not result.has_sorry else (
                    "infra_error" if result.failure_kind == "infra" else "final_assembly_failed"
                ),
                "lean_path": str(lean_path),
                "errors": result.diagnostics,
                "warnings": result.warnings,
                "timings": result.timings,
            }
            atomic_json(runtime.checkpoint_path, runtime.data)
        finally:
            statuses[runtime.data.get("final", {}).get("status", "unknown")] += 1
            progress.update(1)
            progress.set_postfix(dict(statuses), refresh=False)

    try:
        await asyncio.gather(*(verify(runtime) for runtime in runtimes))
    finally:
        progress.close()


def result_row(runtime: RecordRuntime) -> dict[str, Any]:
    positive_counts = Counter(row.get("status", "unknown") for row in runtime.positive.values())
    negative_counts = Counter(row.get("status", "unknown") for row in runtime.negative.values())
    return {
        "record_id": runtime.source.record_id,
        "source_id": runtime.source.source_id,
        "subset": runtime.source.subset,
        "split": runtime.source.split,
        "source_checkpoint": str(runtime.source.checkpoint_path),
        "source_checkpoint_sha256": runtime.source.checkpoint_sha256,
        "negation_semantics": NEGATION_SEMANTICS,
        "checkpoint_path": str(runtime.checkpoint_path),
        "target_theorem": runtime.blueprint.target_theorem,
        "active_node_count": len(runtime.data["active_nodes"]),
        "proof_node_count": len(proof_nodes(runtime)),
        "positive_counts": dict(positive_counts),
        "negative_counts": dict(negative_counts),
        "root_status": runtime.positive.get(runtime.blueprint.target_theorem, {}).get("status", "missing"),
        "final_status": runtime.data.get("final", {}).get("status", "missing"),
        "final": runtime.data.get("final", {}),
    }


def summarize(rows: list[dict[str, Any]], runtimes: list[RecordRuntime]) -> dict[str, Any]:
    positive = Counter()
    negative = Counter()
    turns: list[int] = []
    completion_tokens = 0
    for runtime in runtimes:
        for result in runtime.positive.values():
            positive[result.get("status", "unknown")] += 1
            if result.get("turns"):
                turns.append(int(result["turns"]))
            completion_tokens += int(result.get("completion_tokens") or 0)
        for result in runtime.negative.values():
            negative[result.get("status", "unknown")] += 1
            if result.get("turns"):
                turns.append(int(result["turns"]))
            completion_tokens += int(result.get("completion_tokens") or 0)
    final_counts = Counter(row["final_status"] for row in rows)
    total = len(rows)
    return {
        "negation_semantics": NEGATION_SEMANTICS,
        "blueprints": total,
        "blueprint_solved": final_counts["solved"],
        "blueprint_pass_rate": final_counts["solved"] / total if total else 0.0,
        "root_solved": sum(row["root_status"] == "solved" for row in rows),
        "positive_counts": dict(positive),
        "positive_attempted": sum(value for key, value in positive.items() if key != "blocked_by_dependency"),
        "positive_node_solve_rate_all": positive["solved"] / sum(positive.values()) if positive else 0.0,
        "negative_counts": dict(negative),
        "formally_negated": negative["formally_negated"],
        "final_counts": dict(final_counts),
        "avg_attempt_turns": sum(turns) / len(turns) if turns else 0.0,
        "completion_tokens": completion_tokens,
    }


async def preflight(config: dict[str, Any]) -> None:
    model_url = str(config["model"]["api_base_url"]).rstrip("/") + "/models"
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(model_url)
        response.raise_for_status()
        served = {
            str(item.get("id")) for item in (response.json().get("data") or [])
            if isinstance(item, dict)
        }
        expected = str(config["model"]["name"])
        if expected not in served:
            raise RuntimeError(
                f"vLLM serves {sorted(served)}, but config requires {expected}"
            )
        health = await client.get(str(config["lean"]["api_url"]).rstrip("/") + "/health")
        health.raise_for_status()


async def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.output_root:
        config["output_root"] = str(args.output_root.resolve())
    limit = args.limit if args.limit is not None else config.get("run", {}).get("limit")
    include_ids = set(args.include_id or []) or None
    source_root = Path(config["source_experiment_root"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    await preflight(config)
    accepted = load_accepted_blueprints(
        source_root, limit=None if limit is None else int(limit), include_ids=include_ids,
    )
    if not accepted:
        raise RuntimeError("no strictAccepted Blueprints selected")
    resume = bool(config.get("run", {}).get("resume", True)) and not args.no_resume
    runtimes = [load_runtime(item, output_root, resume) for item in accepted]

    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    model_config = config["model"]
    client = AsyncOpenAI(
        base_url=str(model_config["api_base_url"]),
        api_key=str(model_config.get("api_key") or "EMPTY"),
        timeout=None,
        max_retries=0,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_config["path"]), trust_remote_code=True)
    lean_config = config["lean"]
    compiler = KiminaLeanCompiler(
        api_url=str(lean_config["api_url"]),
        timeout_s=int(lean_config["server_timeout_seconds"]),
        reuse=bool(lean_config["reuse"]),
        debug=bool(lean_config["debug"]),
        max_inflight_snippets=int(lean_config["max_inflight_snippets"]),
        batch_size=int(lean_config["batch_size"]),
        global_batching=bool(lean_config["global_batching"]),
        parallel_batches=int(lean_config["parallel_batches"]),
        batch_wait_ms=float(lean_config["batch_wait_ms"]),
    )
    protocol = str(model_config.get("protocol") or "stepfun_repl")
    prover_cls = {
        "stepfun_repl": StepFunReplProver,
        "goedel_self_correct": GoedelSelfCorrectProver,
    }.get(protocol)
    if prover_cls is None:
        raise ValueError(f"unsupported model protocol: {protocol}")
    prover = prover_cls(client=client, tokenizer=tokenizer, compiler=compiler, config=model_config)
    started = time.time()
    atomic_json(output_root / "manifest.json", {
        "started_at": started,
        "source_root": str(source_root),
        "selected_records": len(accepted),
        "selected_ids": [item.record_id for item in accepted],
        "negation_semantics": NEGATION_SEMANTICS,
        "config": config,
    })
    try:
        print(f"[run] selected strictAccepted Blueprints={len(runtimes)}", flush=True)
        await run_positive(runtimes, prover, output_root)
        print("[run] positive phase complete; starting all eligible negative probes", flush=True)
        await run_negative(runtimes, prover, output_root)
        print("[run] negative phase complete; verifying assembled files", flush=True)
        await verify_finals(runtimes, compiler, output_root)
    finally:
        await client.close()
        compiler.close()
    rows = [result_row(runtime) for runtime in runtimes]
    summary = summarize(rows, runtimes)
    summary["wall_seconds"] = time.time() - started
    summary["lean_runtime"] = compiler.stats()
    write_jsonl(output_root / "results.jsonl", rows)
    atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=HERE / "configs" / "stepfun_7b_wrong76_whole_cot.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-id", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
