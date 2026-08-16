#!/usr/bin/env python3
"""Run the configurable combined or Definition-then-Node Semantic IR experiment."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow.parquet as pq
import hydra
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
from pydantic import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint import _extract_lean_code  # noqa: E402
from blueprint_generation import generation_request_budget  # noqa: E402
from goedel_prompts import render  # noqa: E402
from kimina_lean_compiler import KiminaLeanCompiler  # noqa: E402

from experiments.semantic_ir_blueprint.conversation import (  # noqa: E402
    capture_chat_once,
    persist_conversation,
    to_jsonable,
    utc_now,
    write_json,
    write_text,
)
from experiments.semantic_ir_blueprint.semantic_ir import (  # noqa: E402
    DefinitionsPayload,
    NodesPayload,
    SemanticIR,
    extract_json_object,
    validate_definition_source_unit_references,
    validate_source_unit_references,
)
from experiments.semantic_ir_blueprint.source_units import (  # noqa: E402
    make_boundary_anchors,
    parse_boundaries,
    source_units_from_boundaries,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = EXPERIMENT_DIR / "prompts"
TERMINAL_STATUSES = {
    "source_split_request_failed",
    "source_split_parse_failed",
    "semantic_ir_request_failed",
    "semantic_ir_parse_failed",
    "semantic_ir_validation_failed",
    "definitions_request_failed",
    "definitions_parse_failed",
    "definitions_validation_failed",
    "nodes_request_failed",
    "nodes_parse_failed",
    "nodes_validation_failed",
    "blueprint_request_failed",
    "blueprint_extract_failed",
    "lean_compile_failed",
    "completed",
}


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def _semantic_ir_generation_mode(config: Mapping[str, Any]) -> str:
    semantic_config = config.get("semantic_ir")
    if not isinstance(semantic_config, Mapping):
        raise ValueError("semantic_ir config must be a mapping")
    mode = str(semantic_config.get("generation_mode", "combined"))
    if mode not in {"combined", "definitions_then_nodes"}:
        raise ValueError(
            "semantic_ir.generation_mode must be 'combined' or 'definitions_then_nodes'"
        )
    return mode


def _config_dict(value: Any) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        loaded = OmegaConf.to_container(value, resolve=True)
        if not isinstance(loaded, dict):
            raise TypeError("Hydra config must resolve to a mapping")
        return loaded
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("config must be a mapping")


def _chat_request(
    *, model: str, messages: list[dict[str, str]], sampling: Mapping[str, Any],
) -> dict[str, Any]:
    sampling = dict(sampling)
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(sampling["temperature"]),
        "max_completion_tokens": int(sampling["max_completion_tokens"]),
        "stream": False,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": bool(sampling.get("enable_thinking", True)),
            },
        },
    }
    for key in ("top_p", "presence_penalty"):
        if key in sampling:
            request[key] = float(sampling[key])
    for key in ("top_k", "min_p", "repetition_penalty"):
        if key in sampling:
            request["extra_body"][key] = sampling[key]
    return request


def _dynamic_sampling(
    sampling: Mapping[str, Any],
    messages: list[dict[str, str]],
    *,
    cfg: Mapping[str, Any],
    budgeter: Callable[..., tuple[int, int]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Attach the exact remaining context budget to one generation request."""
    effective = dict(sampling)
    input_tokens, completion_budget = budgeter(
        messages,
        tokenizer_path=cfg["tokenizer_path"],
        model_max_context=int(cfg["model_max_context"]),
        safety_margin=int(cfg["context_safety_margin"]),
        tools=None,
    )
    if completion_budget < 1:
        raise ValueError(
            f"no completion budget remains: input_tokens={input_tokens}, "
            f"context={cfg['model_max_context']}, margin={cfg['context_safety_margin']}"
        )
    effective["max_completion_tokens"] = completion_budget
    return effective, {
        "input_tokens": input_tokens,
        "model_max_context": int(cfg["model_max_context"]),
        "safety_margin": int(cfg["context_safety_margin"]),
        "max_completion_tokens": completion_budget,
    }


def _assistant_content(response: Any) -> str:
    if response is None:
        return ""
    value = getattr(response.choices[0].message, "content", None)
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(to_jsonable(value), ensure_ascii=False)


def _safe_record_name(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).strip("_")


def load_record(data_path: Path, source_id: str) -> dict[str, Any]:
    table = pq.read_table(data_path)
    matches = [row for row in table.to_pylist() if row.get("name") == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row named {source_id!r}; found {len(matches)}")
    row = matches[0]
    required = ("problem", "claimed_answer", "informal_proof")
    missing = [key for key in required if not isinstance(row.get(key), str) or not row[key]]
    if missing:
        raise ValueError(f"prepared row is missing non-empty fields: {missing}")
    return row


def _without_hash_fields(value: Any) -> Any:
    """Remove compiler-added hash metadata from saved artifacts."""
    if isinstance(value, dict):
        return {
            key: _without_hash_fields(item)
            for key, item in value.items()
            if not any(token in str(key).lower() for token in ("hash", "sha256"))
        }
    if isinstance(value, list):
        return [_without_hash_fields(item) for item in value]
    return value


def _compiler_result(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        payload = asdict(value)
    else:
        payload = to_jsonable(value)
    if not isinstance(payload, dict):
        payload = {"success": False, "raw_output": str(payload)}
    return _without_hash_fields(payload)


def _result_path(record_dir: Path) -> Path:
    return record_dir / "result.json"


def _finish(
    record_dir: Path,
    result: dict[str, Any],
    status: str,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unknown terminal status: {status}")
    result.update({"status": status, "finished_at": utc_now(), "error": error})
    write_json(_result_path(record_dir), result)
    return result


def _complete_conversation(
    *,
    record_dir: Path,
    stage_dir: str,
    conversation: dict[str, Any],
    parse_status: str,
    artifact_path: Path | None = None,
    parse_error: str | None = None,
) -> None:
    conversation["parsed_artifact"] = {
        "status": parse_status,
        "path": (
            str(artifact_path.relative_to(record_dir)) if artifact_path is not None else None
        ),
        "error": parse_error,
    }
    persist_conversation(
        record_dir / stage_dir / "conversation.json",
        record_dir / "conversations.jsonl",
        conversation,
    )


def run_record(
    record: Mapping[str, Any],
    config: Mapping[str, Any] | Any,
    record_dir: Path,
    *,
    client: Any,
    compiler: Any,
    budgeter: Callable[..., tuple[int, int]] = generation_request_budget,
) -> dict[str, Any]:
    """Run one record. Dependencies are injectable for no-network tests."""
    cfg = _config_dict(config)
    source_id = str(record["name"])
    cot = str(record["informal_proof"])
    target_theorem = str(cfg["target_theorem"])
    semantic_ir_generation_mode = _semantic_ir_generation_mode(cfg)
    if record_dir.exists() and any(record_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty record directory: {record_dir}")
    record_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    result: dict[str, Any] = {
        "experiment": cfg["experiment_name"],
        "source_id": source_id,
        "model": cfg["model"],
        "target_theorem": target_theorem,
        "semantic_ir_generation_mode": semantic_ir_generation_mode,
        "started_at": started_at,
        "status": "running",
        "llm_requests_completed": 0,
        "lean_compile_requests": 0,
        "artifacts": {},
    }
    input_payload = {
        "name": source_id,
        "source": record.get("source"),
        "row_index": record.get("row_index"),
        "problem": record["problem"],
        "claimed_answer": record["claimed_answer"],
        "cot": cot,
        "target_theorem": target_theorem,
        "semantic_ir_generation_mode": semantic_ir_generation_mode,
    }
    input_path = record_dir / "input.json"
    write_json(input_path, input_payload)
    result["artifacts"]["input"] = "input.json"

    # Phase A0: model-selected boundaries over mechanically lossless anchors.
    anchors = make_boundary_anchors(cot)
    inventory_path = record_dir / "source_split" / "boundary_inventory.json"
    write_json(inventory_path, {"anchors": anchors})
    result["artifacts"]["boundary_inventory"] = str(inventory_path.relative_to(record_dir))
    split_inventory = "\n".join(
        json.dumps({
            "boundary": anchor["anchor_id"],
            "kind": anchor["kind"],
            "text": anchor["source_text"],
        }, ensure_ascii=False)
        for anchor in anchors
    )
    split_messages = [
        {
            "role": "system",
            "content": render(
                _load_prompt("source_split_system"),
                final_boundary=anchors[-1]["anchor_id"],
                max_units=int(cfg["source_split"]["max_units"]),
            ),
        },
        {
            "role": "user",
            "content": render(
                _load_prompt("source_split_user"),
                boundary_count=len(anchors),
                boundary_inventory=split_inventory,
                max_units=int(cfg["source_split"]["max_units"]),
            ),
        },
    ]
    try:
        split_sampling, split_budget = _dynamic_sampling(
            cfg["source_split"], split_messages, cfg=cfg, budgeter=budgeter,
        )
    except ValueError as exc:
        return _finish(record_dir, result, "source_split_request_failed", error=str(exc))
    split_request = _chat_request(
        model=cfg["model"], messages=split_messages, sampling=split_sampling,
    )
    split_response, split_conversation = capture_chat_once(client, "source_split", split_request)
    split_conversation["token_budget"] = split_budget
    if split_response is None:
        _complete_conversation(
            record_dir=record_dir, stage_dir="source_split", conversation=split_conversation,
            parse_status="not_attempted", parse_error="request failed",
        )
        return _finish(
            record_dir, result, "source_split_request_failed",
            error=split_conversation["exception"]["message"],
        )
    result["llm_requests_completed"] += 1
    try:
        boundaries = parse_boundaries(
            _assistant_content(split_response),
            anchors,
            max_units=int(cfg["source_split"]["max_units"]),
        )
        source_units = source_units_from_boundaries(cot, anchors, boundaries)
    except Exception as exc:
        _complete_conversation(
            record_dir=record_dir, stage_dir="source_split", conversation=split_conversation,
            parse_status="failed", parse_error=str(exc),
        )
        return _finish(record_dir, result, "source_split_parse_failed", error=str(exc))
    units_path = record_dir / "source_units.json"
    write_json(units_path, {"source_units": source_units})
    result["artifacts"]["source_units"] = "source_units.json"
    _complete_conversation(
        record_dir=record_dir, stage_dir="source_split", conversation=split_conversation,
        parse_status="parsed", artifact_path=units_path,
    )

    source_unit_ids = [unit["unit_id"] for unit in source_units]
    if semantic_ir_generation_mode == "combined":
        # Phase A1: Definitions and proof Nodes in one response.
        semantic_messages = [
            {"role": "system", "content": _load_prompt("semantic_ir_system")},
            {
                "role": "user",
                "content": render(
                    _load_prompt("semantic_ir_user"),
                    problem=record["problem"],
                    claimed_answer=record["claimed_answer"],
                    target_theorem=target_theorem,
                    source_units_json=json.dumps(
                        {"source_units": source_units}, ensure_ascii=False, indent=2,
                    ),
                ),
            },
        ]
        try:
            semantic_sampling, semantic_budget = _dynamic_sampling(
                cfg["semantic_ir"], semantic_messages, cfg=cfg, budgeter=budgeter,
            )
        except ValueError as exc:
            return _finish(record_dir, result, "semantic_ir_request_failed", error=str(exc))
        semantic_request = _chat_request(
            model=cfg["model"], messages=semantic_messages, sampling=semantic_sampling,
        )
        semantic_response, semantic_conversation = capture_chat_once(
            client, "semantic_ir", semantic_request,
        )
        semantic_conversation["token_budget"] = semantic_budget
        semantic_raw_path = record_dir / "semantic_ir" / "raw_response.txt"
        if semantic_response is None:
            _complete_conversation(
                record_dir=record_dir, stage_dir="semantic_ir",
                conversation=semantic_conversation,
                parse_status="not_attempted", parse_error="request failed",
            )
            return _finish(
                record_dir, result, "semantic_ir_request_failed",
                error=semantic_conversation["exception"]["message"],
            )
        result["llm_requests_completed"] += 1
        semantic_content = _assistant_content(semantic_response)
        write_text(semantic_raw_path, semantic_content)
        result["artifacts"]["semantic_ir_raw"] = str(
            semantic_raw_path.relative_to(record_dir)
        )
        try:
            semantic_payload = extract_json_object(semantic_content)
        except Exception as exc:
            _complete_conversation(
                record_dir=record_dir, stage_dir="semantic_ir",
                conversation=semantic_conversation,
                parse_status="failed", parse_error=str(exc),
            )
            return _finish(record_dir, result, "semantic_ir_parse_failed", error=str(exc))
        try:
            semantic_ir = SemanticIR.model_validate(semantic_payload, strict=True)
            validate_source_unit_references(semantic_ir, source_unit_ids)
            if semantic_ir.nodes[-1].id != target_theorem:
                raise ValueError(
                    f"final theorem id must be configured target {target_theorem!r}"
                )
        except (ValidationError, ValueError) as exc:
            _complete_conversation(
                record_dir=record_dir, stage_dir="semantic_ir",
                conversation=semantic_conversation,
                parse_status="validation_failed", parse_error=str(exc),
            )
            return _finish(
                record_dir, result, "semantic_ir_validation_failed", error=str(exc),
            )
        ir_path = record_dir / "semantic_ir" / "semantic_ir.json"
        write_json(ir_path, semantic_ir.model_dump(mode="json"))
        result["artifacts"]["semantic_ir"] = str(ir_path.relative_to(record_dir))
        _complete_conversation(
            record_dir=record_dir, stage_dir="semantic_ir",
            conversation=semantic_conversation,
            parse_status="parsed", artifact_path=ir_path,
        )
    else:
        # Phase A1a: generate and validate only the Definition Registry.
        separate_cfg = cfg["semantic_ir"]
        definitions_messages = [
            {"role": "system", "content": _load_prompt("definitions_system")},
            {
                "role": "user",
                "content": render(
                    _load_prompt("definitions_user"),
                    problem=record["problem"],
                    claimed_answer=record["claimed_answer"],
                    source_units_json=json.dumps(
                        {"source_units": source_units}, ensure_ascii=False, indent=2,
                    ),
                ),
            },
        ]
        try:
            definitions_sampling, definitions_budget = _dynamic_sampling(
                separate_cfg["definitions"], definitions_messages,
                cfg=cfg, budgeter=budgeter,
            )
        except (KeyError, ValueError, TypeError) as exc:
            return _finish(record_dir, result, "definitions_request_failed", error=str(exc))
        definitions_request = _chat_request(
            model=cfg["model"], messages=definitions_messages,
            sampling=definitions_sampling,
        )
        definitions_response, definitions_conversation = capture_chat_once(
            client, "definitions", definitions_request,
        )
        definitions_conversation["token_budget"] = definitions_budget
        definitions_raw_path = record_dir / "definitions" / "raw_response.txt"
        if definitions_response is None:
            _complete_conversation(
                record_dir=record_dir, stage_dir="definitions",
                conversation=definitions_conversation,
                parse_status="not_attempted", parse_error="request failed",
            )
            return _finish(
                record_dir, result, "definitions_request_failed",
                error=definitions_conversation["exception"]["message"],
            )
        result["llm_requests_completed"] += 1
        definitions_content = _assistant_content(definitions_response)
        write_text(definitions_raw_path, definitions_content)
        result["artifacts"]["definitions_raw"] = str(
            definitions_raw_path.relative_to(record_dir)
        )
        try:
            definitions_json = extract_json_object(definitions_content)
        except Exception as exc:
            _complete_conversation(
                record_dir=record_dir, stage_dir="definitions",
                conversation=definitions_conversation,
                parse_status="failed", parse_error=str(exc),
            )
            return _finish(record_dir, result, "definitions_parse_failed", error=str(exc))
        try:
            definitions_payload = DefinitionsPayload.model_validate(
                definitions_json, strict=True,
            )
            validate_definition_source_unit_references(
                definitions_payload.definitions, source_unit_ids,
            )
        except (ValidationError, ValueError) as exc:
            _complete_conversation(
                record_dir=record_dir, stage_dir="definitions",
                conversation=definitions_conversation,
                parse_status="validation_failed", parse_error=str(exc),
            )
            return _finish(
                record_dir, result, "definitions_validation_failed", error=str(exc),
            )
        definitions_path = record_dir / "definitions" / "definitions.json"
        write_json(definitions_path, definitions_payload.model_dump(mode="json"))
        result["artifacts"]["definitions"] = str(
            definitions_path.relative_to(record_dir)
        )
        _complete_conversation(
            record_dir=record_dir, stage_dir="definitions",
            conversation=definitions_conversation,
            parse_status="parsed", artifact_path=definitions_path,
        )

        # Phase A1b: generate proof Nodes using the frozen Definition Registry.
        nodes_messages = [
            {"role": "system", "content": _load_prompt("nodes_system")},
            {
                "role": "user",
                "content": render(
                    _load_prompt("nodes_user"),
                    problem=record["problem"],
                    claimed_answer=record["claimed_answer"],
                    target_theorem=target_theorem,
                    source_units_json=json.dumps(
                        {"source_units": source_units}, ensure_ascii=False, indent=2,
                    ),
                    definitions_json=json.dumps(
                        definitions_payload.model_dump(mode="json"),
                        ensure_ascii=False, indent=2,
                    ),
                ),
            },
        ]
        try:
            nodes_sampling, nodes_budget = _dynamic_sampling(
                separate_cfg["nodes"], nodes_messages, cfg=cfg, budgeter=budgeter,
            )
        except (KeyError, ValueError, TypeError) as exc:
            return _finish(record_dir, result, "nodes_request_failed", error=str(exc))
        nodes_request = _chat_request(
            model=cfg["model"], messages=nodes_messages, sampling=nodes_sampling,
        )
        nodes_response, nodes_conversation = capture_chat_once(
            client, "nodes", nodes_request,
        )
        nodes_conversation["token_budget"] = nodes_budget
        nodes_raw_path = record_dir / "nodes" / "raw_response.txt"
        if nodes_response is None:
            _complete_conversation(
                record_dir=record_dir, stage_dir="nodes", conversation=nodes_conversation,
                parse_status="not_attempted", parse_error="request failed",
            )
            return _finish(
                record_dir, result, "nodes_request_failed",
                error=nodes_conversation["exception"]["message"],
            )
        result["llm_requests_completed"] += 1
        nodes_content = _assistant_content(nodes_response)
        write_text(nodes_raw_path, nodes_content)
        result["artifacts"]["nodes_raw"] = str(nodes_raw_path.relative_to(record_dir))
        try:
            nodes_json = extract_json_object(nodes_content)
        except Exception as exc:
            _complete_conversation(
                record_dir=record_dir, stage_dir="nodes", conversation=nodes_conversation,
                parse_status="failed", parse_error=str(exc),
            )
            return _finish(record_dir, result, "nodes_parse_failed", error=str(exc))
        try:
            nodes_payload = NodesPayload.model_validate(nodes_json, strict=True)
            semantic_ir = SemanticIR(
                definitions=definitions_payload.definitions,
                nodes=nodes_payload.nodes,
            )
            validate_source_unit_references(semantic_ir, source_unit_ids)
            if semantic_ir.nodes[-1].id != target_theorem:
                raise ValueError(
                    f"final theorem id must be configured target {target_theorem!r}"
                )
        except (ValidationError, ValueError) as exc:
            _complete_conversation(
                record_dir=record_dir, stage_dir="nodes", conversation=nodes_conversation,
                parse_status="validation_failed", parse_error=str(exc),
            )
            return _finish(record_dir, result, "nodes_validation_failed", error=str(exc))
        nodes_path = record_dir / "nodes" / "nodes.json"
        write_json(nodes_path, nodes_payload.model_dump(mode="json"))
        result["artifacts"]["nodes"] = str(nodes_path.relative_to(record_dir))
        ir_path = record_dir / "semantic_ir" / "semantic_ir.json"
        write_json(ir_path, semantic_ir.model_dump(mode="json"))
        result["artifacts"]["semantic_ir"] = str(ir_path.relative_to(record_dir))
        _complete_conversation(
            record_dir=record_dir, stage_dir="nodes", conversation=nodes_conversation,
            parse_status="parsed", artifact_path=nodes_path,
        )

    # Phase B: only the target identifier and Semantic IR are model-visible.
    blueprint_messages = [
        {"role": "system", "content": _load_prompt("blueprint_system")},
        {
            "role": "user",
            "content": render(
                _load_prompt("blueprint_user"),
                target_theorem=target_theorem,
                semantic_ir_json=json.dumps(
                    semantic_ir.model_dump(mode="json"), ensure_ascii=False, indent=2,
                ),
            ),
        },
    ]
    try:
        blueprint_cfg, blueprint_budget = _dynamic_sampling(
            cfg["blueprint"], blueprint_messages, cfg=cfg, budgeter=budgeter,
        )
    except ValueError as exc:
        blueprint_conversation = {
            "stage": "blueprint",
            "started_at": utc_now(), "finished_at": utc_now(), "latency_seconds": 0.0,
            "request": None, "raw_response": None,
            "assistant_reasoning_content": None, "assistant_content": None,
            "finish_reason": None, "usage": None,
            "exception": {"type": "ContextBudgetError", "message": str(exc)},
            "parsed_artifact": {"status": "not_attempted", "path": None, "error": str(exc)},
            "token_budget": None,
        }
        _complete_conversation(
            record_dir=record_dir, stage_dir="blueprint", conversation=blueprint_conversation,
            parse_status="not_attempted", parse_error=str(exc),
        )
        return _finish(record_dir, result, "blueprint_request_failed", error=str(exc))
    blueprint_request = _chat_request(
        model=cfg["model"], messages=blueprint_messages, sampling=blueprint_cfg,
    )
    blueprint_response, blueprint_conversation = capture_chat_once(
        client, "blueprint", blueprint_request,
    )
    blueprint_conversation["token_budget"] = blueprint_budget
    blueprint_raw_path = record_dir / "blueprint" / "raw_response.txt"
    if blueprint_response is None:
        _complete_conversation(
            record_dir=record_dir, stage_dir="blueprint", conversation=blueprint_conversation,
            parse_status="not_attempted", parse_error="request failed",
        )
        return _finish(
            record_dir, result, "blueprint_request_failed",
            error=blueprint_conversation["exception"]["message"],
        )
    result["llm_requests_completed"] += 1
    blueprint_content = _assistant_content(blueprint_response)
    write_text(blueprint_raw_path, blueprint_content)
    result["artifacts"]["blueprint_raw"] = str(blueprint_raw_path.relative_to(record_dir))
    try:
        lean_code = _extract_lean_code(blueprint_content).strip()
        if not lean_code or "@[blueprint" not in lean_code or not re.search(r"\btheorem\b", lean_code):
            raise ValueError("response does not contain a complete Lean Blueprint")
    except Exception as exc:
        _complete_conversation(
            record_dir=record_dir, stage_dir="blueprint", conversation=blueprint_conversation,
            parse_status="failed", parse_error=str(exc),
        )
        return _finish(record_dir, result, "blueprint_extract_failed", error=str(exc))
    lean_path = record_dir / "blueprint" / "blueprint.lean"
    write_text(lean_path, lean_code + "\n")
    result["artifacts"]["blueprint"] = str(lean_path.relative_to(record_dir))
    _complete_conversation(
        record_dir=record_dir, stage_dir="blueprint", conversation=blueprint_conversation,
        parse_status="parsed", artifact_path=lean_path,
    )

    # Exactly one compiler call; its diagnostics are saved even on failure.
    result["lean_compile_requests"] = 1
    compile_started_at = utc_now()
    compile_started = time.monotonic()
    try:
        checked = compiler.check_blueprint(lean_code, target_theorem)
        lean_result = _compiler_result(checked)
        lean_result["exception"] = None
    except Exception as exc:
        lean_result = {
            "success": False,
            "goals": [],
            "errors": [str(exc)],
            "warnings": [],
            "raw_output": "",
            "failure_kind": "infra",
            "timings": {},
            "exception": {"type": type(exc).__name__, "message": str(exc)},
        }
    lean_result["started_at"] = compile_started_at
    lean_result["finished_at"] = utc_now()
    lean_result["latency_seconds"] = time.monotonic() - compile_started
    lean_result_path = record_dir / "blueprint" / "lean_result.json"
    lean_result = _without_hash_fields(lean_result)
    write_json(lean_result_path, lean_result)
    result["artifacts"]["lean_result"] = str(lean_result_path.relative_to(record_dir))
    if not bool(lean_result.get("success")):
        return _finish(
            record_dir, result, "lean_compile_failed",
            error="Lean compilation or Blueprint validation failed",
        )
    return _finish(record_dir, result, "completed")


def run_from_config(config_value: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    config = _config_dict(config_value)
    record = load_record(Path(config["data_path"]), str(config["source_id"]))
    output_root = Path(config["output_root"]) / str(config["experiment_name"])
    record_dir = output_root / "records" / _safe_record_name(str(config["source_id"]))

    api_key_env = str(config.get("openai_api_key_env", "GOEDEL_OPENAI_API_KEY"))
    client = OpenAI(
        base_url=str(config["openai_base_url"]).rstrip("/"),
        api_key=os.environ.get(api_key_env, "dummy"),
        timeout=float(config.get("llm_timeout_seconds", 600)),
        max_retries=0,
    )
    compiler_cfg = dict(config["kimina"])
    compiler = KiminaLeanCompiler(
        api_url=str(compiler_cfg["api_url"]),
        timeout_s=int(compiler_cfg.get("timeout_seconds", 300)),
        reuse=bool(compiler_cfg.get("reuse", True)),
        max_inflight_snippets=1,
        batch_size=1,
        global_batching=False,
        retry_delays_s=(),
        retry_jitter_s=0.0,
    )
    try:
        return run_record(record, config, record_dir, client=client, compiler=compiler)
    finally:
        compiler.close()
        client.close()


@hydra.main(version_base=None, config_path="config", config_name="counting_probability_731")
def main(cfg: DictConfig) -> None:
    result = run_from_config(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
