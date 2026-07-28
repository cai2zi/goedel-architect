from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


# Copy/paste prompt for chat analysis:
#
# 你是 Lean/Mathlib/自动定理证明实验分析专家。我会给你一个 parquet 聚合文件，
# 其中每一行都是实验目录中的一条原始信息或一个原始文件片段；尤其是
# artifact_type='trace_event' 的行包含 traces/**/*.jsonl 中逐条原封不动的事件。
# 请不要只做泛泛统计。请完整读取每条 row 的 raw_text/raw_json 字段，并把
# results.jsonl、rounds.jsonl、trace_event、checkpoint、blueprint、metrics 等信息
# 按 exp_id/source_id/record_id/id/theorem_name 关联起来。

# 请输出：
# 1. 当前实验的全量数据概览：输入文件、样本数、成功/失败数、trace 覆盖率、事件类型、
#    工具调用、Lean 检查、LLM usage、关键异常。
# 2. 对每个失败样本逐条分析失败原因：必须引用对应的 source_id/record_id/id，
#    并说明证据来自哪些 rel_path、trace_event_index、round_iteration 或 raw row。
# 3. 给出错误类别分类。类别不要预设得过粗；请根据逐条 trace/round/result 证据归纳，
#    例如 blueprint/statement 错误、Lean API/mathlib 名称错误、上下文符号丢失、
#    tactic 卡住、依赖级联、LLM 输出格式错误、工具/超时/上下文溢出、重复搜索循环等。
# 4. 对每个类别给出：类别名、判定标准、样本 ID 列表、数量、典型案例 2-5 个。
#    典型案例必须包含 ID、关键 trace 证据、Lean error 或 llm_error 原文片段、
#    以及为什么它代表该类别。
# 5. 最后给出可执行改进建议，并把每条建议连接到具体类别和具体 ID，不要只给通用建议。

# 注意：raw_text/raw_json 是证据源，不能因为 summary 字段较短就忽略原文。


TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".lean",
    ".txt",
    ".md",
    ".csv",
    ".yaml",
    ".yml",
    ".toml",
    ".log",
}


SCHEMA = pa.schema(
    [
        ("exp_dir", pa.string()),
        ("exp_name", pa.string()),
        ("generated_at_unix", pa.float64()),
        ("artifact_type", pa.string()),
        ("rel_path", pa.string()),
        ("file_path", pa.string()),
        ("file_index", pa.int64()),
        ("row_index", pa.int64()),
        ("trace_event_index", pa.int64()),
        ("id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("subset", pa.string()),
        ("split", pa.string()),
        ("theorem_name", pa.string()),
        ("thm_name", pa.string()),
        ("phase", pa.string()),
        ("iteration", pa.int64()),
        ("turn", pa.int64()),
        ("kind", pa.string()),
        ("tool_name", pa.string()),
        ("call_id", pa.string()),
        ("ok", pa.string()),
        ("status", pa.string()),
        ("root_proved", pa.string()),
        ("error_category_hint", pa.string()),
        ("raw_text", pa.string()),
        ("raw_json", pa.string()),
        ("args_json", pa.string()),
        ("result_text", pa.string()),
        ("parse_error", pa.string()),
    ]
)


def _jsonl_lines(path: Path) -> Iterable[tuple[int, str, dict[str, Any] | None, str]]:
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                yield row_index, raw, json.loads(raw), ""
            except json.JSONDecodeError as exc:
                yield row_index, raw, None, str(exc)


def _read_text(path: Path) -> tuple[str, str]:
    try:
        return path.read_text(encoding="utf-8"), ""
    except UnicodeDecodeError as exc:
        return "", f"UnicodeDecodeError: {exc}"


def _json_dumps(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trace_ids_from_path(exp_dir: Path, path: Path) -> dict[str, str]:
    try:
        rel = path.relative_to(exp_dir / "traces")
        parts = rel.parts
    except ValueError:
        parts = ()
    subset = parts[0] if len(parts) >= 3 else ""
    split = parts[1] if len(parts) >= 3 else ""
    record_id = path.stem
    return {"subset": subset, "split": split, "record_id": record_id}


def _record_keys(row: dict[str, Any] | None) -> dict[str, str]:
    row = row or {}
    return {
        "id": _str(row.get("id")),
        "record_id": _str(row.get("record_id")),
        "source_id": _str(row.get("source_id")),
        "subset": _str(row.get("subset")),
        "split": _str(row.get("split")),
        "theorem_name": _str(row.get("theorem_name")),
    }


def _phase(row: dict[str, Any]) -> str:
    if row.get("phase"):
        return _str(row.get("phase"))
    args = row.get("args")
    if isinstance(args, dict):
        return _str(args.get("phase"))
    return ""


def _build_result_maps(exp_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_record: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    results_path = exp_dir / "results.jsonl"
    if not results_path.exists():
        return by_record, by_id
    for _idx, _raw, row, _err in _jsonl_lines(results_path):
        if not row:
            continue
        if row.get("record_id"):
            by_record[_str(row.get("record_id"))] = row
        if row.get("id"):
            by_id[_str(row.get("id"))] = row
    return by_record, by_id


def _classify_error(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    text_parts = [
        row.get("error"),
        row.get("result"),
        row.get("status"),
        row.get("phase"),
        row.get("kind"),
        row.get("tool_name"),
    ]
    args = row.get("args")
    if isinstance(args, dict):
        text_parts.extend([args.get("message"), args.get("raw_output")])
        text_parts.extend(args.get("errors") or [])
        text_parts.extend(args.get("warnings") or [])
    for node in row.get("failed_nodes") or []:
        if isinstance(node, dict):
            text_parts.extend(
                [
                    node.get("name"),
                    node.get("signal"),
                    node.get("analysis"),
                    node.get("suggested_fix"),
                ]
            )
            text_parts.extend(node.get("lean_errors") or [])
    text = "\n".join(_str(x) for x in text_parts if x)
    low = text.lower()
    if "maximum context length" in low or "context length" in low:
        return "context_overflow"
    if "timeout" in low or "timed out" in low:
        return "timeout_or_infra"
    if "unknown identifier" in low or "not in scope" in low:
        return "missing_context_or_symbol_mismatch"
    if "unknown constant" in low or "unknown declaration" in low:
        return "wrong_mathlib_name_or_missing_lemma"
    if "failed to synthesize instance" in low or "type expected" in low or "type mismatch" in low:
        return "invalid_statement_type_or_type_mismatch"
    if "rewrite" in low or "simp" in low or "linarith" in low or "omega" in low or "unsolved goals" in low:
        return "tactic_or_algebra_stuck"
    if "sorry" in low or "unexpected end of input" in low:
        return "incomplete_generated_proof"
    if "blocked_by_dependency" in low or "dependency" in low:
        return "dependency_cascade"
    if row.get("root_proved") is False or row.get("ok") is False:
        return "unsolved_or_failed_check"
    return ""


def _empty_row(exp_dir: Path, generated_at: float) -> dict[str, Any]:
    return {
        "exp_dir": str(exp_dir),
        "exp_name": exp_dir.name,
        "generated_at_unix": generated_at,
        "artifact_type": "",
        "rel_path": "",
        "file_path": "",
        "file_index": None,
        "row_index": None,
        "trace_event_index": None,
        "id": "",
        "record_id": "",
        "source_id": "",
        "subset": "",
        "split": "",
        "theorem_name": "",
        "thm_name": "",
        "phase": "",
        "iteration": None,
        "turn": None,
        "kind": "",
        "tool_name": "",
        "call_id": "",
        "ok": "",
        "status": "",
        "root_proved": "",
        "error_category_hint": "",
        "raw_text": "",
        "raw_json": "",
        "args_json": "",
        "result_text": "",
        "parse_error": "",
    }


def _row_from_jsonl(
    exp_dir: Path,
    generated_at: float,
    path: Path,
    file_index: int,
    row_index: int,
    raw: str,
    parsed: dict[str, Any] | None,
    parse_error: str,
    *,
    artifact_type: str,
    result_by_record: dict[str, dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out = _empty_row(exp_dir, generated_at)
    rel_path = str(path.relative_to(exp_dir))
    out.update(
        {
            "artifact_type": artifact_type,
            "rel_path": rel_path,
            "file_path": str(path),
            "file_index": file_index,
            "row_index": row_index,
            "raw_text": raw,
            "raw_json": raw if parsed is not None else "",
            "parse_error": parse_error,
        }
    )
    if parsed is None:
        return out

    keys = _record_keys(parsed)
    if artifact_type == "trace_event":
        path_keys = _trace_ids_from_path(exp_dir, path)
        keys.update({k: v for k, v in path_keys.items() if not keys.get(k) and v})
        result_row = result_by_record.get(keys["record_id"]) or result_by_id.get(keys["id"]) or {}
        result_keys = _record_keys(result_row)
        keys.update({k: v for k, v in result_keys.items() if not keys.get(k) and v})
        out["trace_event_index"] = row_index

    out.update(keys)
    out.update(
        {
            "thm_name": _str(parsed.get("thm_name")),
            "phase": _phase(parsed),
            "iteration": _int_or_none(parsed.get("iteration")),
            "turn": _int_or_none(parsed.get("turn")),
            "kind": _str(parsed.get("kind")),
            "tool_name": _str(parsed.get("tool_name")),
            "call_id": _str(parsed.get("call_id")),
            "ok": _str(parsed.get("ok")),
            "status": _str(parsed.get("status")),
            "root_proved": _str(parsed.get("root_proved")),
            "error_category_hint": _classify_error(parsed),
            "args_json": _json_dumps(parsed.get("args")),
            "result_text": _str(parsed.get("result")),
        }
    )
    if not out["theorem_name"] and out["thm_name"]:
        out["theorem_name"] = out["thm_name"]
    return out


def _row_from_file(
    exp_dir: Path,
    generated_at: float,
    path: Path,
    file_index: int,
    *,
    artifact_type: str,
    result_by_record: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out = _empty_row(exp_dir, generated_at)
    raw_text, parse_error = _read_text(path)
    rel_path = path.relative_to(exp_dir)
    out.update(
        {
            "artifact_type": artifact_type,
            "rel_path": str(rel_path),
            "file_path": str(path),
            "file_index": file_index,
            "row_index": 1,
            "raw_text": raw_text,
            "parse_error": parse_error,
        }
    )
    if path.suffix == ".json" and raw_text:
        try:
            parsed = json.loads(raw_text)
            out["raw_json"] = _json_dumps(parsed)
            out.update(_record_keys(parsed if isinstance(parsed, dict) else None))
        except json.JSONDecodeError as exc:
            out["parse_error"] = str(exc)

    if artifact_type in {"checkpoint", "blueprint"}:
        record_id = path.stem
        if artifact_type == "blueprint" and len(path.parts) >= 2:
            record_id = path.parent.name
        result_row = result_by_record.get(record_id) or {}
        keys = _record_keys(result_row)
        keys["record_id"] = keys["record_id"] or record_id
        out.update(keys)
    return out


def _artifact_type(exp_dir: Path, path: Path) -> str:
    rel = path.relative_to(exp_dir)
    parts = rel.parts
    if rel.name == "results.jsonl":
        return "result"
    if rel.name == "rounds.jsonl":
        return "round"
    if rel.name == "metrics.json":
        return "metrics"
    if rel.name == "lean_runtime.json":
        return "lean_runtime"
    if parts and parts[0] == "traces" and path.suffix == ".jsonl":
        return "trace_event"
    if parts and parts[0] == "checkpoints":
        return "checkpoint"
    if parts and parts[0] == "blueprints":
        return "blueprint"
    return "file"


def aggregate_experiment(exp_dir: Path) -> list[dict[str, Any]]:
    generated_at = time.time()
    result_by_record, result_by_id = _build_result_maps(exp_dir)
    rows: list[dict[str, Any]] = []
    files = sorted(
        path
        for path in exp_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    )
    for file_index, path in enumerate(files, 1):
        artifact_type = _artifact_type(exp_dir, path)
        if path.suffix == ".jsonl":
            for row_index, raw, parsed, parse_error in _jsonl_lines(path):
                rows.append(
                    _row_from_jsonl(
                        exp_dir,
                        generated_at,
                        path,
                        file_index,
                        row_index,
                        raw,
                        parsed,
                        parse_error,
                        artifact_type=artifact_type,
                        result_by_record=result_by_record,
                        result_by_id=result_by_id,
                    )
                )
        else:
            rows.append(
                _row_from_file(
                    exp_dir,
                    generated_at,
                    path,
                    file_index,
                    artifact_type=artifact_type,
                    result_by_record=result_by_record,
                )
            )
    return rows


def write_parquet(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [row.get(name) for row in rows] for name in SCHEMA.names}
    table = pa.Table.from_pydict(columns, schema=SCHEMA)
    pq.write_table(table, output, compression="zstd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate a RobustPA refine experiment into one trace-complete parquet file.",
    )
    parser.add_argument("exp_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")
    if not exp_dir.is_dir():
        raise NotADirectoryError(f"exp_dir is not a directory: {exp_dir}")

    output = exp_dir / "chat_aggregate.parquet"
    rows = aggregate_experiment(exp_dir)
    write_parquet(rows, output)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["artifact_type"]] = counts.get(row["artifact_type"], 0) + 1
    print(f"[wrote] {output}")
    print(f"[rows] {len(rows)}")
    print(f"[artifact_counts] {json.dumps(counts, ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
