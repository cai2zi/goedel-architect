from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TOKEN_BINS: list[tuple[int, int | None]] = []
lo = 1
hi = 2048
for _ in range(20):
    TOKEN_BINS.append((lo, hi))
    lo, hi = hi, hi * 2

TIME_BINS: list[tuple[str, float, float | None]] = [
    ("0-1min", 0, 60),
    ("1-10min", 60, 10 * 60),
    ("10-30min", 10 * 60, 30 * 60),
    ("30min-1h", 30 * 60, 60 * 60),
    ("1-2h", 60 * 60, 2 * 60 * 60),
    ("2-4h", 2 * 60 * 60, 4 * 60 * 60),
    ("4h+", 4 * 60 * 60, None),
]

PROBLEM_TIME_BINS: list[tuple[str, float, float | None]] = [
    ("0-1min", 0, 60),
    ("1-10min", 60, 10 * 60),
    ("10-30min", 10 * 60, 30 * 60),
    ("30min-1h", 30 * 60, 60 * 60),
    ("1-2h", 60 * 60, 2 * 60 * 60),
    ("2-4h", 2 * 60 * 60, 4 * 60 * 60),
    ("4-8h", 4 * 60 * 60, 8 * 60 * 60),
    ("8h+", 8 * 60 * 60, None),
]

PROOF_OPERATIONS = {"prove_node_initial", "prove_node_next"}


@dataclass(frozen=True)
class NodeAttempt:
    record_id: str
    iteration: int
    node: str
    proof_turn: int
    ok: bool
    wall_time_s: float
    total_tokens: int


@dataclass(frozen=True)
class ProblemAttempt:
    record_id: str
    iteration: int
    ok: bool
    wall_time_s: float


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _event_key(record_id: str, event: dict[str, Any]) -> tuple[str, int, str] | None:
    if event.get("phase") != "phase2":
        return None
    iteration = event.get("iteration")
    node = event.get("thm_name")
    if not isinstance(iteration, int) or not node:
        return None
    return record_id, iteration, str(node)


def _usage_tokens(event: dict[str, Any]) -> int:
    args = event.get("args") or {}
    total = args.get("total_tokens")
    if isinstance(total, int):
        return total
    return int(args.get("prompt_tokens") or 0) + int(args.get("completion_tokens") or 0)


def collect_node_attempts(
    exp_dir: Path,
) -> tuple[list[NodeAttempt], list[tuple[str, int, str]], dict[str, float]]:
    traces_dir = exp_dir / "traces"
    if not traces_dir.exists():
        raise FileNotFoundError(f"missing traces directory: {traces_dir}")

    attempts: list[NodeAttempt] = []
    incomplete: list[tuple[str, int, str]] = []
    trace_durations: dict[str, float] = {}

    for path in sorted(traces_dir.rglob("*.jsonl")):
        record_id = path.stem
        tokens_by_key: collections.Counter[tuple[str, int, str]] = collections.Counter()
        proof_turn_by_key: collections.Counter[tuple[str, int, str]] = collections.Counter()
        starts: set[tuple[str, int, str]] = set()
        finals: dict[tuple[str, int, str], dict[str, Any]] = {}
        timestamps: list[float] = []

        for event in _jsonl(path):
            ts = event.get("ts")
            if isinstance(ts, (int, float)):
                timestamps.append(float(ts))
            key = _event_key(record_id, event)
            if key is None:
                continue
            kind = event.get("kind")
            if kind == "theorem_start":
                starts.add(key)
            elif kind == "llm_usage":
                tokens_by_key[key] += _usage_tokens(event)
            elif kind == "llm_response":
                args = event.get("args") or {}
                if args.get("operation") in PROOF_OPERATIONS:
                    turn = event.get("turn")
                    if isinstance(turn, int):
                        proof_turn_by_key[key] = max(proof_turn_by_key[key], turn)
            elif kind == "final_verify":
                finals[key] = event

        if timestamps:
            trace_durations[record_id] = max(timestamps) - min(timestamps)

        for key in sorted(starts):
            final = finals.get(key)
            if final is None:
                incomplete.append(key)
                continue
            args = final.get("args") or {}
            attempts.append(
                NodeAttempt(
                    record_id=key[0],
                    iteration=key[1],
                    node=key[2],
                    proof_turn=int(proof_turn_by_key.get(key, 0)),
                    ok=bool(final.get("ok")),
                    wall_time_s=float(args.get("wall_time_s") or 0.0),
                    total_tokens=int(tokens_by_key.get(key, 0)),
                )
            )

    return attempts, incomplete, trace_durations


def collect_problem_attempts(
    exp_dir: Path,
    trace_durations: dict[str, float],
) -> tuple[list[ProblemAttempt], list[str], list[str]]:
    results_path = exp_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"missing results file: {results_path}")

    problems: list[ProblemAttempt] = []
    missing_duration: list[str] = []
    result_record_ids: set[str] = set()

    for row in _jsonl(results_path):
        record_id = str(row.get("record_id") or "")
        if not record_id:
            continue
        result_record_ids.add(record_id)
        try:
            iteration = int(row.get("iterations") or 0)
        except (TypeError, ValueError):
            iteration = 0
        duration = trace_durations.get(record_id)
        if duration is None:
            trace_path = row.get("trace_path")
            if trace_path:
                duration = _trace_duration(Path(str(trace_path)))
        if duration is None:
            duration = 0.0
            missing_duration.append(record_id)
        problems.append(
            ProblemAttempt(
                record_id=record_id,
                iteration=iteration,
                ok=bool(row.get("root_proved")),
                wall_time_s=float(duration),
            )
        )

    trace_only = sorted(set(trace_durations) - result_record_ids)
    return problems, missing_duration, trace_only


def _trace_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    timestamps: list[float] = []
    for event in _jsonl(path):
        ts = event.get("ts")
        if isinstance(ts, (int, float)):
            timestamps.append(float(ts))
    if not timestamps:
        return None
    return max(timestamps) - min(timestamps)


def _iterations(attempts: list[NodeAttempt]) -> list[int]:
    return sorted({attempt.iteration for attempt in attempts})


def _problem_iterations(problems: list[ProblemAttempt]) -> list[int]:
    return sorted({problem.iteration for problem in problems})


def _proof_turns(attempts: list[NodeAttempt]) -> list[int]:
    return sorted({attempt.proof_turn for attempt in attempts})


def _token_label(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo}+"
    return f"{lo}-{hi}"


def _token_bin(tokens: int) -> str:
    if tokens <= 0:
        return "0"
    for lo, hi in TOKEN_BINS:
        if tokens >= lo and (hi is None or tokens < hi):
            return _token_label(lo, hi)
    upper = 2 ** math.ceil(math.log2(tokens + 1))
    return f"{upper // 2}-{upper}"


def _time_bin(seconds: float, bins: list[tuple[str, float, float | None]]) -> str:
    for label, lo, hi in bins:
        if seconds >= lo and (hi is None or seconds < hi):
            return label
    return bins[-1][0]


def _markdown_table(title: str, row_name: str, rows: list[tuple[str, dict[int, int]]], columns: list[int]) -> str:
    headers = [row_name] + [str(i) for i in columns]
    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for label, counts in rows:
        values = [label] + [str(counts.get(i, 0)) for i in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def node_compile_table(attempts: list[NodeAttempt], turns: list[int]) -> str:
    success: collections.Counter[int] = collections.Counter()
    failure: collections.Counter[int] = collections.Counter()
    for attempt in attempts:
        if attempt.ok:
            success[attempt.proof_turn] += 1
        else:
            failure[attempt.proof_turn] += 1
    return _markdown_table(
        "Proof Turn vs Lean Compile Result",
        "result",
        [("success", success), ("failure", failure)],
        turns,
    )


def node_token_table(attempts: list[NodeAttempt], turns: list[int]) -> str:
    counts: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    max_tokens = max((attempt.total_tokens for attempt in attempts), default=0)
    labels = ["0"] if any(attempt.total_tokens <= 0 for attempt in attempts) else []
    for lo, hi in TOKEN_BINS:
        labels.append(_token_label(lo, hi))
        if hi is not None and hi > max_tokens:
            break
    for attempt in attempts:
        counts[_token_bin(attempt.total_tokens)][attempt.proof_turn] += 1
    return _markdown_table(
        "Proof Turn vs Total Token Distribution",
        "tokens",
        [(label, counts[label]) for label in labels if counts[label] or label != "0"],
        turns,
    )


def node_time_table(attempts: list[NodeAttempt], turns: list[int]) -> str:
    counts: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    labels = [label for label, _, _ in TIME_BINS]
    for attempt in attempts:
        counts[_time_bin(attempt.wall_time_s, TIME_BINS)][attempt.proof_turn] += 1
    return _markdown_table(
        "Proof Turn vs Node Wall Time Distribution",
        "wall_time",
        [(label, counts[label]) for label in labels],
        turns,
    )


def problem_success_table(problems: list[ProblemAttempt], iterations: list[int]) -> str:
    success: collections.Counter[int] = collections.Counter()
    failure: collections.Counter[int] = collections.Counter()
    for problem in problems:
        if problem.ok:
            success[problem.iteration] += 1
        else:
            failure[problem.iteration] += 1
    return _markdown_table(
        "Problem Iter vs Overall Result",
        "result",
        [("success", success), ("failure", failure)],
        iterations,
    )


def problem_time_table(problems: list[ProblemAttempt], iterations: list[int]) -> str:
    counts: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    labels = [label for label, _, _ in PROBLEM_TIME_BINS]
    for problem in problems:
        counts[_time_bin(problem.wall_time_s, PROBLEM_TIME_BINS)][problem.iteration] += 1
    return _markdown_table(
        "Problem Iter vs Wall Time Distribution",
        "wall_time",
        [(label, counts[label]) for label in labels],
        iterations,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build node-level turn tables and problem-level iteration tables from RobustPA refine traces."
    )
    parser.add_argument("exp_dir", type=Path, help="Experiment output directory.")
    parser.add_argument("--output", type=Path, help="Write Markdown tables to this path.")
    args = parser.parse_args()

    attempts, incomplete, trace_durations = collect_node_attempts(args.exp_dir)
    problems, missing_duration, trace_only = collect_problem_attempts(args.exp_dir, trace_durations)
    proof_turns = _proof_turns(attempts)
    problem_iterations = _problem_iterations(problems)
    parts = [
        node_compile_table(attempts, proof_turns),
        node_token_table(attempts, proof_turns),
        node_time_table(attempts, proof_turns),
        problem_success_table(problems, problem_iterations),
        problem_time_table(problems, problem_iterations),
    ]
    text = "\n\n".join(parts) + "\n"

    if incomplete:
        print(
            f"warning: {len(incomplete)} phase2 node(s) have theorem_start but no final_verify; excluded",
            file=sys.stderr,
        )
        for record_id, iteration, node in incomplete[:10]:
            print(f"  {record_id} iter={iteration} node={node}", file=sys.stderr)
        if len(incomplete) > 10:
            print(f"  ... +{len(incomplete) - 10} more", file=sys.stderr)
    if missing_duration:
        print(
            f"warning: {len(missing_duration)} result row(s) have no trace duration; counted as 0s",
            file=sys.stderr,
        )
    if trace_only:
        print(
            f"warning: {len(trace_only)} trace file(s) have no results row; excluded from problem tables",
            file=sys.stderr,
        )
        for record_id in trace_only[:10]:
            print(f"  {record_id}", file=sys.stderr)
        if len(trace_only) > 10:
            print(f"  ... +{len(trace_only) - 10} more", file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
