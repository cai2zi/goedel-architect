from __future__ import annotations

import argparse
import json
import re
import socketserver
import sys
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(text: Any, limit: int | None = None) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if limit is not None and len(value) > limit:
        return value[:limit] + "..."
    return value


def _trim_payload(value: Any, max_text: int, depth: int = 0) -> Any:
    if depth > 4:
        return _norm(value, max_text)
    if isinstance(value, str):
        return value if len(value) <= max_text else value[:max_text] + "..."
    if isinstance(value, list):
        trimmed = [_trim_payload(item, max_text, depth + 1) for item in value[:20]]
        if len(value) > 20:
            trimmed.append(f"... (+{len(value) - 20} more)")
        return trimmed
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 30:
                out["..."] = f"+{len(value) - 30} more keys"
                break
            out[str(key)] = _trim_payload(item, max_text, depth + 1)
        return out
    return value


def _safe_record_id(value: str) -> str:
    return value.strip().removeprefix("robustpa_")


@dataclass
class ProblemRef:
    row: dict[str, Any]
    trace_path: Path | None
    checkpoint_path: Path | None


class ExperimentIndex:
    def __init__(self, exp_dir: Path) -> None:
        self.exp_dir = exp_dir.resolve()
        self.results = _jsonl(self.exp_dir / "results.jsonl")
        self.rounds = _jsonl(self.exp_dir / "rounds.jsonl")
        self.results_by_id: dict[str, dict[str, Any]] = {}
        self.trace_by_record_id: dict[str, Path] = {}
        self.rounds_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._build()

    def _build(self) -> None:
        for row in self.results:
            keys = {
                str(row.get("id") or ""),
                str(row.get("record_id") or ""),
                str(row.get("source_id") or ""),
                str(row.get("theorem_name") or ""),
            }
            record_id = str(row.get("record_id") or "")
            if record_id.startswith("robustpa_"):
                keys.add(record_id.removeprefix("robustpa_"))
            for key in keys:
                if key:
                    self.results_by_id[key] = row
        traces_dir = self.exp_dir / "traces"
        if traces_dir.exists():
            for path in sorted(traces_dir.rglob("*.jsonl")):
                self.trace_by_record_id[path.stem] = path
                if path.stem.startswith("robustpa_"):
                    self.trace_by_record_id[path.stem.removeprefix("robustpa_")] = path
        for row in self.rounds:
            row_id = str(row.get("id") or "")
            if row_id:
                self.rounds_by_id[row_id].append(row)

    def list_problems(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self.results:
            out.append({
                "id": row.get("id"),
                "record_id": row.get("record_id"),
                "source_id": row.get("source_id"),
                "status": row.get("status"),
                "success": bool(row.get("root_proved")),
                "iterations": row.get("iterations"),
                "total_nodes": row.get("total_nodes"),
                "proved_node_count": row.get("proved_node_count"),
                "failed_count": len(row.get("failed_nodes") or []),
            })
        return sorted(out, key=lambda item: str(item.get("source_id") or ""))

    def resolve(self, query: str) -> ProblemRef | None:
        query = query.strip()
        if not query:
            return None
        row = self.results_by_id.get(query)
        if row is None:
            row = self.results_by_id.get(f"robustpa_{_safe_record_id(query)}")
        if row is None:
            lowered = query.lower()
            for candidate in self.results:
                haystack = " ".join(
                    str(candidate.get(key) or "")
                    for key in ("id", "record_id", "source_id", "theorem_name")
                ).lower()
                if lowered in haystack:
                    row = candidate
                    break
        if row is None:
            return None
        record_id = str(row.get("record_id") or "")
        trace_path = self.trace_by_record_id.get(record_id) or self.trace_by_record_id.get(record_id.removeprefix("robustpa_"))
        checkpoint = row.get("checkpoint_path")
        checkpoint_path = Path(checkpoint) if checkpoint else None
        if checkpoint_path and not checkpoint_path.exists():
            checkpoint_path = self.exp_dir / "checkpoints" / str(row.get("subset") or "") / str(row.get("split") or "") / f"{record_id}.json"
        return ProblemRef(row=row, trace_path=trace_path, checkpoint_path=checkpoint_path)


def _latest_rounds(index: ExperimentIndex, unique_id: str) -> list[dict[str, Any]]:
    rows = list(index.rounds_by_id.get(unique_id) or [])
    return sorted(rows, key=lambda row: (int(row.get("iteration") or 0), str(row.get("phase") or "")))


def _event_summary(event: dict[str, Any], max_text: int) -> dict[str, Any]:
    kind = str(event.get("kind") or "")
    tool = str(event.get("tool_name") or "")
    args = event.get("args")
    result = event.get("result")
    out = {
        "kind": kind,
        "turn": event.get("turn"),
        "thm_name": event.get("thm_name"),
        "tool_name": tool,
        "phase": event.get("phase"),
        "iteration": event.get("iteration"),
        "ok": event.get("ok"),
        "ts": event.get("ts"),
        "args": _trim_payload(args, max_text),
        "result": _trim_payload(result, max_text),
        "short": "",
    }
    if kind == "theorem_start":
        out["short"] = _norm((args or {}).get("thm_stmt"), max_text)
    elif kind == "model_text":
        out["short"] = _norm(result, max_text)
    elif kind == "llm_response":
        llm_args = args or {}
        prefix = (
            f"{llm_args.get('operation') or 'llm_response'} "
            f"finish={llm_args.get('finish_reason')} "
            f"tool_calls={llm_args.get('tool_call_count')}"
        )
        body = llm_args.get("reconstructed_tool_calls_text") or result
        out["short"] = f"{prefix}; {_norm(body, max_text)}"
    elif kind == "tool_call":
        out["short"] = f"{tool} {_norm(args, max_text)}"
    elif kind == "tool_result":
        if tool == "lean_compile":
            errors = (args or {}).get("errors") or []
            out["short"] = "ok" if event.get("ok") else _norm(errors[0] if errors else result, max_text)
        else:
            out["short"] = _norm(result, max_text)
    elif kind == "llm_usage":
        out["short"] = json.dumps(args or {}, ensure_ascii=False, sort_keys=True)
    elif kind == "node_finished":
        out["short"] = f"wall={float((args or {}).get('wall_time_s') or 0):.1f}s; signal={(args or {}).get('signal')}"
    elif kind == "final_verify":
        errors = (args or {}).get("lean_errors") or []
        out["short"] = "root closure ok" if event.get("ok") else _norm(errors, max_text)
    elif kind == "lean_check_result":
        errors = (args or {}).get("errors") or []
        out["short"] = "ok" if event.get("ok") else _norm(errors[0] if errors else "", max_text)
    elif kind == "llm_error":
        out["short"] = _norm((args or {}).get("message") or result, max_text)
    return out


def _node_question_map(events: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for event in events:
        if event.get("kind") == "theorem_start":
            thm_name = str(event.get("thm_name") or "")
            stmt = str((event.get("args") or {}).get("thm_stmt") or "")
            if thm_name and stmt:
                out[thm_name] = stmt
    return out


def build_trace_payload(
    index: ExperimentIndex,
    query: str,
    *,
    node_filter: str = "",
    max_text: int = 500,
    include_tool_results: bool = True,
) -> tuple[int, dict[str, Any]]:
    ref = index.resolve(query)
    if ref is None:
        return HTTPStatus.NOT_FOUND, {"error": f"id not found: {query}"}
    row = ref.row
    unique_id = str(row.get("id") or "")
    events = _jsonl(ref.trace_path) if ref.trace_path else []
    events.sort(key=lambda event: float(event.get("ts") or 0.0))

    node_filter = node_filter.strip()
    if node_filter:
        events = [
            event for event in events
            if node_filter.lower() in str(event.get("thm_name") or "").lower()
        ]
    if not include_tool_results:
        events = [event for event in events if event.get("kind") != "tool_result"]

    checkpoint = _read_json(ref.checkpoint_path) if ref.checkpoint_path and ref.checkpoint_path.exists() else None
    question_by_node = _node_question_map(events)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    phase_events: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("kind")
        thm_name = str(event.get("thm_name") or "")
        turn = int(event.get("turn") or 0)
        if kind in {"theorem_start", "llm_response", "tool_call", "tool_result", "model_text", "node_finished", "llm_usage", "llm_error"} and thm_name != unique_id:
            grouped[(thm_name, turn)].append(event)
        else:
            phase_events.append(event)

    turns: list[dict[str, Any]] = []
    for (node, turn), group in sorted(
        grouped.items(),
        key=lambda item: (min(float(event.get("ts") or 0) for event in item[1]), item[0][0], item[0][1]),
    ):
        kinds = Counter(str(event.get("kind") or "") for event in group)
        tools = Counter(str(event.get("tool_name") or "") for event in group if event.get("kind") == "tool_call")
        answer = next((event.get("result") for event in group if event.get("kind") == "model_text" and event.get("result")), "")
        turns.append({
            "node": node,
            "turn": turn,
            "question": question_by_node.get(node, ""),
            "answer": answer,
            "kind_counts": dict(kinds),
            "tool_counts": dict(tools),
            "events": [_event_summary(event, max_text) for event in group],
        })

    rounds = _latest_rounds(index, unique_id)
    latest_nodes = (rounds[-1].get("nodes") if rounds else []) or []
    trace_stats = {
        "events": len(events),
        "kind_counts": dict(Counter(str(event.get("kind") or "") for event in events)),
        "tool_counts": dict(Counter(str(event.get("tool_name") or "") for event in events if event.get("kind") == "tool_call")),
        "nodes_with_turns": len({turn["node"] for turn in turns}),
        "turn_groups": len(turns),
    }
    return HTTPStatus.OK, {
        "problem": {
            "id": row.get("id"),
            "record_id": row.get("record_id"),
            "source_id": row.get("source_id"),
            "theorem_name": row.get("theorem_name"),
            "status": row.get("status"),
            "phase": row.get("phase"),
            "success": bool(row.get("root_proved")),
            "iterations": row.get("iterations"),
            "total_nodes": row.get("total_nodes"),
            "proved_node_count": row.get("proved_node_count"),
            "failed_nodes": row.get("failed_nodes") or [],
            "error": row.get("error") or "",
            "trace_path": str(ref.trace_path) if ref.trace_path else "",
            "checkpoint_path": str(ref.checkpoint_path) if ref.checkpoint_path else "",
        },
        "checkpoint": {
            "informal_statement": (checkpoint or {}).get("informal_statement", ""),
            "status": (checkpoint or {}).get("status"),
            "root_proved": (checkpoint or {}).get("status") == "solved",
            "iteration": (checkpoint or {}).get("iteration"),
            "proved_cache_count": len((checkpoint or {}).get("proved_cache") or {}),
            "refinement_history_count": len((checkpoint or {}).get("refinement_history") or []),
        },
        "rounds": [
            {
                "iteration": round_row.get("iteration"),
                "phase": round_row.get("phase"),
                "blueprint_path": round_row.get("blueprint_path"),
                "node_counts": dict(Counter(str(node.get("signal") or "pending") for node in round_row.get("nodes") or [])),
            }
            for round_row in rounds
        ],
        "latest_nodes": [
            {
                "name": node.get("name"),
                "kind": node.get("kind"),
                "signal": node.get("signal"),
                "dependencies": node.get("dependencies"),
                "errors": [_norm(error, max_text) for error in (node.get("lean_errors") or [])[:3]],
            }
            for node in latest_nodes
        ],
        "trace_stats": trace_stats,
        "phase_events": [_event_summary(event, max_text) for event in phase_events],
        "turns": turns,
    }


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RobustPA Trace Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #222522;
      --muted: #656a63;
      --line: #d9ddd4;
      --panel: #ffffff;
      --accent: #176b6b;
      --bad: #a13b2c;
      --good: #267047;
      --warn: #9a6b16;
      --code: #101411;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 247, 244, 0.96);
      backdrop-filter: blur(8px);
    }
    .bar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(180px, 0.55fr) auto auto auto;
      gap: 8px;
      align-items: center;
      max-width: 1500px;
      margin: 0 auto;
      padding: 12px 16px;
    }
    input, select, button {
      height: 38px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
      min-width: 0;
    }
    button {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      cursor: pointer;
      font-weight: 650;
    }
    button.secondary {
      background: #fff;
      color: var(--accent);
    }
    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 16px;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 16px;
    }
    aside, section {
      min-width: 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel h2, .panel h3 {
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    .body { padding: 12px; }
    .summary {
      display: grid;
      grid-template-columns: repeat(6, minmax(90px, 1fr));
      gap: 8px;
      margin-bottom: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 9px 10px;
      min-height: 58px;
    }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 17px; font-weight: 700; margin-top: 3px; overflow-wrap: anywhere; }
    .ok { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .muted { color: var(--muted); }
    .list {
      max-height: calc(100vh - 106px);
      overflow: auto;
    }
    .problem-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
    }
    .problem-row:hover { background: #f1f4ef; }
    .problem-row strong { display: block; overflow-wrap: anywhere; }
    .tag {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 22px;
      padding: 0 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 12px;
      background: #fff;
      white-space: nowrap;
    }
    .tag.good { border-color: #b6d6c1; background: #eef8f1; }
    .tag.bad { border-color: #e2b8ae; background: #fff0ed; }
    .tag.warn { border-color: #dfcc9c; background: #fff8e8; }
    .grid2 {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      margin-bottom: 16px;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--code);
    }
    details {
      border-top: 1px solid var(--line);
      background: #fff;
    }
    details:first-child { border-top: 0; }
    summary {
      cursor: pointer;
      padding: 9px 12px;
      list-style: none;
      display: grid;
      grid-template-columns: minmax(170px, 0.4fr) minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
    }
    summary::-webkit-details-marker { display: none; }
    details[open] summary { border-bottom: 1px solid var(--line); background: #fbfcfa; }
    .turn-body {
      padding: 12px;
      display: grid;
      grid-template-columns: minmax(0, 0.8fr) minmax(0, 1fr);
      gap: 12px;
    }
    .event {
      border: 1px solid var(--line);
      border-radius: 7px;
      margin-bottom: 8px;
      overflow: hidden;
    }
    .event-head {
      padding: 7px 9px;
      background: #f6f8f4;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .event-content { padding: 9px; }
    .nodes {
      display: grid;
      gap: 6px;
      max-height: 320px;
      overflow: auto;
    }
    .node-row {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px;
      background: #fff;
    }
    .node-row button {
      height: 26px;
      padding: 0 8px;
      font-size: 12px;
      float: right;
      margin-left: 8px;
    }
    .small { font-size: 12px; }
    .hidden { display: none !important; }
    @media (max-width: 1000px) {
      .bar { grid-template-columns: 1fr; }
      main { grid-template-columns: 1fr; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid2, .turn-body { grid-template-columns: 1fr; }
      .list { max-height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <input id="idInput" placeholder="source_id / record_id / unique_id" autocomplete="off" />
      <input id="nodeInput" placeholder="node filter" autocomplete="off" />
      <select id="resultMode">
        <option value="1">tool results on</option>
        <option value="0">tool results off</option>
      </select>
      <button id="loadBtn">Load</button>
      <button id="clearBtn" class="secondary">Clear</button>
    </div>
  </header>
  <main>
    <aside class="panel">
      <h2>Problems</h2>
      <div class="body">
        <input id="listFilter" placeholder="filter list" />
      </div>
      <div id="problemList" class="list"></div>
    </aside>
    <section>
      <div id="status" class="muted">Ready.</div>
      <div id="content"></div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let problems = [];

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function tag(text, cls = '') {
      return `<span class="tag ${cls}">${esc(text)}</span>`;
    }

    function jsonBlock(value) {
      if (value === null || value === undefined || value === '') return '<pre></pre>';
      if (typeof value === 'string') return `<pre>${esc(value)}</pre>`;
      return `<pre>${esc(JSON.stringify(value, null, 2))}</pre>`;
    }

    async function api(path) {
      const res = await fetch(path);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }

    function renderProblems() {
      const q = $('listFilter').value.trim().toLowerCase();
      const root = $('problemList');
      const rows = problems.filter((item) => {
        if (!q) return true;
        return [item.source_id, item.record_id, item.id, item.status].join(' ').toLowerCase().includes(q);
      });
      root.innerHTML = rows.map((item) => {
        const cls = item.success ? 'good' : (item.status === 'error' ? 'warn' : 'bad');
        return `
          <div class="problem-row" data-id="${esc(item.source_id)}">
            <div>
              <strong>${esc(item.source_id)}</strong>
              <span class="small muted">${esc(item.record_id)}</span>
            </div>
            <div>${tag(item.status, cls)} ${tag('i' + item.iterations)}</div>
          </div>`;
      }).join('');
      root.querySelectorAll('.problem-row').forEach((el) => {
        el.addEventListener('click', () => {
          $('idInput').value = el.dataset.id;
          loadTrace();
        });
      });
    }

    function metric(label, value, cls = '') {
      return `<div class="metric"><div class="label">${esc(label)}</div><div class="value ${cls}">${esc(value)}</div></div>`;
    }

    function renderNodeRows(nodes) {
      return nodes.map((node) => {
        const cls = node.signal === 'solved' ? 'good' : (node.signal === 'pending' ? 'warn' : 'bad');
        return `<div class="node-row">
          <button data-node="${esc(node.name)}">View</button>
          <strong>${esc(node.name)}</strong> ${tag(node.signal, cls)} ${tag(node.kind || '')}
          <div class="small muted">${esc((node.dependencies || []).join(', '))}</div>
          ${(node.errors || []).map((err) => `<pre>${esc(err)}</pre>`).join('')}
        </div>`;
      }).join('');
    }

    function renderEvent(event) {
      const ok = event.ok === true ? tag('ok', 'good') : event.ok === false ? tag('fail', 'bad') : '';
      const tool = event.tool_name ? tag(event.tool_name) : '';
      const phase = event.phase ? tag(event.phase + ':' + (event.iteration ?? '')) : '';
      return `<div class="event">
        <div class="event-head">${tag(event.kind)} ${phase} ${tool} ${ok} ${tag('turn ' + (event.turn ?? 0))}</div>
        <div class="event-content">
          ${event.short ? `<pre>${esc(event.short)}</pre>` : ''}
          <details>
            <summary><span>args/result</span><span class="muted">${esc(event.thm_name || '')}</span><span></span></summary>
            <div class="turn-body">
              <div>${jsonBlock(event.args)}</div>
              <div>${jsonBlock(event.result)}</div>
            </div>
          </details>
        </div>
      </div>`;
    }

    function renderTurn(turn, index) {
      const tools = Object.entries(turn.tool_counts || {}).map(([name, count]) => `${name}:${count}`).join(' ');
      const kindText = Object.entries(turn.kind_counts || {}).map(([name, count]) => `${name}:${count}`).join(' ');
      return `<details class="turn" ${index < 3 ? 'open' : ''}>
        <summary>
          <span><strong>${esc(turn.node)}</strong> <span class="muted">turn ${esc(turn.turn)}</span></span>
          <span class="small muted">${esc(kindText)}</span>
          <span>${tools ? tag(tools) : ''}</span>
        </summary>
        <div class="turn-body">
          <div>
            <h3>Question</h3>
            ${jsonBlock(turn.question || '')}
            <h3>Answer</h3>
            ${jsonBlock(turn.answer || '')}
          </div>
          <div>
            <h3>Events</h3>
            ${(turn.events || []).map(renderEvent).join('')}
          </div>
        </div>
      </details>`;
    }

    function renderTrace(data) {
      const p = data.problem;
      const statusCls = p.success ? 'ok' : (p.status === 'error' ? 'warn' : 'bad');
      const rounds = data.rounds || [];
      const stats = data.trace_stats || {};
      $('content').innerHTML = `
        <div class="summary">
          ${metric('source_id', p.source_id)}
          ${metric('status', p.status, statusCls)}
          ${metric('iter', p.iterations)}
          ${metric('nodes', `${p.proved_node_count}/${p.total_nodes}`)}
          ${metric('trace events', stats.events || 0)}
          ${metric('turn groups', stats.turn_groups || 0)}
        </div>
        <div class="grid2">
          <div class="panel">
            <h2>Problem</h2>
            <div class="body">
              <div>${tag(p.record_id)} ${tag(p.theorem_name || '')}</div>
              ${p.error ? `<pre class="bad">${esc(p.error)}</pre>` : ''}
              <details open>
                <summary><span>informal statement</span><span></span><span></span></summary>
                <div class="body">${jsonBlock(data.checkpoint.informal_statement || '')}</div>
              </details>
            </div>
          </div>
          <div class="panel">
            <h2>Rounds</h2>
            <div class="body">
              ${(rounds || []).map((r) => `<div class="small">
                ${tag('i' + r.iteration)} ${tag(r.phase)} ${esc(JSON.stringify(r.node_counts))}
              </div>`).join('')}
              <pre>${esc(JSON.stringify(stats, null, 2))}</pre>
            </div>
          </div>
        </div>
        <div class="grid2">
          <div class="panel">
            <h2>Latest Nodes</h2>
            <div class="body nodes">${renderNodeRows(data.latest_nodes || [])}</div>
          </div>
          <div class="panel">
            <h2>Phase Events</h2>
            <div class="body">${(data.phase_events || []).map(renderEvent).join('')}</div>
          </div>
        </div>
        <div class="panel">
          <h2>Dialogue Turns</h2>
          <div>${(data.turns || []).map(renderTurn).join('')}</div>
        </div>`;
      document.querySelectorAll('.node-row button').forEach((button) => {
        button.addEventListener('click', () => {
          $('nodeInput').value = button.dataset.node;
          loadTrace();
        });
      });
    }

    async function loadProblems() {
      problems = await api('/api/problems');
      renderProblems();
    }

    async function loadTrace() {
      const id = $('idInput').value.trim();
      if (!id) return;
      $('status').textContent = 'Loading...';
      try {
        const params = new URLSearchParams({
          id,
          node: $('nodeInput').value.trim(),
          include_results: $('resultMode').value,
          max_text: '900'
        });
        const data = await api('/api/trace?' + params.toString());
        renderTrace(data);
        $('status').textContent = `${data.problem.source_id} loaded.`;
      } catch (err) {
        $('status').textContent = err.message;
        $('content').innerHTML = '';
      }
    }

    $('loadBtn').addEventListener('click', loadTrace);
    $('clearBtn').addEventListener('click', () => {
      $('nodeInput').value = '';
      loadTrace();
    });
    $('idInput').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') loadTrace(); });
    $('nodeInput').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') loadTrace(); });
    $('listFilter').addEventListener('input', renderProblems);
    $('resultMode').addEventListener('change', loadTrace);
    loadProblems().catch((err) => $('status').textContent = err.message);
  </script>
</body>
</html>
"""


class TraceViewerHandler(BaseHTTPRequestHandler):
    index: ExperimentIndex

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[trace-viewer] " + fmt % args + "\n")

    def _send_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self) -> None:
        data = HTML_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)
        if path == "/":
            self._send_html()
            return
        if path == "/api/problems":
            self._send_json(HTTPStatus.OK, self.index.list_problems())
            return
        if path == "/api/trace":
            query = params.get("id", [""])[0]
            node = params.get("node", [""])[0]
            include_results = params.get("include_results", ["1"])[0] != "0"
            try:
                max_text = int(params.get("max_text", ["500"])[0])
            except ValueError:
                max_text = 500
            status, payload = build_trace_payload(
                self.index,
                query,
                node_filter=node,
                max_text=max_text,
                include_tool_results=include_results,
            )
            self._send_json(status, payload)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"not found: {path}"})


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local RobustPA trace viewer.")
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    if not (exp_dir / "results.jsonl").exists():
        raise FileNotFoundError(f"results.jsonl not found under {exp_dir}")
    index = ExperimentIndex(exp_dir)
    TraceViewerHandler.index = index
    with ReusableTCPServer((args.host, args.port), TraceViewerHandler) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(f"[trace-viewer] exp_dir={exp_dir}")
        print(f"[trace-viewer] open {url}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[trace-viewer] stopped")


if __name__ == "__main__":
    main()
