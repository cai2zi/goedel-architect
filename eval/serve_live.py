#!/usr/bin/env python3
"""
Live-polling trace viewer — watch the prover work in real time.

Unlike serve_viz.py (which renders the HTML once from a finished trace),
this re-reads the JSONL trace file on every browser poll, so nodes and tool
calls appear as the pipeline actually emits them.

run_putnam.py writes one trace file per problem (trace_<problem>.jsonl) so
that two problems in the same batch never get overlaid into one graph. Point
this viewer at the directory containing those files and it will always
follow whichever one was written to most recently; a path to a single file
still works too, for one-off single-problem runs.

Usage:
    python eval/serve_live.py results/putnam/
    python eval/serve_live.py results/putnam/trace_putnam_1962_a3.jsonl --port 9090
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from graph_viz import load_trace, build_theorem_data

LIVE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Goedel-Architect — Live Trace</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
#header { background: #16213e; padding: 12px 20px; border-bottom: 1px solid #0f3460; display: flex; align-items: center; gap: 20px; flex-shrink: 0; }
#header h1 { font-size: 16px; font-weight: 600; color: #90CAF9; }
#header .stat { font-size: 13px; color: #aaa; }
#header .stat b { color: #e0e0e0; }
#live-dot { width: 9px; height: 9px; border-radius: 50%; background: #4CAF50; display: inline-block; animation: pulse 1.4s infinite; margin-right: 6px; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }
#main { display: flex; flex: 1; overflow: hidden; }
#sidebar { width: 260px; background: #16213e; border-right: 1px solid #0f3460; overflow-y: auto; flex-shrink: 0; }
#sidebar-header { padding: 10px 12px; font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #0f3460; }
.thm-item { padding: 8px 12px; border-bottom: 1px solid #0f3460; cursor: pointer; display: flex; align-items: center; gap: 8px; }
.thm-item:hover { background: #1a2a4a; }
.thm-item.active { background: #0f3460; }
.badge { font-size: 10px; font-weight: bold; padding: 2px 5px; border-radius: 3px; flex-shrink: 0; }
.badge.pass { background: #2E7D32; color: #fff; }
.badge.fail { background: #C62828; color: #fff; }
.badge.running { background: #F57F17; color: #fff; }
.thm-label { font-size: 12px; color: #ccc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.thm-meta { font-size: 10px; color: #666; margin-top: 2px; }
#graph-wrap { flex: 1; position: relative; overflow: hidden; }
#graph-container { width: 100%; height: 100%; }
#empty-msg { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #555; pointer-events: none; }
#detail { width: 360px; background: #16213e; border-left: 1px solid #0f3460; overflow-y: auto; flex-shrink: 0; display: flex; flex-direction: column; }
#detail-header { padding: 12px 14px; border-bottom: 1px solid #0f3460; font-size: 13px; font-weight: 600; color: #90CAF9; }
#detail-body { padding: 14px; flex: 1; font-size: 12px; line-height: 1.6; color: #ccc; }
#detail-body pre { background: #0a0a1a; border: 1px solid #0f3460; border-radius: 4px; padding: 10px; font-size: 11px; overflow-x: auto; white-space: pre-wrap; word-break: break-all; margin-top: 8px; color: #a8d8a8; }
#detail-body b { color: #90CAF9; }
#detail-placeholder { color: #444; font-size: 13px; text-align: center; padding-top: 40px; }
</style>
</head>
<body>
<div id="header">
  <h1><span id="live-dot"></span>Goedel-Architect — Live Trace</h1>
  <span class="stat">watching <b id="stat-problem">—</b></span>
  <span class="stat"><b id="stat-count">0</b> nodes</span>
  <span class="stat"><b id="stat-pass">0</b> solved</span>
  <span class="stat"><b id="stat-running">0</b> in progress</span>
  <span class="stat">updated <b id="stat-updated">—</b></span>
</div>
<div id="main">
  <div id="sidebar">
    <div id="sidebar-header">Blueprint Nodes</div>
    <div id="thm-list"></div>
  </div>
  <div id="graph-wrap">
    <div id="graph-container"></div>
    <div id="empty-msg"><p>Select a node from the left</p></div>
  </div>
  <div id="detail">
    <div id="detail-header">Node Detail</div>
    <div id="detail-body"><div id="detail-placeholder">Click a node in the graph</div></div>
  </div>
</div>

<script>
let DATA = [];
let activeIdx = null;
let network = null;
let currentTraceFile = null;

const NETWORK_OPTIONS = {
  layout: { hierarchical: { enabled: true, direction: 'UD', sortMethod: 'directed', levelSeparation: 110, nodeSpacing: 160, treeSpacing: 200 } },
  physics: { enabled: false },
  edges: { color: { color: '#555', highlight: '#90CAF9' }, smooth: { type: 'cubicBezier', forceDirection: 'vertical', roundness: 0.4 } },
  nodes: { borderWidth: 2, shadow: false },
  interaction: { hover: true, tooltipDelay: 200 },
};

function statusOf(thm) {
  if (thm.finished) return thm.ok ? 'pass' : 'fail';
  return 'running';
}

function renderSidebar() {
  const list = document.getElementById('thm-list');
  list.innerHTML = '';
  DATA.forEach((thm, i) => {
    const st = statusOf(thm);
    const div = document.createElement('div');
    div.className = 'thm-item' + (i === activeIdx ? ' active' : '');
    div.innerHTML =
      `<span class="badge ${st}">${st === 'running' ? '…' : st.toUpperCase()}</span>` +
      `<div><div class="thm-label" title="${thm.thm_name}">${thm.thm_name.split('.').pop()}</div>` +
      `<div class="thm-meta">${thm.tool_calls_used} calls${thm.wall_time_s ? ' · ' + thm.wall_time_s.toFixed(1) + 's' : ''}</div></div>`;
    div.addEventListener('click', () => { activeIdx = i; renderGraph(); renderSidebar(); });
    list.appendChild(div);
  });
  document.getElementById('stat-count').textContent = DATA.length;
  document.getElementById('stat-pass').textContent = DATA.filter(t => t.finished && t.ok).length;
  document.getElementById('stat-running').textContent = DATA.filter(t => !t.finished).length;
  document.getElementById('stat-updated').textContent = new Date().toLocaleTimeString();
}

function renderGraph() {
  if (activeIdx === null || !DATA[activeIdx]) return;
  const thm = DATA[activeIdx];
  document.getElementById('empty-msg').style.display = 'none';

  const nodes = new vis.DataSet(thm.graph.nodes);
  const edges = new vis.DataSet(thm.graph.edges);
  if (network) network.destroy();
  network = new vis.Network(document.getElementById('graph-container'), { nodes, edges }, NETWORK_OPTIONS);
  network.on('click', (params) => {
    if (params.nodes.length > 0) {
      const detail = thm.graph.nodeDetails[params.nodes[0]];
      if (detail) {
        document.getElementById('detail-header').textContent = detail.title;
        document.getElementById('detail-body').innerHTML = detail.content;
      }
    }
  });
}

async function poll() {
  try {
    const res = await fetch('/data.json', { cache: 'no-store' });
    const payload = await res.json();
    const newTraceFile = payload.trace_file;
    const traceChanged = newTraceFile !== currentTraceFile;
    currentTraceFile = newTraceFile;
    document.getElementById('stat-problem').textContent = newTraceFile || '—';

    const activeName = (!traceChanged && activeIdx !== null && DATA[activeIdx]) ? DATA[activeIdx].thm_name : null;
    DATA = payload.nodes;
    activeIdx = null;
    if (activeName) {
      const idx = DATA.findIndex(t => t.thm_name === activeName);
      if (idx >= 0) activeIdx = idx;
    }
    if (activeIdx === null && DATA.length > 0) {
      // Prefer the node matching the problem name (the target theorem).
      const problemName = newTraceFile ? newTraceFile.replace(/^trace_/, '').replace(/\.jsonl$/, '') : null;
      const mainIdx = problemName ? DATA.findIndex(t => t.thm_name === problemName) : -1;
      activeIdx = mainIdx >= 0 ? mainIdx : (DATA.length === 1 ? 0 : null);
    }
    renderSidebar();
    if (activeIdx !== null) renderGraph();
  } catch (e) {
    console.error('poll failed', e);
  }
}

poll();
setInterval(poll, 2000);
</script>
</body>
</html>
"""


def to_theorem_data(trace_path: Path) -> list[dict]:
    """Parse the trace file fresh and mark each node finished/running."""
    if not trace_path.exists():
        return []
    groups = load_trace(trace_path)
    out = []
    for thm_name, events in groups.items():
        data = build_theorem_data(thm_name, events)
        data["finished"] = any(e["kind"] == "final_verify" for e in events)
        out.append(data)
    out.sort(key=lambda t: (t["finished"], t["thm_name"]))
    return out


def _resolve_trace_file(target: Path) -> Path | None:
    """
    If `target` is a directory, follow whichever trace_*.jsonl file inside it
    was written to most recently (i.e. the problem currently being proved).
    If `target` is a file, watch it directly (single-problem mode).
    """
    if target.is_dir():
        candidates = list(target.glob("trace*.jsonl"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return target if target.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-polling Goedel-Architect trace viewer")
    parser.add_argument(
        "trace",
        help="Directory of trace_<problem>.jsonl files to follow (auto-picks the most "
             "recently written one), or a single JSONL trace file to watch directly.",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    trace_target = Path(args.trace)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass  # keep stdout quiet

        def do_GET(self):
            if self.path == "/data.json":
                current = _resolve_trace_file(trace_target)
                payload = {
                    "trace_file": current.name if current else None,
                    "nodes": to_theorem_data(current) if current else [],
                }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = LIVE_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Watching: {trace_target}")
    print(f"Open in browser: http://localhost:{args.port}/")
    print("Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
