#!/usr/bin/env python3
"""
Start a local web server for the VeriSoftBench graph visualization.

Usage:
    python eval/serve_viz.py                              # uses latest trace in results/
    python eval/serve_viz.py results/verisoftbench/trace.jsonl
    python eval/serve_viz.py --port 9090
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Ensure eval/ is on the path
sys.path.insert(0, str(Path(__file__).parent))
import graph_viz

RESULTS_ROOT = Path(__file__).parent.parent / "results"


def find_latest_trace() -> Path | None:
    traces = list(RESULTS_ROOT.rglob("*.jsonl"))
    return max(traces, key=lambda p: p.stat().st_mtime) if traces else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Goedel-Architect VeriSoftBench visualization at localhost"
    )
    parser.add_argument(
        "trace", nargs="?",
        help="JSONL trace file (default: newest in results/)",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    trace_path = Path(args.trace) if args.trace else find_latest_trace()
    if trace_path is None or not trace_path.exists():
        print(
            "No trace file found.\n"
            "Run a benchmark first:\n"
            "  python eval/run_verisoftbench.py --repo VCV-io --limit 5 --trace results/verisoftbench/trace.jsonl"
        )
        sys.exit(1)

    print(f"Building graph from: {trace_path}")
    html_path = graph_viz.generate(trace_path)
    serve_dir = html_path.parent
    url = f"http://localhost:{args.port}/{html_path.name}"

    handler = partial(SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = HTTPServer(("127.0.0.1", args.port), handler)

    print(f"Open in browser: {url}")
    print("Ctrl+C to stop.\n")

    def open_browser() -> None:
        time.sleep(0.4)
        try:
            subprocess.Popen(["xdg-open", url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass  # xdg-open not available

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
