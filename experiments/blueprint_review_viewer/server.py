"""Loopback-only, readonly HTTP reviewer for Blueprint experiments."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from blueprint_review_viewer.diff import whole_file_diff  # noqa: E402
from blueprint_review_viewer.review_schema import build_review_artifact  # noqa: E402


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ReviewStore:
    def __init__(self, experiment_root: Path):
        self.root = experiment_root.resolve()
        self._lock = threading.Lock()
        self._cases: dict[str, dict] = {}
        self._stamp: tuple[int, int] = (-1, -1)

    def _results_path(self) -> Path:
        return self.root / "results.jsonl"

    def _current_stamp(self) -> tuple[int, int]:
        results = self._results_path()
        index = self.root / "review_index.jsonl"
        return (results.stat().st_mtime_ns if results.exists() else -1,
                index.stat().st_mtime_ns if index.exists() else -1)

    def refresh(self) -> None:
        stamp = self._current_stamp()
        with self._lock:
            if stamp == self._stamp:
                return
            cases: dict[str, dict] = {}
            results = self._results_path()
            if results.is_file():
                for line in results.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    identifier = str(row.get("id", ""))
                    if not identifier:
                        continue
                    artifact_path = Path(str(row.get("review_artifact_path") or ""))
                    if artifact_path.is_file() and self._inside(artifact_path):
                        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                    else:
                        artifact = build_review_artifact(self.root, row)
                    cases[identifier] = artifact
            self._cases, self._stamp = cases, stamp

    def _inside(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    def summaries(self) -> list[dict]:
        self.refresh()
        with self._lock:
            result = []
            for identifier, artifact in self._cases.items():
                source, final = artifact.get("source", {}), artifact.get("result", {})
                validation = artifact.get("validation", {})
                audit = artifact.get("semanticAudit", {})
                result.append({"id": identifier, "sourceId": source.get("source_id", ""),
                               "subset": source.get("subset", ""), "status": final.get("status", ""),
                               "error": final.get("error", ""), "candidateCount": len(artifact.get("candidates", [])),
                               "leanPassed": bool(validation.get("leanPassed") or validation.get("passed")),
                               "semanticClassification": audit.get("classification", "") if isinstance(audit, dict) else ""})
            return sorted(result, key=lambda item: (item["subset"], item["sourceId"], item["id"]))

    def case(self, identifier: str) -> dict | None:
        self.refresh()
        with self._lock:
            return self._cases.get(identifier)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def make_handler(store: ReviewStore):
    static_root = Path(__file__).resolve().parent / "static"
    template = (Path(__file__).resolve().parent / "templates" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "BlueprintReview/1"

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[blueprint-review] {self.address_string()} {fmt % args}")

        def _send(self, status: int, content_type: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _api(self, parsed) -> bool:
            query = parse_qs(parsed.query)
            if parsed.path == "/api/meta":
                self._send(200, "application/json; charset=utf-8", _json_bytes({"readOnly": True, "schemaVersion": 1, "experimentRoot": str(store.root)})); return True
            if parsed.path == "/api/cases":
                self._send(200, "application/json; charset=utf-8", _json_bytes(store.summaries())); return True
            if parsed.path == "/api/case":
                identifier = query.get("id", [""])[0]
                case = store.case(identifier)
                self._send(200 if case else 404, "application/json; charset=utf-8", _json_bytes(case or {"error": "case not found"})); return True
            if parsed.path == "/api/diff":
                case = store.case(query.get("id", [""])[0])
                if not case:
                    self._send(404, "application/json; charset=utf-8", _json_bytes({"error": "case not found"})); return True
                by_id = {str(item.get("candidateId")): item for item in case.get("candidates", [])}
                left, right = by_id.get(query.get("left", [""])[0]), by_id.get(query.get("right", [""])[0])
                if not left or not right:
                    self._send(400, "application/json; charset=utf-8", _json_bytes({"error": "unknown candidate"})); return True
                self._send(200, "application/json; charset=utf-8", _json_bytes(whole_file_diff(left, right))); return True
            return False

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                if not self._api(parsed): self._send(404, "application/json; charset=utf-8", _json_bytes({"error": "not found"}))
                return
            if parsed.path in {"/", "/index.html"}:
                self._send(200, "text/html; charset=utf-8", template); return
            if parsed.path in {"/app.js", "/style.css"}:
                filename = parsed.path[1:]
                data = (static_root / filename).read_bytes()
                self._send(200, "application/javascript; charset=utf-8" if filename.endswith(".js") else "text/css; charset=utf-8", data); return
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

        def do_HEAD(self) -> None: self.do_GET()
        def do_POST(self) -> None: self._send(HTTPStatus.METHOD_NOT_ALLOWED, "application/json", b'{"error":"readonly"}')
        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Blueprint experiment reviewer (SSH -L by default).")
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ssh-target", default="", help="e.g. user@g0065; printed as a copyable SSH -L command")
    parser.add_argument("--unsafe-public", action="store_true", help="explicitly permit a non-loopback listener")
    args = parser.parse_args()
    if args.host not in LOOPBACK_HOSTS and not args.unsafe_public:
        raise SystemExit("Refusing non-loopback listener. Use --unsafe-public only if you intentionally accept network exposure.")
    root = args.experiment_root.resolve()
    if not (root / "results.jsonl").is_file():
        raise SystemExit(f"results.jsonl not found: {root}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(ReviewStore(root)))
    target = args.ssh_target or "<user>@<remote-host>"
    print(f"[blueprint-review] read-only URL: http://127.0.0.1:{args.port}")
    print(f"[blueprint-review] local tunnel: ssh -N -L {args.port}:127.0.0.1:{args.port} {target}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
