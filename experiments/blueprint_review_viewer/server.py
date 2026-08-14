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
from blueprint_review_viewer import REVIEW_SCHEMA_VERSION  # noqa: E402
from blueprint_review_viewer.review_schema import build_review_artifact  # noqa: E402


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_OUTPUT_BASE = Path("/ssd/czx/czx_work/cot_blueprint_refine")


def experiment_root(experiment_name: str, output_base: Path = DEFAULT_OUTPUT_BASE) -> Path:
    """Resolve an experiment name to its RobustPA Blueprint artifact directory."""
    name = experiment_name.strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"invalid experiment name: {experiment_name!r}")
    base = output_base.expanduser().resolve()
    root = (base / name / "robustpa" / "blueprint").resolve()
    try:
        root.relative_to(base)
    except ValueError as error:
        raise ValueError(f"experiment resolves outside output base: {experiment_name!r}") from error
    return root


class ReviewStore:
    def __init__(self, experiment_root: Path):
        self.root = experiment_root.resolve()
        self._lock = threading.Lock()
        self._cases: dict[str, dict] = {}
        self._stamp: tuple[int, int, int, int] = (-1, -1, -1, -1)
        self._load_warnings: list[dict[str, object]] = []

    def _results_path(self) -> Path:
        return self.root / "results.jsonl"

    def _current_stamp(self) -> tuple[int, int, int, int]:
        results = self._results_path()
        index = self.root / "review_index.jsonl"
        results_stat = results.stat() if results.exists() else None
        index_stat = index.stat() if index.exists() else None
        return (
            results_stat.st_mtime_ns if results_stat else -1,
            results_stat.st_size if results_stat else -1,
            index_stat.st_mtime_ns if index_stat else -1,
            index_stat.st_size if index_stat else -1,
        )

    def _result_rows(self, results: Path) -> tuple[list[dict], list[dict[str, object]]]:
        """Read an append-only JSONL snapshot without failing on an in-flight tail."""
        data = results.read_bytes()
        raw_lines = data.splitlines(keepends=True)
        rows: list[dict] = []
        warnings: list[dict[str, object]] = []
        for line_number, raw_line in enumerate(raw_lines, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row must be an object")
                rows.append(row)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                is_unterminated_tail = (
                    line_number == len(raw_lines) and not data.endswith(b"\n")
                )
                warnings.append({
                    "kind": "pendingTail" if is_unterminated_tail else "invalidResultRow",
                    "line": line_number,
                    "byteCount": len(raw_line),
                    "error": f"{type(error).__name__}: {error}",
                })
        return rows, warnings

    def refresh(self) -> None:
        stamp = self._current_stamp()
        with self._lock:
            if stamp == self._stamp:
                return
            cases: dict[str, dict] = {}
            warnings: list[dict[str, object]] = []
            results = self._results_path()
            if results.is_file():
                rows, warnings = self._result_rows(results)
                for row in rows:
                    identifier = str(row.get("id", ""))
                    if not identifier:
                        continue
                    artifact_path = Path(str(row.get("review_artifact_path") or ""))
                    artifact = None
                    if artifact_path.is_file() and self._inside(artifact_path):
                        loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
                        if (
                            isinstance(loaded, dict)
                            and loaded.get("schemaVersion") == REVIEW_SCHEMA_VERSION
                        ):
                            artifact = loaded
                    if artifact is None:
                        artifact = build_review_artifact(self.root, row)
                    cases[identifier] = artifact
            self._cases = cases
            self._load_warnings = warnings
            # Size is part of the stamp, so completion of an in-flight trailing
            # row invalidates this snapshot even on coarse-mtime filesystems.
            self._stamp = stamp

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

    def diagnostics(self) -> list[dict[str, object]]:
        self.refresh()
        with self._lock:
            return [dict(item) for item in self._load_warnings]


class ExperimentCatalog:
    def __init__(self, output_base: Path):
        self.output_base = output_base.expanduser().resolve()
        self._lock = threading.Lock()
        self._stores: dict[str, ReviewStore] = {}

    def experiment_names(self) -> list[str]:
        if not self.output_base.is_dir():
            return []
        names: list[str] = []
        for path in self.output_base.iterdir():
            if not path.is_dir():
                continue
            try:
                root = experiment_root(path.name, self.output_base)
            except ValueError:
                continue
            if (root / "results.jsonl").is_file():
                names.append(path.name)
        return sorted(names)

    def store(self, experiment_name: str) -> ReviewStore:
        if experiment_name not in self.experiment_names():
            raise ValueError(f"unknown or invalid experiment: {experiment_name!r}")
        with self._lock:
            store = self._stores.get(experiment_name)
            if store is None:
                store = ReviewStore(experiment_root(experiment_name, self.output_base))
                self._stores[experiment_name] = store
            return store


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def make_handler(catalog: ExperimentCatalog):
    static_root = Path(__file__).resolve().parent / "static"
    template = (Path(__file__).resolve().parent / "templates" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = f"BlueprintReview/{REVIEW_SCHEMA_VERSION}"

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
            if parsed.path == "/api/experiments":
                self._send(200, "application/json; charset=utf-8", _json_bytes({
                    "outputBase": str(catalog.output_base),
                    "experiments": catalog.experiment_names(),
                })); return True
            experiment_name = query.get("experiment", [""])[0]
            try:
                store = catalog.store(experiment_name)
            except ValueError as error:
                self._send(400, "application/json; charset=utf-8", _json_bytes({
                    "error": str(error),
                })); return True
            if parsed.path == "/api/meta":
                self._send(200, "application/json; charset=utf-8", _json_bytes({"readOnly": True, "schemaVersion": REVIEW_SCHEMA_VERSION, "experiment": experiment_name, "experimentRoot": str(store.root), "loadWarnings": store.diagnostics()})); return True
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
    parser = argparse.ArgumentParser(description="Read-only Blueprint experiment catalog and reviewer.")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f"experiment output base (default: {DEFAULT_OUTPUT_BASE})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--ssh-target", default="", help="e.g. user@g0065; printed as a copyable SSH -L command")
    parser.add_argument("--unsafe-public", action="store_true", help="explicitly permit a non-loopback listener")
    args = parser.parse_args()
    if args.host not in LOOPBACK_HOSTS and not args.unsafe_public:
        raise SystemExit("Refusing non-loopback listener. Use --unsafe-public only if you intentionally accept network exposure.")
    output_base = args.output_base.expanduser().resolve()
    if not output_base.is_dir():
        raise SystemExit(f"experiment output base not found: {output_base}")
    catalog = ExperimentCatalog(output_base)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog))
    target = args.ssh_target or "<user>@<remote-host>"
    print(f"[blueprint-review] read-only URL: http://127.0.0.1:{args.port}")
    print(
        f"[blueprint-review] output base: {output_base} "
        f"experiments={len(catalog.experiment_names())}"
    )
    print(f"[blueprint-review] local tunnel: ssh -N -L {args.port}:127.0.0.1:{args.port} {target}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
