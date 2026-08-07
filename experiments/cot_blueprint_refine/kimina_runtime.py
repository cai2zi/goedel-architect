from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.request import urlopen

from omegaconf import DictConfig

from cot_blueprint_refine.common import append_jsonl, output_root, write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PersistentKiminaRuntime:
    """Own the experiment's Kimina process and every REPL it creates."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.service = config.kimina
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.root = output_root(config) / "kimina"
        self.log_path = self.root / "server.log"
        self.session_path = self.root / "session.json"
        self.metrics_path = self.root / "kimina_metrics.jsonl"
        self.started_at: str | None = None
        self.ready_at: str | None = None
        self.stopped_at: str | None = None
        self.stop_reason = ""
        self.forced_kill = False
        self.attached_stages: list[str] = []
        self._metrics_stop = threading.Event()
        self._metrics_thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.service.host}:{int(self.service.port)}"

    def command(self) -> list[str]:
        return [str(self.config.python_bin), "-m", "server"]

    def child_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update({
            "LEAN_SERVER_HOST": str(self.service.host),
            "LEAN_SERVER_PORT": str(int(self.service.port)),
            "LEAN_SERVER_ENVIRONMENT": "prod",
            "LEAN_SERVER_LOG_LEVEL": "WARNING",
            "LEAN_SERVER_MAX_REPL_MEM": str(self.service.max_repl_mem),
            "LEAN_SERVER_MAX_REPLS": str(int(self.service.max_repls)),
            "LEAN_SERVER_MAX_REPL_USES": str(int(self.service.max_repl_uses)),
            "LEAN_SERVER_MAX_SNIPPETS_PER_REQUEST": str(
                int(self.service.max_snippets_per_request)
            ),
            "LEAN_SERVER_PROJECT_DIR": str(self.service.project_dir),
            "LEAN_SERVER_REPL_PATH": str(self.service.repl_path),
        })
        python_dir = str(Path(str(self.config.python_bin)).parent)
        env["PATH"] = os.pathsep.join((python_dir, env.get("PATH", "")))
        client_dir = str(Path(str(self.service.root)) / "client")
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (client_dir, env.get("PYTHONPATH", "")) if value
        )
        return env

    def _port_in_use(self) -> bool:
        try:
            with socket.create_connection(
                (str(self.service.host), int(self.service.port)), timeout=1.0,
            ):
                return True
        except OSError:
            return False

    def _metadata(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "pid": None if self.process is None else self.process.pid,
            "pgid": None if self.process is None else self.process.pid,
            "command": self.command(),
            "cwd": str(self.service.root),
            "base_url": self.base_url,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "stopped_at": self.stopped_at,
            "stop_reason": self.stop_reason,
            "forced_kill": self.forced_kill,
            "attached_stages": self.attached_stages,
            "settings": {
                "max_repl_mem": str(self.service.max_repl_mem),
                "max_repls": int(self.service.max_repls),
                "max_repl_uses": int(self.service.max_repl_uses),
                "max_snippets_per_request": int(self.service.max_snippets_per_request),
                "project_dir": str(self.service.project_dir),
                "repl_path": str(self.service.repl_path),
            },
            "log_path": str(self.log_path),
            "metrics_path": str(self.metrics_path),
        }

    def _write_metadata(self, status: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.session_path, self._metadata(status))

    def _log_tail(self, lines: int = 100) -> str:
        if not self.log_path.exists():
            return "(Kimina log was not created)"
        return "\n".join(
            self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )

    def _fetch_json(self, path: str, timeout: float = 3.0) -> dict[str, Any]:
        with urlopen(self.base_url + path, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + float(self.service.startup_timeout_s)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Kimina exited during startup with code {self.process.returncode}:\n"
                    f"{self._log_tail()}"
                )
            try:
                if self._fetch_json("/health").get("status") == "ok":
                    self.ready_at = _utc_now()
                    self._write_metadata("ready")
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(float(self.service.poll_interval_s))
        raise TimeoutError(
            f"Kimina was not ready within {self.service.startup_timeout_s}s:\n{self._log_tail()}"
        )

    def _sample_metrics(self) -> None:
        while not self._metrics_stop.wait(float(self.service.metrics_interval_s)):
            try:
                row = self._fetch_json("/health/stats")
                row["recorded_at"] = _utc_now()
                append_jsonl(self.metrics_path, row)
            except Exception as exc:  # noqa: BLE001
                append_jsonl(self.metrics_path, {
                    "recorded_at": _utc_now(), "status": "error", "error": str(exc),
                })

    def start(self) -> None:
        if self.process is not None:
            return
        if not bool(self.service.auto_start):
            raise RuntimeError("This experiment requires kimina.auto_start=true")
        if self._port_in_use():
            raise RuntimeError(
                f"Kimina requires exclusive port {self.service.host}:{self.service.port}, "
                "but it is already in use"
            )
        for required in (self.service.root, self.service.project_dir, self.service.repl_path):
            if not Path(str(required)).exists():
                raise FileNotFoundError(str(required))
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        self.started_at = _utc_now()
        try:
            self.process = subprocess.Popen(
                self.command(),
                cwd=str(self.service.root),
                env=self.child_environment(),
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self._write_metadata("starting")
            self._wait_ready()
        except BaseException:
            self.stop_reason = "startup_failed"
            self.stop(force=True)
            raise
        self._metrics_stop.clear()
        self._metrics_thread = threading.Thread(
            target=self._sample_metrics, name="kimina-metrics", daemon=True,
        )
        self._metrics_thread.start()
        print(f"[kimina-ready] pid={self.process.pid} base_url={self.base_url}", flush=True)

    def ensure(self, stage: str) -> None:
        if self.process is None:
            self.start()
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(f"Kimina is not running for stage {stage}:\n{self._log_tail()}")
        if stage not in self.attached_stages:
            self.attached_stages.append(stage)
            self._write_metadata("ready")

    def stop(self, *, force: bool = False) -> None:
        self._metrics_stop.set()
        if self._metrics_thread is not None:
            self._metrics_thread.join(timeout=10)
            self._metrics_thread = None
        if self.process is None:
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            return
        if not bool(self.service.auto_destroy) and not force:
            raise RuntimeError("Owned Kimina cannot be left running; set auto_destroy=true")
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=float(self.service.shutdown_timeout_s))
            except subprocess.TimeoutExpired:
                self.forced_kill = True
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=30)
        self.stopped_at = _utc_now()
        self.stop_reason = self.stop_reason or "experiment_finished"
        if self._port_in_use():
            raise RuntimeError("Owned Kimina exited but its port is still in use")
        self._write_metadata("stopped")
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        print(
            f"[kimina-stop] pid={self.process.pid} returncode={self.process.returncode} "
            f"forced={self.forced_kill}", flush=True,
        )

    def __enter__(self) -> "PersistentKiminaRuntime":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.stop_reason = f"experiment_exception:{exc_type.__name__}"
        self.stop()
