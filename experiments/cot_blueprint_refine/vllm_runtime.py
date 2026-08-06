from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from omegaconf import DictConfig

from cot_blueprint_refine.common import output_root, write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _endpoint_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host


def _hosts_match(left: str, right: str) -> bool:
    loopback = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "::"}
    return left == right or (left in loopback and right in loopback)


def validate_service_config(
    client_model: str,
    base_url: str,
    service: DictConfig,
) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid OpenAI base URL: {base_url!r}")
    expected_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not _hosts_match(parsed.hostname, str(service.host)):
        raise ValueError(
            f"vLLM host {service.host!r} does not match client URL {base_url!r}"
        )
    if expected_port != int(service.port):
        raise ValueError(
            f"vLLM port {service.port} does not match client URL {base_url!r}"
        )
    if client_model != str(service.served_model_name):
        raise ValueError(
            f"vLLM served model {service.served_model_name!r} does not match "
            f"client model {client_model!r}"
        )


class VLLMServer:
    def __init__(
        self,
        config: DictConfig,
        *,
        stage: str,
        client_model: str,
        base_url: str,
        service: DictConfig,
    ) -> None:
        self.config = config
        self.stage = stage
        self.client_model = client_model
        self.base_url = base_url
        self.service = service
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.started_at: str | None = None
        self.ready_at: str | None = None
        self.stop_reason = ""
        self._forced_kill = False
        self.root = output_root(config) / "vllm"
        self.log_path = self.root / f"{stage}.log"
        self.metadata_path = self.root / f"{stage}.json"

    @property
    def auto_start(self) -> bool:
        return bool(self.config.vllm.auto_start)

    @property
    def auto_destroy(self) -> bool:
        return bool(self.config.vllm.auto_destroy)

    def command(self) -> list[str]:
        service = self.service
        command = [
            str(self.config.python_bin),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(Path(str(service.model_path)).expanduser()),
            "--served-model-name",
            str(service.served_model_name),
            "--host",
            str(service.host),
            "--port",
            str(int(service.port)),
            "--tensor-parallel-size",
            str(int(service.tensor_parallel_size)),
            "--max-model-len",
            str(int(service.max_model_len)),
            "--max-num-seqs",
            str(int(service.max_num_seqs)),
            "--gpu-memory-utilization",
            str(float(service.gpu_memory_utilization)),
        ]
        if bool(service.get("trust_remote_code", False)):
            command.append("--trust-remote-code")
        reasoning_parser = service.get("reasoning_parser")
        if reasoning_parser:
            command.extend(["--reasoning-parser", str(reasoning_parser)])
        tool_call_parser = service.get("tool_call_parser")
        if tool_call_parser:
            command.extend(["--tool-call-parser", str(tool_call_parser)])
        if bool(service.get("enable_auto_tool_choice", False)):
            command.append("--enable-auto-tool-choice")
        command.extend(str(value) for value in (service.get("extra_args") or []))
        return command

    def _metadata(self, status: str) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": status,
            "pid": None if self.process is None else self.process.pid,
            "command": self.command(),
            "base_url": self.base_url,
            "model": self.client_model,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "stopped_at": _utc_now() if status in {"stopped", "startup_failed"} else None,
            "stop_reason": self.stop_reason,
            "forced_kill": self._forced_kill,
            "auto_start": self.auto_start,
            "auto_destroy": self.auto_destroy,
            "log_path": str(self.log_path),
        }

    def _write_metadata(self, status: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.metadata_path, self._metadata(status))

    def _port_is_in_use(self) -> bool:
        host = _endpoint_host(str(self.service.host))
        try:
            with socket.create_connection((host, int(self.service.port)), timeout=1.0):
                return True
        except OSError:
            return False

    def _available_models(self) -> set[str]:
        request = Request(self.base_url.rstrip("/") + "/models")
        with urlopen(request, timeout=3.0) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        return {
            str(item.get("id") or "")
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }

    def _log_tail(self, lines: int = 80) -> str:
        if not self.log_path.exists():
            return "(vLLM log was not created)"
        content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + float(self.config.vllm.startup_timeout_s)
        poll_interval = float(self.config.vllm.poll_interval_s)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM for stage {self.stage!r} exited during startup with "
                    f"code {self.process.returncode}:\n{self._log_tail()}"
                )
            try:
                available = self._available_models()
                if self.client_model in available:
                    self.ready_at = _utc_now()
                    self._write_metadata("ready")
                    print(
                        f"[vllm-ready] stage={self.stage} pid={self.process.pid if self.process else None} "
                        f"model={self.client_model} base_url={self.base_url}",
                        flush=True,
                    )
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(poll_interval)
        raise TimeoutError(
            f"vLLM for stage {self.stage!r} was not ready within "
            f"{self.config.vllm.startup_timeout_s}s:\n{self._log_tail()}"
        )

    def start(self) -> None:
        validate_service_config(self.client_model, self.base_url, self.service)
        if not self.auto_start:
            self._write_metadata("external_service_expected")
            return
        model_path = Path(str(self.service.model_path)).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"vLLM model path not found: {model_path}")
        if self._port_is_in_use():
            raise RuntimeError(
                f"vLLM stage {self.stage!r} requires exclusive port "
                f"{self.service.host}:{self.service.port}, but it is already in use"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        self.log_handle.write(
            f"\n===== stage={self.stage} launch={_utc_now()} command="
            f"{json.dumps(self.command(), ensure_ascii=False)} =====\n"
        )
        self.log_handle.flush()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.vllm.cuda_visible_devices)
        python_bin_dir = str(Path(str(self.config.python_bin)).expanduser().parent)
        env["PATH"] = os.pathsep.join(
            value for value in (python_bin_dir, env.get("PATH", "")) if value
        )
        self.started_at = _utc_now()
        try:
            self.process = subprocess.Popen(
                self.command(),
                cwd=str(Path(__file__).resolve().parents[2]),
                env=env,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except BaseException:
            self.stop_reason = "popen_failed"
            self._write_metadata("startup_failed")
            self.log_handle.close()
            self.log_handle = None
            raise
        self._write_metadata("starting")
        print(
            f"[vllm-start] stage={self.stage} pid={self.process.pid} log={self.log_path}",
            flush=True,
        )
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop_reason = "startup_failed"
            self.stop(force_cleanup=True)
            self._write_metadata("startup_failed")
            raise

    def stop(self, *, force_cleanup: bool = False) -> None:
        if self.process is None:
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            return
        if not self.auto_destroy and not force_cleanup:
            self.stop_reason = self.stop_reason or "auto_destroy_disabled"
            self._write_metadata("left_running")
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            print(
                f"[vllm-left-running] stage={self.stage} pid={self.process.pid}",
                flush=True,
            )
            return
        self.stop_reason = self.stop_reason or "stage_finished"
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=float(self.config.vllm.shutdown_timeout_s))
            except subprocess.TimeoutExpired:
                self._forced_kill = True
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=30.0)
        self._write_metadata("stopped")
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        print(
            f"[vllm-stop] stage={self.stage} pid={self.process.pid} "
            f"returncode={self.process.returncode} forced={self._forced_kill}",
            flush=True,
        )

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.stop_reason = f"stage_exception:{exc_type.__name__}"
        self.stop()
