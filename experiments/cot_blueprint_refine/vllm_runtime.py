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

from omegaconf import DictConfig, OmegaConf

from cot_blueprint_refine.common import append_jsonl, output_root, write_json


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
        self.pgid: int | None = None
        self.log_handle: Any = None
        self.started_at: str | None = None
        self.ready_at: str | None = None
        self.stop_reason = ""
        self._forced_kill = False
        self.preflight: dict[str, Any] = {}
        self.root = output_root(config) / "vllm"
        self.log_path = self.root / f"{stage}.log"
        self.metadata_path = self.root / f"{stage}.json"

    @property
    def auto_start(self) -> bool:
        return bool(self.config.vllm.auto_start)

    @property
    def auto_destroy(self) -> bool:
        return bool(self.config.vllm.auto_destroy)

    @property
    def use_existing(self) -> bool:
        # ``get`` preserves compatibility with historical profiles that
        # predate the explicit ownership switch.
        return bool(self.config.vllm.get("use_existing", False))

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
            "pgid": self.pgid,
            "returncode": None if self.process is None else getattr(self.process, "returncode", None),
            "command": None if self.use_existing else self.command(),
            "base_url": self.base_url,
            "model": self.client_model,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "stopped_at": _utc_now()
            if status in {
                "stopped", "startup_failed", "shutdown_failed",
                "detached_existing", "existing_preflight_failed",
            } else None,
            "stop_reason": self.stop_reason,
            "forced_kill": self._forced_kill,
            # External attachment never acquires lifecycle authority, even if
            # the managed-service defaults are true in the surrounding config.
            "auto_start": False if self.use_existing else self.auto_start,
            "auto_destroy": False if self.use_existing else self.auto_destroy,
            "configured_auto_start": self.auto_start,
            "configured_auto_destroy": self.auto_destroy,
            "use_existing": self.use_existing,
            "ownership": "external" if self.use_existing else "managed",
            "preflight": self.preflight,
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

    def _process_group_alive(self) -> bool:
        """Check the owned PGID without signalling any unrelated process."""
        if self.process is not None and self.process.poll() is None:
            return True
        if self.pgid is None:
            return False
        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return self.process is not None and self.process.poll() is None
        for stat_path in proc_root.glob("[0-9]*/stat"):
            try:
                # Fields after the final ')' begin with state, ppid, pgrp.
                suffix = stat_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
                if len(suffix) > 2 and int(suffix[2]) == self.pgid:
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False

    def _wait_for_group_exit(self, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if not self._process_group_alive():
                return True
            time.sleep(0.1)
        return not self._process_group_alive()

    def _wait_for_port_release(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._port_is_in_use():
                return True
            time.sleep(0.1)
        return not self._port_is_in_use()

    def _available_models(self) -> set[str]:
        request = Request(self.base_url.rstrip("/") + "/models")
        with urlopen(request, timeout=3.0) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        return {
            str(item.get("id") or "")
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }

    def _health_url(self) -> str:
        parsed = urlparse(self.base_url)
        return parsed._replace(path="/health", params="", query="", fragment="").geturl()

    def _check_health(self) -> None:
        request = Request(self._health_url())
        with urlopen(request, timeout=3.0) as response:  # noqa: S310
            status = int(getattr(response, "status", 200))
            if status < 200 or status >= 300:
                raise RuntimeError(f"health endpoint returned HTTP {status}")
            response.read()

    def _preflight_existing(self) -> None:
        self.started_at = _utc_now()
        self.preflight = {
            "checked_at": self.started_at,
            "host": str(self.service.host),
            "port": int(self.service.port),
            "host_port_reachable": False,
            "health_ok": False,
            "model_ok": False,
            "available_models": [],
        }
        if not self._port_is_in_use():
            raise RuntimeError(
                f"vLLM use_existing=true requires a service at "
                f"{self.service.host}:{self.service.port}, but the endpoint is not reachable"
            )
        self.preflight["host_port_reachable"] = True
        try:
            self._check_health()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"existing vLLM health preflight failed at {self._health_url()}: {exc}"
            ) from exc
        self.preflight["health_ok"] = True
        try:
            available = self._available_models()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"existing vLLM model preflight failed at {self.base_url.rstrip('/')}/models: {exc}"
            ) from exc
        self.preflight["available_models"] = sorted(available)
        if self.client_model not in available:
            raise RuntimeError(
                f"existing vLLM at {self.base_url!r} does not serve required model "
                f"{self.client_model!r}; available={sorted(available)!r}"
            )
        self.preflight["model_ok"] = True
        self.ready_at = _utc_now()

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
        if self.use_existing:
            try:
                self._preflight_existing()
            except BaseException:
                self.stop_reason = "existing_preflight_failed"
                self._write_metadata("existing_preflight_failed")
                raise
            self._write_metadata("attached_existing")
            print(
                f"[vllm-existing-ready] stage={self.stage} pid=None "
                f"model={self.client_model} base_url={self.base_url}",
                flush=True,
            )
            return
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
            try:
                self.pgid = os.getpgid(self.process.pid)
            except ProcessLookupError:
                # Preserve the launch-time session identity for diagnostics and
                # cleanup even if a mocked or immediately failing child exits.
                self.pgid = self.process.pid
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
        if self.use_existing:
            self.stop_reason = self.stop_reason or "runtime_detached"
            self._write_metadata("detached_existing")
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            print(
                f"[vllm-existing-detach] stage={self.stage} pid=None "
                f"model={self.client_model}",
                flush=True,
            )
            return
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
        shutdown_deadline = time.monotonic() + float(self.config.vllm.shutdown_timeout_s)
        if self.process.poll() is None or self._process_group_alive():
            try:
                os.killpg(self.pgid or self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=max(0.1, shutdown_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        group_exited = self._wait_for_group_exit(shutdown_deadline)
        if not group_exited:
            self._forced_kill = True
            try:
                os.killpg(self.pgid or self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if self.process.poll() is None:
                self.process.wait(timeout=30.0)
            self._wait_for_group_exit(time.monotonic() + 30.0)
        port_released = self._wait_for_port_release()
        group_exited = not self._process_group_alive()
        if not group_exited or not port_released:
            self._write_metadata("shutdown_failed")
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            raise RuntimeError(
                f"owned vLLM shutdown incomplete: pgid={self.pgid} "
                f"group_alive={not group_exited} port_in_use={not port_released}"
            )
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


class PersistentVLLMRuntime:
    """Own one vLLM process and attach compatible model stages to it.

    The manager deliberately compares the complete effective service definition,
    rather than only the endpoint. This prevents silently reusing a process whose
    parsers, context window, or scheduling limits differ from the requested stage.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.root = output_root(config) / "vllm"
        self.attachments_path = self.root / "stage_attachments.jsonl"
        self.session_path = self.root / "session.json"
        self.server: VLLMServer | None = None
        self.fingerprint: str | None = None
        self.started_at: str | None = None
        self.stopped_at: str | None = None
        self.start_count = 0
        self.stop_count = 0
        self.reuse_count = 0
        self.switch_count = 0
        self.external_attach_count = 0
        self.attached_stages: list[str] = []
        self.closed = False

    def _service_fingerprint(
        self,
        *,
        client_model: str,
        base_url: str,
        service: DictConfig,
    ) -> str:
        payload = {
            "client_model": client_model,
            "base_url": base_url.rstrip("/"),
            "python_bin": str(self.config.python_bin),
            "cuda_visible_devices": str(self.config.vllm.cuda_visible_devices),
            "use_existing": bool(self.config.vllm.get("use_existing", False)),
            "service": OmegaConf.to_container(service, resolve=True),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def pid(self) -> int | None:
        if self.server is None or self.server.process is None:
            return None
        return self.server.process.pid

    def _write_session(self, status: str) -> None:
        write_json(
            self.session_path,
            {
                "status": status,
                "pid": self.pid,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "updated_at": _utc_now(),
                "start_count": self.start_count,
                "stop_count": self.stop_count,
                "reuse_count": self.reuse_count,
                "switch_count": self.switch_count,
                "external_attach_count": self.external_attach_count,
                "use_existing": bool(self.config.vllm.get("use_existing", False)),
                "ownership": (
                    "external" if bool(self.config.vllm.get("use_existing", False))
                    else "managed"
                ),
                "attached_stages": self.attached_stages,
                "service_fingerprint": self.fingerprint,
            },
        )

    def ensure(
        self,
        *,
        stage: str,
        client_model: str,
        base_url: str,
        service: DictConfig,
    ) -> VLLMServer:
        if self.closed:
            raise RuntimeError("cannot attach a stage to a closed vLLM runtime")
        requested = self._service_fingerprint(
            client_model=client_model,
            base_url=base_url,
            service=service,
        )
        reused = self.server is not None and requested == self.fingerprint
        if self.server is not None and not reused:
            self.switch_count += 1
            self.server.stop_reason = f"service_switch_before:{stage}"
            was_existing = self.server.use_existing
            self.server.stop()
            if not was_existing:
                self.stop_count += 1
            self.server = None
        if self.server is None:
            candidate = VLLMServer(
                self.config,
                stage="experiment_397b",
                client_model=client_model,
                base_url=base_url,
                service=service,
            )
            candidate.start()
            self.server = candidate
            self.fingerprint = requested
            self.started_at = self.server.started_at or _utc_now()
            if self.server.use_existing:
                self.external_attach_count += 1
            else:
                self.start_count += 1
        else:
            self.reuse_count += 1
            print(
                f"[vllm-reuse] stage={stage} pid={self.pid} model={client_model}",
                flush=True,
            )
        attached_at = _utc_now()
        self.attached_stages.append(stage)
        append_jsonl(
            self.attachments_path,
            {
                "stage": stage,
                "attached_at": attached_at,
                "pid": self.pid,
                "model": client_model,
                "base_url": base_url,
                "reused": reused,
                "reuse_count": self.reuse_count,
                "use_existing": self.server.use_existing,
                "ownership": "external" if self.server.use_existing else "managed",
                "preflight": self.server.preflight,
            },
        )
        self._write_session("running")
        return self.server

    def close(self, *, reason: str = "experiment_finished") -> None:
        if self.closed:
            return
        self.closed = True
        if self.server is not None:
            self.server.stop_reason = reason
            was_existing = self.server.use_existing
            self.server.stop()
            if not was_existing:
                self.stop_count += 1
        self.stopped_at = _utc_now()
        self._write_session(
            "detached_existing"
            if bool(self.config.vllm.get("use_existing", False))
            else "stopped"
        )

    def __enter__(self) -> "PersistentVLLMRuntime":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        reason = "experiment_finished" if exc_type is None else f"experiment_exception:{exc_type.__name__}"
        self.close(reason=reason)
