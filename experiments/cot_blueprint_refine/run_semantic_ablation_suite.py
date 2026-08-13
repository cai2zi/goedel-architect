from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from omegaconf import OmegaConf

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR.parent))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_blueprint_refine.common import load_config, output_root, write_json
from cot_blueprint_refine.kimina_runtime import PersistentKiminaRuntime
from cot_blueprint_refine.run_experiment import preflight_kimina, preflight_model
from cot_blueprint_refine.summarize_semantic_ablation import build_summary
from cot_blueprint_refine.vllm_runtime import PersistentVLLMRuntime


SCRIPT_DIR = Path(__file__).resolve().parent / "script"
SUITE_EXP_NAME = "qwen3_8b_397b_semantic_audit_ablation_suite_runtime"
BATCHES = (
    (
        "qwen3_8b_397b_wrong76_subtractive_separate_t06",
        "qwen3_8b_397b_wrong76_subtractive_separate_t00",
    ),
    (
        "qwen3_8b_397b_wrong76_subtractive_joint_t06",
        "qwen3_8b_397b_wrong76_subtractive_joint_t00",
    ),
    ("qwen3_8b_397b_all646_subtractive_separate_t06",),
)


def _select_suite_exp_name(output_base: Path) -> str:
    primary = output_base / SUITE_EXP_NAME
    if not primary.exists():
        return SUITE_EXP_NAME
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{SUITE_EXP_NAME}_attempt_{timestamp}_{os.getpid()}"
    candidate = stem
    suffix = 1
    while (output_base / candidate).exists():
        candidate = f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _existing_vllm_is_compatible(config) -> bool:
    service = config.shared_vllm_397b
    host = "127.0.0.1" if str(service.host) in {"0.0.0.0", "::", ""} else str(service.host)
    port = int(service.port)
    try:
        with socket.create_connection((host, port), timeout=1):
            pass
    except OSError:
        return False
    try:
        with urlopen(f"http://{host}:{port}/health", timeout=5) as response:  # noqa: S310
            if not 200 <= response.status < 300:
                raise RuntimeError(f"health returned HTTP {response.status}")
            response.read()
        with urlopen(f"http://{host}:{port}/v1/models", timeout=5) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"port {host}:{port} is occupied by an incompatible or unhealthy service; "
            "it was left untouched"
        ) from exc
    available = {
        str(item.get("id") or "")
        for item in payload.get("data", []) if isinstance(item, dict)
    }
    required = str(config.blueprint.model)
    if required not in available:
        raise RuntimeError(
            f"existing vLLM at {host}:{port} does not serve {required!r}; "
            f"available={sorted(available)!r}; service was left untouched"
        )
    return True


def _preflight_eight_idle_gpus(config) -> None:
    configured = [
        item.strip() for item in str(
            config.vllm.cuda_visible_devices
        ).split(",") if item.strip()
    ]
    if len(configured) != 8:
        raise RuntimeError(f"suite requires 8 configured GPUs; got {configured!r}")
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    missing = sorted(set(configured) - {item.strip() for item in inventory})
    if missing:
        raise RuntimeError(f"configured GPUs are not visible: {missing!r}")
    active = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if active:
        raise RuntimeError(
            "397B suite requires all 8 GPUs idle; existing compute processes were "
            "left untouched:\n" + active
        )


def _run_batch(profiles: tuple[str, ...], output_base: Path) -> None:
    processes = [
        subprocess.Popen(
            [
                str(SCRIPT_DIR / f"{profile}.sh"),
                f"output_base={output_base}",
            ],
            start_new_session=True,
        )
        for profile in profiles
    ]
    try:
        return_codes = [process.wait() for process in processes]
    except BaseException:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        raise
    failures = [
        f"{profile}={code}"
        for profile, code in zip(profiles, return_codes, strict=True)
        if code != 0
    ]
    if failures:
        raise RuntimeError("semantic ablation batch failed: " + ", ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="fresh output root shared by all five experiments",
    )
    args = parser.parse_args()
    overrides = []
    if args.output_base is not None:
        overrides.append(f"output_base={args.output_base.expanduser().resolve()}")
    config = load_config(BATCHES[0][0], overrides)
    config.exp_name = _select_suite_exp_name(Path(str(config.output_base)))
    attach_existing = _existing_vllm_is_compatible(config)
    config.vllm.use_existing = attach_existing
    config.vllm.auto_start = not attach_existing
    config.vllm.auto_destroy = not attach_existing
    config.kimina.use_existing = True
    config.kimina.auto_start = False
    config.kimina.auto_destroy = False
    suite_root = output_root(config)
    if attach_existing:
        print(
            "[suite-vllm] attaching compatible existing service; "
            "the suite will not stop or destroy it",
            flush=True,
        )
    else:
        _preflight_eight_idle_gpus(config)
    suite_root.mkdir(parents=True)
    print(f"[suite-runtime] {suite_root}", flush=True)
    (suite_root / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True), encoding="utf-8",
    )

    previous_handlers = {}

    def terminate(signum, _frame) -> None:
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, terminate)
    try:
        with (
            PersistentVLLMRuntime(config) as vllm_runtime,
            PersistentKiminaRuntime(config) as kimina_runtime,
        ):
            vllm_runtime.ensure(
                stage="semantic_ablation_suite",
                client_model=str(config.blueprint.model),
                base_url=str(config.blueprint.openai_base_url),
                service=config.shared_vllm_397b,
            )
            kimina_runtime.ensure("semantic_ablation_suite")
            preflight_model(
                str(config.blueprint.model), str(config.blueprint.openai_base_url)
            )
            preflight_kimina(config)
            for profiles in BATCHES:
                print(f"[suite-batch-start] {','.join(profiles)}", flush=True)
                _run_batch(profiles, Path(str(config.output_base)))
                print(f"[suite-batch-done] {','.join(profiles)}", flush=True)
            write_json(
                suite_root / "semantic_ablation_summary.json",
                build_summary(Path(str(config.output_base))),
            )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
