"""Kimina runtime owned by the RobustPA experiment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kimina_lean_compiler import KiminaLeanCompiler


@dataclass
class LeanRuntime:
    compiler: KiminaLeanCompiler
    metadata: dict[str, Any]

    def close(self) -> None:
        self.compiler.close()


def make_lean_runtime(args) -> LeanRuntime:
    metadata = {
        "backend": "kimina_server",
        "api_url": str(args.lean_api_url).rstrip("/"),
        "timeout_s": args.lean_server_timeout,
        "reuse": args.lean_server_reuse,
        "debug": args.lean_server_debug,
        "check_concurrency": args.lean_check_concurrency,
    }
    compiler = KiminaLeanCompiler(
        api_url=args.lean_api_url,
        api_key_env=args.lean_api_key_env,
        timeout_s=args.lean_server_timeout,
        reuse=args.lean_server_reuse,
        debug=args.lean_server_debug,
        check_concurrency=args.lean_check_concurrency,
    )
    return LeanRuntime(compiler, metadata)


def write_lean_runtime_metadata(output_root: Path, metadata: dict[str, Any]) -> Path:
    path = output_root / "lean_runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
