from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kimina_lean_compiler import KiminaLeanCompiler
from lean_compiler import AbstractLeanCompiler, LeanCompiler


DEFAULT_KIMINA_URL = "http://localhost:8000"


@dataclass
class LeanRuntime:
    compiler: LeanCompiler
    compiler_factory: Callable[[], AbstractLeanCompiler] | None
    metadata: dict[str, Any]

    def close(self) -> None:
        close = getattr(self.compiler, "close", None)
        if callable(close):
            close()


def add_lean_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lean-backend",
        choices=["kimina_server", "local"],
        default="kimina_server",
    )
    parser.add_argument(
        "--lean-api-url",
        default=os.environ.get("KIMINA_API_URL", DEFAULT_KIMINA_URL),
    )
    parser.add_argument("--lean-api-key-env", default="KIMINA_API_KEY")
    parser.add_argument("--lean-server-timeout", type=int, default=300)
    parser.add_argument(
        "--lean-server-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lean-server-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lean-check-concurrency", type=int, default=8)


def make_lean_runtime(args: argparse.Namespace) -> LeanRuntime:
    metadata = lean_runtime_metadata(args)
    if args.lean_backend == "local":
        return LeanRuntime(
            compiler=LeanCompiler(),
            compiler_factory=LeanCompiler,
            metadata=metadata,
        )
    compiler = KiminaLeanCompiler(
        api_url=args.lean_api_url,
        api_key_env=args.lean_api_key_env,
        timeout_s=args.lean_server_timeout,
        reuse=args.lean_server_reuse,
        debug=args.lean_server_debug,
        check_concurrency=args.lean_check_concurrency,
    )
    return LeanRuntime(compiler=compiler, compiler_factory=None, metadata=metadata)


def lean_runtime_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": args.lean_backend,
        "api_url": str(args.lean_api_url).rstrip("/"),
        "timeout_s": args.lean_server_timeout,
        "reuse": args.lean_server_reuse,
        "debug": args.lean_server_debug,
        "check_concurrency": args.lean_check_concurrency,
    }


def prepare_lean_runtime_metadata(
    output_root: Path,
    *,
    resume: bool,
    metadata: dict[str, Any],
) -> Path:
    path = output_root / "lean_runtime.json"
    if resume:
        if not path.exists():
            has_existing_output = output_root.exists() and any(output_root.iterdir())
            if has_existing_output:
                raise RuntimeError(
                    f"Cannot resume: {path} is missing; refusing to mix output without Lean runtime metadata."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                f.write("\n")
            return path
        with path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != metadata:
            raise RuntimeError(
                "Cannot resume with different Lean runtime metadata: "
                f"existing={existing!r}, requested={metadata!r}"
            )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path
