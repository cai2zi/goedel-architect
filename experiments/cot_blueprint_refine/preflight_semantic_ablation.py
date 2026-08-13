from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent))

from cot_blueprint_refine.common import load_config, output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("expected_exp_name")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.profile, args.overrides)
    if str(config.exp_name) != args.expected_exp_name:
        raise RuntimeError(
            f"profile exp_name={config.exp_name!s} expected={args.expected_exp_name!r}"
        )
    if bool(config.resume):
        raise RuntimeError("semantic ablation profiles must use resume=false")
    target = output_root(config)
    if target.exists():
        raise RuntimeError(f"refusing to reuse existing output directory: {target}")
    with urlopen("http://127.0.0.1:8000/health", timeout=5) as response:  # noqa: S310
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Kimina health returned HTTP {response.status}")
        response.read()
    with urlopen("http://127.0.0.1:8001/v1/models", timeout=5) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    available = {str(item.get("id") or "") for item in payload.get("data", [])}
    required = str(config.blueprint.model)
    if required not in available:
        raise RuntimeError(
            f"vLLM does not serve {required!r}; available={sorted(available)!r}"
        )
    print(
        f"[preflight-ok] profile={args.profile} exp_name={config.exp_name} "
        f"mode={config.blueprint.semantic_audit_mode} "
        f"semantic_temperature={config.blueprint.semantic_audit_temperature} "
        f"output={target}",
        flush=True,
    )


if __name__ == "__main__":
    main()
