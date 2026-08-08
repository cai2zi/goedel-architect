"""One-pass quality audit for the minimal Claim/Scope annotations.

This audit is analysis-only: it never rewrites a manifest or gates Blueprint
generation.  It asks whether the source organization itself preserved all
assertions at a useful granularity.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from cot_blueprint_refine.claim_scope_manifest import (  # noqa: E402
    decode_claim_scope_manifest,
)
from cot_blueprint_refine.common import latest_rows, load_config, output_root, write_json  # noqa: E402


AUDIT_VERSION = "claim-scope-quality-audit-v2"
_OPEN = "[[CLAIM_SCOPE_AUDIT_V1]]"
_CLOSE = "[[/CLAIM_SCOPE_AUDIT_V1]]"
_CODES = {
    "MISSING_ASSERTION", "NONCLAIM", "COMPOUND_CLAIM", "SCOPE_IS_ASSERTION",
    "SCOPE_UNDERREACH", "SCOPE_OVERREACH", "CLAIM_NOT_SELF_CONTAINED",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _render(manifest: Mapping[str, Any]) -> str:
    source = str(manifest["source_text"])
    semantic = sorted([
        (int(row["source_start"]), int(row["source_end"]), "claim", row)
        for row in manifest["claims"]
    ] + [
        (int(row["source_start"]), int(row["source_end"]), "scope", row)
        for row in manifest["scopes"]
    ])
    blocks: list[str] = []
    cursor = 0
    for start, end, kind, row in semantic:
        if cursor < start:
            blocks.append(f"[CONTEXT {cursor}:{start}]\n{source[cursor:start]}\n[/CONTEXT]")
        if kind == "claim":
            blocks.append(
                f"[CLAIM {row['claim_id']} scopes={','.join(row.get('scope_ids', [])) or 'none'}]\n"
                f"{source[start:end]}\n[/CLAIM]"
            )
        else:
            blocks.append(
                f"[SCOPE {row['scope_id']} type={row['scope_type']} "
                f"applies_to={','.join(row['applies_to_claim_ids'])}]\n"
                f"{source[start:end]}\n[/SCOPE]"
            )
        cursor = end
    if cursor < len(source):
        blocks.append(f"[CONTEXT {cursor}:{len(source)}]\n{source[cursor:]}\n[/CONTEXT]")
    return "\n\n".join(blocks)


def _messages(problem: str, manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    system = f"""You audit semantic organization, not mathematical correctness.
The source COT may be wrong. A false or unsupported assertion must still be a Claim.

Audit the entire annotated COT for only these defects:
- MISSING_ASSERTION: an independently checkable mathematical assertion is hidden in CONTEXT.
- NONCLAIM: a Claim contains no truth-evaluable mathematical proposition/action.
- COMPOUND_CLAIM: one Claim contains multiple independent propositions that should become
  separate formal nodes. Do not flag a single algebraic chain or one assertion plus explanation.
- SCOPE_IS_ASSERTION: a Scope is itself a complete conclusion/assertion rather than a modifier.
- SCOPE_UNDERREACH / SCOPE_OVERREACH: a Scope omits a Claim it semantically governs, or is
  attached to a later Claim it does not govern.
- CLAIM_NOT_SELF_CONTAINED: Claim plus its listed Scopes lacks essential local wording needed
  to know what proposition the source asserts.

Headings, separators, task requests, and method narration may remain CONTEXT. A given,
definition, equation, inequality, count, case conclusion, exhaustiveness/uniqueness statement,
derived value, verification, and final answer are assertions. Do not solve or repair the COT.
In particular, a symbolic equation, equality chain, intermediate algebraic transformation,
general identity/formula, or explicit definition is truth-evaluable and MUST NOT be called
NONCLAIM merely because it has no prose, is intermediate, or is not instantiated with numbers.
Do not demand coordinates, values, derivations, or constraints absent from the source. A task
request such as "find x" is not an assertion, but "x satisfies ...", "the region is a rhombus",
and "the area formula is ..." are assertions even when later claims add details.

Return exactly one {_OPEN} ... {_CLOSE} block containing JSON:
{{"verdict":"PASS","issues":[]}}
or {{"verdict":"FAIL","issues":[{{"code":"NONCLAIM","id":"C003","detail":"brief concrete reason"}}]}}.
Use a Claim/Scope ID for `id`; use `CONTEXT` for an unlabelled source span. Report every
material issue but do not nitpick stylistic grouping."""
    user = f"ORIGINAL PROBLEM:\n{problem}\n\nANNOTATED ORIGINAL COT:\n{_render(manifest)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse(content: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if content.count(_OPEN) != 1 or content.count(_CLOSE) != 1:
        raise ValueError("response must contain exactly one audit block")
    start = content.find(_OPEN) + len(_OPEN)
    end = content.find(_CLOSE)
    value = json.loads(content[start:end].strip())
    if not isinstance(value, dict) or value.get("verdict") not in {"PASS", "FAIL"}:
        raise ValueError("invalid audit verdict")
    issues = value.get("issues")
    if not isinstance(issues, list):
        raise ValueError("issues must be a list")
    valid_ids = {"CONTEXT"} | {
        str(row["claim_id"]) for row in manifest["claims"]
    } | {str(row["scope_id"]) for row in manifest["scopes"]}
    checked: list[dict[str, str]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            raise ValueError("issue must be an object")
        code = str(issue.get("code") or "")
        item_id = str(issue.get("id") or "")
        detail = str(issue.get("detail") or "").strip()
        if code not in _CODES or item_id not in valid_ids or not detail:
            raise ValueError(f"invalid issue: {issue}")
        checked.append({"code": code, "id": item_id, "detail": detail})
    if (value["verdict"] == "PASS") != (not checked):
        raise ValueError("verdict and issue list disagree")
    return {"verdict": str(value["verdict"]), "issues": checked}


async def _audit_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    *,
    row: Mapping[str, Any],
    model: str,
    timeout_s: float,
) -> dict[str, Any]:
    row_id = str(row["name"])
    manifest = decode_claim_scope_manifest(row["cot_manifest_json"])
    messages = _messages(str(row.get("problem") or ""), manifest)
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()
    request_messages = messages
    async with semaphore:
        for attempt_number in (1, 2):
            attempt_started = time.monotonic()
            attempt: dict[str, Any] = {"attempt": attempt_number}
            content = ""
            choice = None
            usage = None
            try:
                response = await client.chat.completions.create(
                    model=model, messages=request_messages, temperature=0.0,
                    max_tokens=8192, timeout=timeout_s,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                content = str(choice.message.content or "")
                parsed = _parse(content, manifest)
                usage = getattr(response, "usage", None)
                attempt.update({
                    "status": "ok", "raw_content": content,
                    "finish_reason": choice.finish_reason,
                    "usage": {key: int(value) for key in (
                        "prompt_tokens", "completion_tokens", "total_tokens"
                    ) if (value := getattr(usage, key, None)) is not None},
                    "latency_s": time.monotonic() - attempt_started,
                })
                attempts.append(attempt)
                return {
                    "ID": row_id, "status": "ok", "audit_version": AUDIT_VERSION, **parsed,
                    "manifest_sha256": _sha(str(row["cot_manifest_json"])),
                    "attempts": attempts, "latency_s": time.monotonic() - started,
                }
            except Exception as exc:  # noqa: BLE001
                attempt.update({
                    "status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "raw_content": content,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "usage": {key: int(value) for key in (
                        "prompt_tokens", "completion_tokens", "total_tokens"
                    ) if (value := getattr(usage, key, None)) is not None},
                    "latency_s": time.monotonic() - attempt_started,
                })
                attempts.append(attempt)
                if attempt_number == 1:
                    request_messages = [*messages, {
                        "role": "user",
                        "content": (
                            f"Your output was rejected: {exc}. Do not explain. Begin with the exact "
                            f"token {_OPEN}, use valid compact JSON with no LaTeX backslash in "
                            f"details, and end with the exact token {_CLOSE}."
                        ),
                    }]
    return {
        "ID": row_id, "status": "error", "audit_version": AUDIT_VERSION,
        "verdict": "",
        "issues": [], "manifest_sha256": _sha(str(row["cot_manifest_json"])),
        "attempts": attempts, "latency_s": time.monotonic() - started,
    }


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


async def run(profile: str) -> dict[str, Any]:
    config = load_config(profile, [])
    root = output_root(config)
    rows = latest_rows(root / "prepared" / "generation_inputs.jsonl", "name")
    out_dir = root / "claim_scope_quality_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"
    cached = {
        str(row.get("ID") or ""): row
        for row in latest_rows(path, "ID")
        if str(row.get("status") or "") == "ok"
        and str(row.get("audit_version") or "") == AUDIT_VERSION
    }
    pending = [row for row in rows if (
        str(row["name"]) not in cached
        or cached[str(row["name"])].get("manifest_sha256")
        != _sha(str(row["cot_manifest_json"]))
    )]
    client = AsyncOpenAI(
        api_key=str(config.cot_splitter.api_key),
        base_url=str(config.cot_splitter.openai_base_url), max_retries=0,
    )
    semaphore = asyncio.Semaphore(int(config.cot_splitter.concurrency))
    try:
        results = await asyncio.gather(*(
            _audit_one(
                client, semaphore, row=row,
                model=str(config.cot_splitter.model),
                timeout_s=float(config.cot_splitter.timeout_s),
            ) for row in pending
        ))
    finally:
        await client.close()
    selected = dict(cached)
    selected.update({str(row["ID"]): row for row in results})
    ordered = [selected[str(row["name"])] for row in rows]
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered
    ), encoding="utf-8")
    code_counts: Counter[str] = Counter()
    code_sample_counts: Counter[str] = Counter()
    for row in ordered:
        codes = [str(issue["code"]) for issue in row.get("issues", [])]
        code_counts.update(codes)
        code_sample_counts.update(set(codes))
    latencies = [float(row.get("latency_s") or 0) for row in results]
    summary = {
        "audit_version": AUDIT_VERSION, "rows": len(ordered), "new_requests": len(pending),
        "status_counts": dict(Counter(str(row.get("status")) for row in ordered)),
        "verdict_counts": dict(Counter(str(row.get("verdict")) for row in ordered)),
        "issue_counts": dict(sorted(code_counts.items())),
        "issue_sample_counts": dict(sorted(code_sample_counts.items())),
        "latency_s": {"p50": _percentile(latencies, .5), "p90": _percentile(latencies, .9), "max": max(latencies) if latencies else None},
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="qwen3_8b_397b_wrong76_claim_scope")
    args = parser.parse_args()
    asyncio.run(run(args.profile))


if __name__ == "__main__":
    main()
