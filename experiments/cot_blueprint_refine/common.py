from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "base.yaml"
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def load_config(profile: str, overrides: list[str]) -> DictConfig:
    configs_dir = DEFAULT_CONFIG.parent
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if profile and profile != "base":
        profile_path = configs_dir / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"profile config not found: {profile_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(profile_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def run_root(cfg: DictConfig) -> Path:
    return Path(str(cfg.output_base)).expanduser().resolve() / str(cfg.exp_name)


def output_root(cfg: DictConfig) -> Path:
    return run_root(cfg)


def prepared_dir(cfg: DictConfig) -> Path:
    return run_root(cfg) / "prepared"


def robustpa_dir(cfg: DictConfig) -> Path:
    return run_root(cfg) / "robustpa" / "blueprint"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def latest_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            latest[value] = row
    return latest


def latest_rows(path: Path, key: str) -> list[dict[str, Any]]:
    return list(latest_by(read_jsonl(path), key).values())


def stable_name(original_id: str) -> str:
    digest = hashlib.sha256(original_id.encode("utf-8")).hexdigest()[:16]
    return f"cot_{digest}"


def safe_component(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", text)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def tag_counts(text: str) -> tuple[int, int]:
    return len(THINK_OPEN_RE.findall(text)), len(THINK_CLOSE_RE.findall(text))


def restore_implicit_think_start(text: str) -> tuple[str, bool]:
    """Restore the Qwen3.5 think opener injected by its chat template."""
    opens, closes = tag_counts(text)
    if opens == 0 and closes == 1:
        return f"<think>\n{text}", True
    return text, False


def validate_think_and_extract(text: str) -> tuple[str, str]:
    """Return (post-think text, rejection reason)."""
    tags = sorted(
        [(match.start(), 1, match.end()) for match in THINK_OPEN_RE.finditer(text)]
        + [(match.start(), -1, match.end()) for match in THINK_CLOSE_RE.finditer(text)]
    )
    if not tags:
        return "", "missing_think_tags"
    depth = 0
    last_close_end = -1
    for _start, kind, end in tags:
        if kind == 1:
            depth += 1
        else:
            if depth <= 0:
                return "", "unmatched_think_close"
            depth -= 1
            if depth == 0:
                last_close_end = end
    if depth:
        return "", "unclosed_think"
    if last_close_end < 0:
        return "", "missing_think_close"
    post = text[last_close_end:].strip()
    if not post:
        return "", "empty_post_think"
    return post, ""


def extract_post_think(text: str) -> tuple[str, str]:
    normalized, _restored = restore_implicit_think_start(text)
    return validate_think_and_extract(normalized)


def extract_boxed_texts(text: str) -> list[str]:
    return [content for _start, _end, content in extract_boxed_spans(text)]


def extract_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    needle = r"\boxed{"
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos < 0:
            return spans
        index = pos + len(needle)
        depth = 1
        chars: list[str] = []
        while index < len(text) and depth:
            char = text[index]
            if char == "{":
                depth += 1
                chars.append(char)
            elif char == "}":
                depth -= 1
                if depth:
                    chars.append(char)
            else:
                chars.append(char)
            index += 1
        if depth == 0:
            spans.append((pos, index, "".join(chars).strip()))
        start = max(index, pos + len(needle))


def claimed_answer(text: str) -> str:
    boxes = extract_boxed_texts(text)
    if not boxes:
        return ""
    answer = boxes[-1].strip()
    return answer


def extract_boxed_contents(text: str) -> list[str]:
    return extract_boxed_texts(text)


def prompt_safe_comment_lines(label: str, text: str) -> list[str]:
    """Render arbitrary diagnostics in line comments so Lean remains parseable."""
    lines = str(text).splitlines() or [""]
    return [f"-- {label}: {lines[0]}", *[f"-- {line}" for line in lines[1:]]]


def response_to_json(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "dict"):
        return response.dict()
    return json.loads(json.dumps(response, default=str))


def result_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
