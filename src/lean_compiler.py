"""Lean 4 compiler abstraction + default lake-env-lean implementation."""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from blueprint_text import BLUEPRINT_PROOF_RE, extract_current_node_decl

# goedel_lean project root — where lakefile.toml lives (depends on the real
# LeanArchitect package, see goedel_lean/lakefile.toml)
LEAN_PROJECT_ROOT = Path(__file__).parent.parent / "goedel_lean"

# Standard imports prepended to snippet-mode compilation.
# GoedelArch provides #validate_blueprint; Architect provides @[blueprint] / sorry_using.
MATHLIB_HEADER = "import Mathlib\nimport Architect\n\n"
GOEDEL_HEADER = "import Mathlib\nimport Architect\nimport GoedelArch\n\n"


@dataclass
class CompilerResult:
    success: bool
    goals: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_output: str = ""
    # True only when `success` came from an actual Lean compile (`lake env
    # lean` / LeanREPL). Some fallback paths (e.g. a structural-only check
    # when the target repo isn't on disk) report success without ever
    # invoking Lean - those must set this to False so callers can tell a
    # genuinely-validated result from a give-up/best-effort one.
    validated: bool = True

    @property
    def safeguard_rejected(self) -> bool:
        return "Safeguard rejected" in self.raw_output

    @property
    def has_sorry(self) -> bool:
        return any(
            "declaration uses" in w and ("sorry" in w or "admit" in w)
            for w in self.warnings
        )


_SORRY_USING_RE = BLUEPRINT_PROOF_RE

_FORBIDDEN_COMMANDS = {
    "axiom",
    "class",
    "inductive",
    "instance",
    "macro",
    "namespace",
    "notation",
    "section",
    "structure",
    "syntax",
    "variable",
}


def _mask_comments_and_strings(code: str) -> str:
    """Return code with comments/strings replaced by spaces, preserving lines."""
    out: list[str] = []
    i = 0
    block_depth = 0
    in_line_comment = False
    in_string = False
    while i < len(code):
        ch = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue

        if block_depth:
            if ch == "/" and nxt == "-":
                block_depth += 1
                out.extend("  ")
                i += 2
                continue
            if ch == "-" and nxt == "/":
                block_depth -= 1
                out.extend("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        if in_string:
            if ch == "\\" and nxt:
                out.extend("  ")
                i += 2
                continue
            if ch == "\"":
                in_string = False
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        if ch == "-" and nxt == "-":
            in_line_comment = True
            out.extend("  ")
            i += 2
            continue
        if ch == "/" and nxt == "-":
            block_depth = 1
            out.extend("  ")
            i += 2
            continue
        if ch == "\"":
            in_string = True
            out.append(" ")
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _command_words(line: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for ch in line.strip():
        if ch.isalnum() or ch in {"_", "'"}:
            current.append(ch)
        else:
            if current:
                words.append("".join(current))
                current = []
            if len(words) >= 2:
                break
    if current:
        words.append("".join(current))
    return words[:2]


def _find_forbidden_construct(code: str) -> str | None:
    """Find disallowed commands/tokens while ignoring comments and strings."""
    masked = _mask_comments_and_strings(code)
    in_attr = False
    for raw_line in masked.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if in_attr:
            if "]" in line:
                line = line.split("]", 1)[1].strip()
                in_attr = False
                if not line:
                    continue
            else:
                continue
        if line.startswith("@["):
            if "]" in line:
                line = line.split("]", 1)[1].strip()
                if not line:
                    continue
            else:
                in_attr = True
                continue
        words = _command_words(line)
        if not words:
            continue
        if words[:2] == ["noncomputable", "section"]:
            return "noncomputable section"
        if words[:2] == ["partial", "def"]:
            return "partial def"
        if words[:2] == ["local", "notation"]:
            return "local notation"
        if words[0] in _FORBIDDEN_COMMANDS:
            return words[0]

    current: list[str] = []
    for ch in masked:
        if ch.isalnum() or ch in {"_", "'"}:
            current.append(ch)
        else:
            token = "".join(current)
            if token == "native_decide":
                return token
            current = []
    if "".join(current) == "native_decide":
        return "native_decide"
    return None


def _assemble_node_attempt(
    node_decl: str,
    aux_lemmas: str,
    proof_body: str,
    header: str | None = None,
) -> str:
    """Build a standalone compilable file from a blueprint node's declaration.

    Strips the `@[blueprint ...]` attribute and swaps `:= by sorry_using [...]`
    for `:= ` + the prover's tactic proof.
    """
    decl_text = _extract_current_node_decl(node_decl)
    decl_text, replacements = _SORRY_USING_RE.subn(
        f":= {proof_body}", decl_text, count=1,
    )
    if replacements != 1:
        raise ValueError(
            "proof node declaration must contain exactly one "
            "`:= by sorry_using [...]` placeholder"
        )
    parts = [(header or MATHLIB_HEADER).rstrip("\n")]
    if aux_lemmas.strip():
        parts.append(aux_lemmas.strip())
    parts.append(decl_text)
    return "\n\n".join(parts) + "\n"


def _extract_current_node_decl(node_decl: str) -> str:
    """Extract only the declaration for the current blueprint node.

    Blueprint parsing can leave trailing text from the root theorem or a doc
    comment in `lean_declaration` when that trailing declaration is not itself
    annotated with @[blueprint]. The prover only wants to compile the current
    node, so stop at this node's `sorry_using [...]` proof body when present.
    """
    return extract_current_node_decl(node_decl)


class AbstractLeanCompiler(ABC):
    """Interface every Lean compiler backend must implement."""

    @abstractmethod
    def check(self, lean_code: str, **kwargs) -> CompilerResult: ...

    def check_blueprint(self, lean_code: str, target_name: str) -> CompilerResult:
        """Validate a @[blueprint]-annotated file. Default: no-op (just compile)."""
        return self.check(lean_code)

    async def check_async(self, lean_code: str, **kwargs) -> CompilerResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check, lean_code)


class LeanCompiler(AbstractLeanCompiler):
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or LEAN_PROJECT_ROOT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        lean_code: str,
        prepend_header: bool = False,
        aux_lemmas: str = "",
        node_decl: str = "",
        header: str | None = None,
        allow_sorry: bool = False,
        **_,
    ) -> CompilerResult:
        """Compile a Lean snippet and return structured feedback.

        If `prepend_header` is True, `import Mathlib\\nimport Architect` is
        prepended automatically.  For blueprint and prover code the imports
        are already present in the LLM-generated text, so False is the default.

        If `node_decl` is given, `lean_code` is treated as a bare proof body
        (tactics starting with `by`, no theorem header) that gets substituted
        into `node_decl`'s `:= by sorry_using [...]` tail — mirroring the
        VSBLeanCompiler contract where the prover only ever submits tactics.
        `aux_lemmas` (if any) is compiled ahead of the assembled declaration.
        """
        if node_decl:
            try:
                code = _assemble_node_attempt(node_decl, aux_lemmas, lean_code, header=header)
            except ValueError as exc:
                message = f"Node assembly rejected: {exc}"
                return CompilerResult(
                    success=False,
                    errors=[message],
                    raw_output=message,
                    validated=False,
                )
        else:
            code = (MATHLIB_HEADER + lean_code) if prepend_header else lean_code

        forbidden = _find_forbidden_construct(code)
        if forbidden:
            return CompilerResult(
                success=False,
                errors=[f"Safeguard rejected: forbidden construct `{forbidden}` is not allowed."],
                raw_output="Safeguard rejected",
            )

        result = self._run_lean(code)
        if result.success and result.has_sorry and not allow_sorry:
            return CompilerResult(
                success=False,
                goals=result.goals,
                errors=result.errors + ["Proof contains `sorry` — not a complete proof."],
                warnings=result.warnings,
                raw_output=result.raw_output,
            )
        return result

    def check_blueprint(self, lean_code: str, target_name: str) -> CompilerResult:
        """Compile blueprint code and run #validate_blueprint <target>.

        Prepends import GoedelArch (which provides #validate_blueprint) if not
        already present, then appends the validation command.
        """
        forbidden = _find_forbidden_construct(lean_code)
        if forbidden:
            return CompilerResult(
                success=False,
                errors=[f"Safeguard rejected: forbidden construct `{forbidden}` is not allowed."],
                raw_output="Safeguard rejected",
            )
        if "import GoedelArch" not in lean_code:
            # Insert GoedelArch import after the existing Architect import
            code = lean_code.replace(
                "import Architect",
                "import Architect\nimport GoedelArch",
                1,
            )
            if "import GoedelArch" not in code:
                code = GOEDEL_HEADER + lean_code
        else:
            code = lean_code
        code = code.rstrip() + f"\n\n#validate_blueprint {target_name}\n"
        return self._run_lean(code)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_lean(self, code: str) -> CompilerResult:
        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["lake", "env", "lean", "--json", tmp_path],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300,
            )
            return self._parse_output(result.stdout + result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return CompilerResult(success=False, errors=["Lean compilation timed out after 180s"])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _parse_output(self, raw: str, returncode: int) -> CompilerResult:
        goals: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                severity = msg.get("severity", "")
                # Lean 4 --json uses "data" for the message text
                text = msg.get("data") or msg.get("message") or ""
                if severity == "error":
                    errors.append(text)
                elif severity == "warning":
                    warnings.append(text)
                elif severity == "information":
                    if "⊢" in text or "goals" in text.lower():
                        goals.append(text)
            except json.JSONDecodeError:
                # Non-JSON line (shouldn't happen with --json, but be safe)
                if re.search(r"\berror\b", line, re.IGNORECASE):
                    errors.append(line)

        # sorry_using sorries are expected; don't treat them as errors
        errors = [e for e in errors if "declaration uses `sorry`" not in e]

        return CompilerResult(
            success=returncode == 0 and not errors,
            goals=goals,
            errors=errors,
            warnings=warnings,
            raw_output=raw,
        )
