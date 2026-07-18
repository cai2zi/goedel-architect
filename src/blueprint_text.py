"""Shared text helpers for the ``@[blueprint]`` Lean grammar.

This module deliberately has no project-local imports so both ``blueprint``
and ``lean_compiler`` can use the same regexes without creating a circular
dependency.
"""
from __future__ import annotations

import re


BLUEPRINT_DECL_KW = r"(?:noncomputable\s+def|def|lemma|theorem|abbrev)"
BLUEPRINT_ATTR_RE = re.compile(
    rf"@\[blueprint\s*.*?\]\s*(?={BLUEPRINT_DECL_KW}\s+\w+)",
    re.DOTALL,
)
BLUEPRINT_PROOF_RE = re.compile(
    r":=\s*by\s+sorry_using\s*\[[^\]]*\]",
    re.DOTALL,
)
DECL_START_RE = re.compile(
    rf"^[ \t]*{BLUEPRINT_DECL_KW}\s+\w+",
    re.MULTILINE,
)
LEMMA_KW_RE = re.compile(r"(?m)^(\s*)lemma\b")


def strip_blueprint_attr(text: str) -> str:
    """Remove blueprint attributes without stopping at brackets in comments."""
    return BLUEPRINT_ATTR_RE.sub("", text)


def lemma_to_theorem(text: str) -> str:
    """Convert declaration-level ``lemma`` keywords to ``theorem``."""
    return LEMMA_KW_RE.sub(r"\1theorem", text)


def extract_blueprint_signature(text: str) -> str:
    """Return a node declaration without its blueprint attribute or proof."""
    text = lemma_to_theorem(strip_blueprint_attr(text))
    proof_match = BLUEPRINT_PROOF_RE.search(text)
    if proof_match:
        return text[:proof_match.start()].strip()

    # Compatibility for old checkpoints or non-standard declarations that do
    # not use the blueprint ``:= by sorry_using [...]`` proof placeholder.
    return text.split(":=", 1)[0].strip()


def extract_current_node_decl(text: str) -> str:
    """Return the current declaration, excluding its blueprint attribute.

    The attribute must be removed before looking for a declaration keyword:
    statement/proof doc comments are free text and may themselves contain
    phrases such as ``theorem statement``.
    """
    stripped = strip_blueprint_attr(text)
    start = DECL_START_RE.search(stripped)
    if not start:
        return stripped.strip()

    tail = stripped[start.start():]
    proof_match = BLUEPRINT_PROOF_RE.search(tail)
    if proof_match:
        return tail[:proof_match.end()].strip()
    return tail.strip()


def proof_body_to_decl_suffix(body: str) -> str:
    """Return a declaration suffix from a cached proof body."""
    stripped = body.strip()
    if stripped.startswith(":="):
        return stripped
    return f":= {stripped}"
