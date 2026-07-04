"""Loads prompt templates from the prompts/ directory and renders them.

Templates use a minimal {{variable}} / {{#if var}} ... {{/if}} syntax so the
prompt text stays readable as plain markdown without requiring Jinja2.
"""
from __future__ import annotations

import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load(name: str) -> str:
    """Read a prompt file by stem name (e.g. 'blueprint_system')."""
    return (PROMPTS_DIR / f"{name}.md").read_text()


def render(template: str, **kwargs) -> str:
    """
    Render a prompt template.

    Supported syntax:
      {{variable}}          — replaced with str(kwargs[variable])
      {{#if variable}} ... {{/if}}  — block included only when kwargs[variable] is truthy
    """
    # Process {{#if ...}} ... {{/if}} blocks
    def replace_if(m: re.Match) -> str:
        var = m.group(1).strip()
        body = m.group(2)
        return body if kwargs.get(var) else ""

    result = re.sub(
        r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}",
        replace_if,
        template,
        flags=re.DOTALL,
    )

    # Replace {{variable}} placeholders
    def replace_var(m: re.Match) -> str:
        key = m.group(1).strip()
        return str(kwargs.get(key, ""))

    result = re.sub(r"\{\{(\w+)\}\}", replace_var, result)
    return result
