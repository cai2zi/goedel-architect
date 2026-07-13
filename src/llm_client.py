"""Shared OpenAI-client construction, routing Fireworks- and Mistral-hosted
models to their own base_url instead of OpenAI's.

Fireworks addresses its models as "accounts/<org>/models/<name>" (e.g.
"accounts/fireworks/models/deepseek-v4-flash"), and its API is
OpenAI-compatible for both chat.completions and the Responses API, so a
model_id in that shape is routed there.

Mistral's "Labs" models (e.g. "labs-leanstral-1-5") are addressed with a
"labs-" prefix; Mistral's API is OpenAI-compatible for chat.completions
only — it has no Responses API equivalent, so callers must use
chat.completions.create with this client, not client.responses.create.

Everything else uses the normal OpenAI route unless GOEDEL_OPENAI_BASE_URL or
OPENAI_BASE_URL points it at another OpenAI-compatible endpoint.
"""
from __future__ import annotations

import os

from openai import OpenAI


def make_client(model_id: str, timeout: float | None = None) -> OpenAI:
    if model_id.startswith("accounts/"):
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FIREWORKS_API_KEY"],
            timeout=timeout,
        )
    if model_id.startswith("labs-"):
        return OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=os.environ["MISTRAL_API_KEY"],
            timeout=timeout,
        )
    kwargs = {"timeout": timeout}
    base_url = os.environ.get("GOEDEL_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("GOEDEL_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)
