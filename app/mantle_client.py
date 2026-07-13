"""The only module that speaks HTTP to Amazon Bedrock Mantle.

Everything about *how* to reach Mantle (base URLs, auth header conventions
per contract, compression, timeouts) lives here. Callers (the orchestrator)
only deal with a `Contract` and a plain dict payload.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

from .auth import get_bedrock_token
from .config import settings
from .contracts import Contract

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


def _strip_unsupported_anthropic_fields(payload: dict) -> dict:
    """Mantle's native Anthropic Messages API endpoint doesn't support
    structured outputs yet — a request carrying `output_config.format` is
    rejected outright with `400 output_config.format: Extra inputs are not
    permitted`, even though the field is valid on the real Anthropic API.
    Claude Code sends it for some internal calls (e.g. session-title
    generation via a json_schema format), which would otherwise hard-fail
    every time. Drop just that sub-field rather than failing the whole
    request; other output_config keys (e.g. effort) are left untouched.
    """
    output_config = payload.get("output_config")
    if not isinstance(output_config, dict) or "format" not in output_config:
        return payload

    logger.warning(
        "Dropping unsupported output_config.format before forwarding to Mantle's "
        "Anthropic endpoint (structured outputs aren't supported there)."
    )
    payload = dict(payload)
    remaining = {k: v for k, v in output_config.items() if k != "format"}
    if remaining:
        payload["output_config"] = remaining
    else:
        payload.pop("output_config", None)
    return payload


def _target(
    contract: Contract, extra_headers: dict, token: str, openai_path_prefix: bool = False
) -> tuple[str, dict]:
    if contract == Contract.ANTHROPIC:
        url = f"{settings.mantle_base_url}/anthropic/v1/messages"
        headers = {
            # Mantle's native Anthropic Messages API endpoint authenticates
            # with x-api-key (Anthropic's own convention), unlike the
            # OpenAI-compatible endpoint below, which expects a Bearer token.
            "x-api-key": token,
            "anthropic-version": extra_headers.get("anthropic-version", DEFAULT_ANTHROPIC_VERSION),
            "Content-Type": "application/json",
        }
    else:
        # Some OpenAI-contract models (e.g. Gemma 4, GPT-5.x) are only
        # reachable on this second, "/openai"-prefixed path — see
        # model_registry.needs_openai_v1_prefix for how this is decided.
        path = "/openai/v1/chat/completions" if openai_path_prefix else "/v1/chat/completions"
        url = f"{settings.mantle_base_url}{path}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return url, headers


async def call(
    contract: Contract, payload: dict, extra_headers: dict, openai_path_prefix: bool = False
) -> tuple[int, dict, dict]:
    """Non-streaming call. Returns (status_code, parsed_json_body, headers)."""
    if contract == Contract.ANTHROPIC:
        payload = _strip_unsupported_anthropic_fields(payload)
    token = await get_bedrock_token()
    url, headers = _target(contract, extra_headers, token, openai_path_prefix)
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        body = {"raw": resp.text}
    return resp.status_code, body, dict(resp.headers)


@asynccontextmanager
async def open_stream(
    contract: Contract, payload: dict, extra_headers: dict, openai_path_prefix: bool = False
):
    """Async context manager yielding the live httpx streaming response."""
    if contract == Contract.ANTHROPIC:
        payload = _strip_unsupported_anthropic_fields(payload)
    token = await get_bedrock_token()
    url, headers = _target(contract, extra_headers, token, openai_path_prefix)
    headers = {
        **headers,
        "Accept": "text/event-stream",
        # Force an uncompressed response so no decoding step (gzip/deflate/
        # br) can ever desync mid-stream while we relay/parse bytes.
        "Accept-Encoding": "identity",
    }
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            yield resp


async def iter_sse_json(response: httpx.Response) -> AsyncIterator[dict]:
    """Parse a streaming response's `data: {...}` lines into dicts. Works
    for both Anthropic and OpenAI SSE framing — both put a JSON payload
    after `data: `; Anthropic's preceding `event: <type>` lines are ignored
    since the event type is already carried in the JSON payload's `type`."""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


async def list_models() -> tuple[int, dict, dict]:
    """Passthrough of Mantle's OpenAI-shaped GET /v1/models."""
    token = await get_bedrock_token()
    url = f"{settings.mantle_base_url}/v1/models"
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        body = {"raw": resp.text}
    return resp.status_code, body, dict(resp.headers)
