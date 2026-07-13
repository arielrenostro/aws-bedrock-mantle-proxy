"""Ties the layers together: resolve which contract the target model needs,
translate the request if the entry contract differs from it, call Mantle,
translate the response back. This module has no FastAPI/httpx-specific
knowledge beyond what `mantle_client` already exposes — it deals in plain
dicts and async iterators of SSE text, so it's the seam between the pure
translation layer and the HTTP-facing routers.
"""

import json
import logging
from typing import AsyncIterator

import httpx

from . import mantle_client
from .contracts import Contract
from .logging_context import current_model
from .model_registry import needs_openai_v1_prefix, resolve_contract
from .translation import converters as tr

logger = logging.getLogger(__name__)

_RAW_CHUNK_LOG_SAMPLE = 3


async def _log_raw_chunks(events: AsyncIterator[dict]) -> AsyncIterator[dict]:
    """Pass parsed SSE chunks through unchanged, logging a small sample plus
    the total count. This is the only way to tell, from the logs alone,
    whether an empty translated stream came from Mantle sending nothing
    usable versus Mantle sending chunks in a shape the translator doesn't
    recognize (e.g. a field name other than delta.content) — both look
    identical downstream without this."""
    count = 0
    async for event in events:
        count += 1
        if count <= _RAW_CHUNK_LOG_SAMPLE:
            logger.info("Mantle raw stream chunk #%d: %s", count, json.dumps(event)[:1000])
        yield event
    logger.info("Mantle raw stream yielded %d chunk(s) before translation", count)


def _translate_request(entry: Contract, target: Contract, body: dict) -> dict:
    if entry == target:
        return body
    if entry == Contract.ANTHROPIC:  # target == OPENAI
        return tr.anthropic_request_to_openai(body)
    return tr.openai_request_to_anthropic(body)  # entry == OPENAI, target == ANTHROPIC


def _translate_response(entry: Contract, target: Contract, model: str, resp_body: dict) -> dict:
    if entry == target:
        return resp_body
    if entry == Contract.ANTHROPIC:  # response came back in OPENAI shape
        return tr.openai_response_to_anthropic(resp_body, model)
    return tr.anthropic_response_to_openai(resp_body, model)  # response came back in ANTHROPIC shape


def _translate_error(entry: Contract, target: Contract, resp_body: dict) -> dict:
    if entry == target:
        return resp_body
    if entry == Contract.ANTHROPIC:
        return tr.openai_error_to_anthropic(resp_body)
    return tr.anthropic_error_to_openai(resp_body)


async def handle_request(entry: Contract, body: dict, extra_headers: dict) -> tuple[int, dict]:
    # The router already set current_model for the duration of this call
    # (it needs the var alive for its own error-logging too, which runs
    # after this coroutine returns) — nothing to set/reset here.
    model = body.get("model", "")
    target = await resolve_contract(model)
    openai_path_prefix = needs_openai_v1_prefix(model) if target == Contract.OPENAI else False
    logger.info(
        "Routing entry=%s target=%s openai_path_prefix=%s", entry.value, target.value, openai_path_prefix
    )

    payload = _translate_request(entry, target, body)
    status, resp_body, _ = await mantle_client.call(
        target, payload, extra_headers, openai_path_prefix=openai_path_prefix
    )

    if status >= 400:
        return status, _translate_error(entry, target, resp_body)
    return status, _translate_response(entry, target, model, resp_body)


async def handle_stream(entry: Contract, body: dict, extra_headers: dict) -> AsyncIterator[bytes | str]:
    # Unlike the non-streaming path, the router returns immediately after
    # constructing the StreamingResponse — this generator body is what
    # actually runs later, driven by Starlette as it sends the response. So
    # current_model has to be set/reset in here, for the generator's own
    # lifetime, not in the router.
    model = body.get("model", "")
    token = current_model.set(model or "-")
    try:
        target = await resolve_contract(model)
        openai_path_prefix = needs_openai_v1_prefix(model) if target == Contract.OPENAI else False
        logger.info(
            "Routing (stream) entry=%s target=%s openai_path_prefix=%s",
            entry.value,
            target.value,
            openai_path_prefix,
        )

        payload = _translate_request(entry, target, body)

        try:
            async with mantle_client.open_stream(
                target, payload, extra_headers, openai_path_prefix=openai_path_prefix
            ) as resp:
                logger.info(
                    "Mantle %s stream status=%s headers=%s", target.value, resp.status_code, dict(resp.headers)
                )
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    logger.error(
                        "Mantle returned an error for a streaming request: %s %s",
                        resp.status_code,
                        error_body[:2000],
                    )
                    yield tr.format_error_sse(entry, error_body)
                    return

                if entry == target:
                    chunk_count = 0
                    async for chunk in resp.aiter_bytes():
                        chunk_count += 1
                        yield chunk
                    logger.info("Mantle stream ended normally after %d chunk(s)", chunk_count)
                    return

                events = _log_raw_chunks(mantle_client.iter_sse_json(resp))
                event_count = 0
                if entry == Contract.ANTHROPIC:  # target == OPENAI
                    async for event in tr.translate_openai_stream_to_anthropic(events, model):
                        event_count += 1
                        yield event
                else:  # entry == OPENAI, target == ANTHROPIC
                    async for event in tr.translate_anthropic_stream_to_openai(events, model):
                        event_count += 1
                        yield event
                logger.info("Mantle stream translated into %d event(s)", event_count)
        except httpx.HTTPError as exc:
            logger.exception("Mantle stream request failed: %s", exc)
            yield tr.format_error_sse(entry, str(exc).encode())
    finally:
        current_model.reset(token)
