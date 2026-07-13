"""Covers app.orchestrator.handle_stream directly: same-contract streams are
relayed as raw bytes, cross-contract streams get parsed and translated.
"""

from contextlib import asynccontextmanager

import pytest

import app.mantle_client as mantle_client
import app.model_registry as model_registry
import app.orchestrator as orchestrator
from app.contracts import Contract


class _FakeStreamResponse:
    def __init__(self, status_code: int, lines: list[str]):
        self.status_code = status_code
        self._lines = lines
        self.headers = {}

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aiter_bytes(self):
        for line in self._lines:
            yield (line + "\n").encode()

    async def aread(self):
        return "\n".join(self._lines).encode()


def _fake_open_stream(response: _FakeStreamResponse, captured: dict):
    @asynccontextmanager
    async def fake(contract, payload, extra_headers):
        captured["contract"] = contract
        captured["payload"] = payload
        yield response

    return fake


@pytest.fixture(autouse=True)
def _no_real_model_listing(monkeypatch):
    async def fake_list_models():
        return 200, {"data": []}, {}

    monkeypatch.setattr(mantle_client, "list_models", fake_list_models)


@pytest.mark.asyncio
async def test_same_contract_stream_is_relayed_raw(monkeypatch):
    lines = ['data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}', "data: [DONE]"]
    response = _FakeStreamResponse(200, lines)
    captured: dict = {}
    monkeypatch.setattr(mantle_client, "open_stream", _fake_open_stream(response, captured))

    body = {"model": "qwen.some-model", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    chunks = [c async for c in orchestrator.handle_stream(Contract.OPENAI, body, {})]
    joined = b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks)

    assert captured["contract"] == Contract.OPENAI
    assert b'"content":"hi"' in joined
    assert b"[DONE]" in joined


@pytest.mark.asyncio
async def test_cross_contract_stream_is_translated(monkeypatch):
    """Entry is OpenAI, but the model only lives on Mantle's Anthropic
    contract — the raw Anthropic SSE from Mantle must come back to the
    client translated into OpenAI chat.completion.chunk SSE."""
    lines = [
        'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}',
        'data: {"type":"message_stop"}',
    ]
    response = _FakeStreamResponse(200, lines)
    captured: dict = {}
    monkeypatch.setattr(mantle_client, "open_stream", _fake_open_stream(response, captured))

    body = {
        "model": "anthropic.claude-sonnet-4-6-v1",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    chunks = [c async for c in orchestrator.handle_stream(Contract.OPENAI, body, {})]
    joined = "".join(c if isinstance(c, str) else c.decode() for c in chunks)

    assert captured["contract"] == Contract.ANTHROPIC
    # Request sent to Mantle was translated into Anthropic shape.
    assert "max_tokens" in captured["payload"]
    # Response back to the (OpenAI-entry) client is OpenAI-chunk-shaped.
    assert '"object": "chat.completion.chunk"' in joined
    assert '"content": "hi"' in joined
    assert '"finish_reason": "stop"' in joined
    assert "data: [DONE]" in joined


@pytest.mark.asyncio
async def test_stream_error_status_is_translated_to_entry_error_shape(monkeypatch):
    response = _FakeStreamResponse(400, ['{"error":{"message":"bad model id"}}'])
    captured: dict = {}
    monkeypatch.setattr(mantle_client, "open_stream", _fake_open_stream(response, captured))

    body = {"model": "anthropic.claude-sonnet-4-6-v1", "stream": True, "messages": []}
    chunks = [c async for c in orchestrator.handle_stream(Contract.OPENAI, body, {})]
    joined = "".join(c if isinstance(c, str) else c.decode() for c in chunks)

    assert '"message": "bad model id"' in joined
    assert '"type": "api_error"' in joined
