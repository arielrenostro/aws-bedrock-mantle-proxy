"""Covers app.orchestrator.handle_stream directly: same-contract streams are
relayed as raw bytes, cross-contract streams get parsed and translated.
"""

import json
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
    async def fake(contract, payload, extra_headers, openai_path_prefix=False):
        captured["contract"] = contract
        captured["payload"] = payload
        captured["openai_path_prefix"] = openai_path_prefix
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


@pytest.mark.asyncio
async def test_stream_passes_openai_path_prefix_for_flagged_model(monkeypatch, tmp_path):
    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_text(
        json.dumps({"google.gemma-4-31b": {"contract": "openai", "openai_path_prefix": True}})
    )
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)

    lines = ['data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}', "data: [DONE]"]
    response = _FakeStreamResponse(200, lines)
    captured: dict = {}
    monkeypatch.setattr(mantle_client, "open_stream", _fake_open_stream(response, captured))

    body = {"model": "google.gemma-4-31b", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    _ = [c async for c in orchestrator.handle_stream(Contract.OPENAI, body, {})]

    assert captured["openai_path_prefix"] is True


@pytest.mark.asyncio
async def test_stream_no_openai_path_prefix_when_target_is_anthropic(monkeypatch, tmp_path):
    """Even if the overrides file (hypothetically) had prefix data keyed to
    this model, target == ANTHROPIC must force openai_path_prefix False —
    the flag only means something for the OpenAI contract."""
    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_text(json.dumps({"anthropic.claude-sonnet-4-6-v1": "anthropic"}))
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)

    lines = [
        'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}',
        'data: {"type":"content_block_stop","index":0}',
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
    _ = [c async for c in orchestrator.handle_stream(Contract.OPENAI, body, {})]

    assert captured["openai_path_prefix"] is False


@pytest.mark.asyncio
async def test_log_raw_chunks_is_a_faithful_pass_through():
    """Diagnostic tap added for the Gemma-4-through-Claude-Code report
    (translated stream produced only message_start/message_delta/
    message_stop — no way to tell from the outside whether Mantle sent zero
    chunks or chunks in an unrecognized shape). Must never drop or alter
    events, only observe them."""

    async def _events():
        for e in [{"a": 1}, {"b": 2}, {"c": 3}]:
            yield e

    out = [e async for e in orchestrator._log_raw_chunks(_events())]
    assert out == [{"a": 1}, {"b": 2}, {"c": 3}]


@pytest.mark.asyncio
async def test_log_raw_chunks_passes_through_empty_stream():
    async def _events():
        return
        yield  # pragma: no cover - makes this an async generator

    out = [e async for e in orchestrator._log_raw_chunks(_events())]
    assert out == []
