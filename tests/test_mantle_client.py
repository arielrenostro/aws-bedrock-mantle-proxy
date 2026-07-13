"""Tests for app.mantle_client: mainly the Anthropic-endpoint payload
sanitization Mantle currently requires (structured outputs / output_config
.format, context_management, ... aren't supported there and get rejected
with a 400 otherwise), plus that the client's anthropic-beta header is never
forwarded (Mantle rejects unrecognized beta flags outright).
"""

import pytest

import app.mantle_client as mantle_client
from app.contracts import Contract
from app.mantle_client import _strip_unsupported_anthropic_fields

# ---------------------------------------------------------------------------
# _strip_unsupported_anthropic_fields — pure, no I/O
# ---------------------------------------------------------------------------


def test_strips_output_config_format_only():
    payload = {
        "model": "anthropic.claude-sonnet-4-6-v1",
        "output_config": {"format": {"type": "json_schema", "schema": {}}, "effort": "high"},
    }
    out = _strip_unsupported_anthropic_fields(payload)
    assert out["output_config"] == {"effort": "high"}


def test_removes_output_config_entirely_when_format_was_the_only_key():
    payload = {"model": "m", "output_config": {"format": {"type": "json_schema", "schema": {}}}}
    out = _strip_unsupported_anthropic_fields(payload)
    assert "output_config" not in out


def test_leaves_payload_untouched_when_no_output_config():
    payload = {"model": "m", "messages": []}
    out = _strip_unsupported_anthropic_fields(payload)
    assert out == payload


def test_leaves_payload_untouched_when_output_config_has_no_format():
    payload = {"model": "m", "output_config": {"effort": "high"}}
    out = _strip_unsupported_anthropic_fields(payload)
    assert out["output_config"] == {"effort": "high"}


def test_does_not_mutate_the_original_payload_dict():
    original = {"model": "m", "output_config": {"format": {}, "effort": "high"}}
    _strip_unsupported_anthropic_fields(original)
    assert original["output_config"] == {"format": {}, "effort": "high"}


def test_strips_context_management():
    payload = {"model": "m", "context_management": {"edits": [{"type": "compact_20260112"}]}}
    out = _strip_unsupported_anthropic_fields(payload)
    assert "context_management" not in out


def test_leaves_payload_untouched_when_no_context_management():
    payload = {"model": "m", "messages": []}
    out = _strip_unsupported_anthropic_fields(payload)
    assert out == payload


def test_strips_both_unsupported_fields_at_once():
    payload = {
        "model": "m",
        "output_config": {"format": {}, "effort": "high"},
        "context_management": {"edits": []},
    }
    out = _strip_unsupported_anthropic_fields(payload)
    assert "context_management" not in out
    assert out["output_config"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# call() — sanitization only kicks in for the Anthropic contract
# ---------------------------------------------------------------------------


class _FakeHttpxResponse:
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self._json_body = json_body if json_body is not None else {}
        self.headers = {}

    def json(self):
        return self._json_body


class _FakeAsyncClient:
    last_json = None
    last_headers = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None, **kwargs):
        _FakeAsyncClient.last_json = json
        _FakeAsyncClient.last_headers = headers
        return _FakeHttpxResponse(json_body={"content": []})


@pytest.mark.asyncio
async def test_call_strips_format_for_anthropic_contract(monkeypatch):
    monkeypatch.setattr("app.auth.provide_token", lambda **kw: "tok")
    monkeypatch.setattr(mantle_client.httpx, "AsyncClient", _FakeAsyncClient)

    payload = {"model": "anthropic.claude-sonnet-4-6-v1", "output_config": {"format": {}, "effort": "high"}}
    await mantle_client.call(Contract.ANTHROPIC, payload, {})

    assert "format" not in _FakeAsyncClient.last_json.get("output_config", {})
    assert _FakeAsyncClient.last_json["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_call_does_not_touch_openai_contract_payload(monkeypatch):
    monkeypatch.setattr("app.auth.provide_token", lambda **kw: "tok")
    monkeypatch.setattr(mantle_client.httpx, "AsyncClient", _FakeAsyncClient)

    # output_config isn't a real OpenAI Chat Completions field, but this
    # proves the stripping is gated on contract rather than field presence
    # alone — it must be a no-op here regardless.
    payload = {"model": "m", "output_config": {"format": {}}}
    await mantle_client.call(Contract.OPENAI, payload, {})

    assert _FakeAsyncClient.last_json == payload


@pytest.mark.asyncio
async def test_call_strips_context_management_for_anthropic_contract(monkeypatch):
    monkeypatch.setattr("app.auth.provide_token", lambda **kw: "tok")
    monkeypatch.setattr(mantle_client.httpx, "AsyncClient", _FakeAsyncClient)

    payload = {"model": "anthropic.claude-sonnet-4-6-v1", "context_management": {"edits": []}}
    await mantle_client.call(Contract.ANTHROPIC, payload, {})

    assert "context_management" not in _FakeAsyncClient.last_json


# ---------------------------------------------------------------------------
# anthropic-beta header — deliberately never forwarded. Mantle's Anthropic
# endpoint returns "invalid beta flag" outright for beta values Claude Code
# sends by default (e.g. context-management-2025-06-27); since the request
# fields those betas would gate are already stripped, forwarding the header
# has no upside and only trades one hard failure for another.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_never_forwards_anthropic_beta_header(monkeypatch):
    monkeypatch.setattr("app.auth.provide_token", lambda **kw: "tok")
    monkeypatch.setattr(mantle_client.httpx, "AsyncClient", _FakeAsyncClient)

    payload = {"model": "anthropic.claude-sonnet-4-6-v1", "messages": []}
    await mantle_client.call(
        Contract.ANTHROPIC, payload, {"anthropic-beta": "context-management-2025-06-27"}
    )

    assert "anthropic-beta" not in _FakeAsyncClient.last_headers


@pytest.mark.asyncio
async def test_call_openai_contract_unaffected_by_anthropic_beta_header(monkeypatch):
    monkeypatch.setattr("app.auth.provide_token", lambda **kw: "tok")
    monkeypatch.setattr(mantle_client.httpx, "AsyncClient", _FakeAsyncClient)

    payload = {"model": "m", "messages": []}
    await mantle_client.call(Contract.OPENAI, payload, {"anthropic-beta": "context-management-2025-06-27"})

    assert "anthropic-beta" not in _FakeAsyncClient.last_headers
