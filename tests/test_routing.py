"""End-to-end-ish tests through the FastAPI routers, with `mantle_client`
monkeypatched so nothing touches the network or AWS credentials. These pin
down the exact URLs/headers used per contract, and — the point of this
layer — that a client can freely mix its entry format with a model that
needs the *other* Mantle contract and still get a correctly-shaped response.
"""

import json

import pytest
from fastapi.testclient import TestClient

import app.mantle_client as mantle_client
import app.model_registry as model_registry
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_real_model_listing(monkeypatch):
    # Keep contract resolution deterministic (prefix heuristic) without
    # hitting the network for every test.
    async def fake_list_models():
        return 200, {"data": []}, {}

    monkeypatch.setattr(mantle_client, "list_models", fake_list_models)


@pytest.fixture(autouse=True)
def _fake_token(monkeypatch):
    monkeypatch.setattr("app.auth.provide_token", lambda **kw: "tok")


def _fake_call(expected_contract, response_body, calls: list):
    async def fake_call(contract, payload, extra_headers, openai_path_prefix=False):
        calls.append(
            {
                "contract": contract,
                "payload": payload,
                "headers": extra_headers,
                "openai_path_prefix": openai_path_prefix,
            }
        )
        assert contract == expected_contract
        return 200, response_body, {}

    return fake_call


def test_anthropic_entry_same_contract_passthrough(monkeypatch):
    from app.contracts import Contract

    calls = []
    monkeypatch.setattr(
        mantle_client,
        "call",
        _fake_call(
            Contract.ANTHROPIC,
            {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": {}},
            calls,
        ),
    )

    resp = client.post(
        "/anthropic/v1/messages",
        json={"model": "anthropic.claude-sonnet-4-6-v1", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == [{"type": "text", "text": "hi"}]
    # Same-contract: request forwarded unmodified (still Anthropic-shaped).
    assert calls[0]["payload"]["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_entry_same_contract_passthrough(monkeypatch):
    from app.contracts import Contract

    calls = []
    monkeypatch.setattr(
        mantle_client,
        "call",
        _fake_call(
            Contract.OPENAI,
            {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "usage": {}},
            calls,
        ),
    )

    resp = client.post(
        "/openai/v1/chat/completions",
        json={"model": "qwen.qwen3-coder-30b-a3b-instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi"


def test_openai_entry_with_anthropic_only_model_translates_both_ways(monkeypatch):
    """The whole point of this layer: hit the OpenAI entry point, but ask
    for a model that only exists on Mantle's Anthropic contract — the
    request must be translated to Anthropic shape before it's sent, and the
    Anthropic-shaped response must come back as an OpenAI-shaped response."""
    from app.contracts import Contract

    calls = []
    monkeypatch.setattr(
        mantle_client,
        "call",
        _fake_call(
            Contract.ANTHROPIC,
            {
                "content": [{"type": "text", "text": "hello from claude"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
            calls,
        ),
    )

    resp = client.post(
        "/openai/v1/chat/completions",
        json={
            "model": "anthropic.claude-sonnet-4-6-v1",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    # Client gets an OpenAI-shaped response back, even though Mantle was
    # called through its Anthropic contract.
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello from claude"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}

    # And what was actually sent to Mantle was Anthropic-shaped (has
    # "max_tokens" / no "stream" collision, and messages are plain, no
    # OpenAI-only fields like tool_choice="auto" leaking through).
    sent = calls[0]["payload"]
    assert "max_tokens" in sent
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_entry_with_openai_only_model_translates_both_ways(monkeypatch):
    """Mirror case: hit the Anthropic entry point (e.g. Claude Code) but ask
    for a non-Claude model that only exists on Mantle's OpenAI contract."""
    from app.contracts import Contract

    calls = []
    monkeypatch.setattr(
        mantle_client,
        "call",
        _fake_call(
            Contract.OPENAI,
            {
                "choices": [{"message": {"role": "assistant", "content": "hello from qwen"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
            calls,
        ),
    )

    resp = client.post(
        "/anthropic/v1/messages",
        json={
            "model": "qwen.qwen3-coder-30b-a3b-instruct",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    # Client gets an Anthropic-shaped response back.
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "hello from qwen"}]
    assert body["usage"] == {"input_tokens": 2, "output_tokens": 3}

    sent = calls[0]["payload"]
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert "max_tokens" in sent  # translated into an OpenAI-shaped payload


def test_openai_models_endpoint_passthrough(monkeypatch):
    async def fake_list_models():
        return 200, {"data": [{"id": "some-model"}]}, {}

    monkeypatch.setattr(mantle_client, "list_models", fake_list_models)
    resp = client.get("/openai/v1/models")
    assert resp.status_code == 200
    assert resp.json() == {"data": [{"id": "some-model"}]}


def test_unknown_paths_404():
    assert client.get("/v1/models").status_code == 404
    assert client.post("/v1/messages", json={}).status_code == 404
    assert client.post("/v1/chat/completions", json={}).status_code == 404


def test_openai_entry_uses_openai_path_prefix_for_flagged_model(monkeypatch, tmp_path):
    """Regression test for Gemma 4 / GPT-5.x-style models that Mantle only
    serves on /openai/v1/chat/completions, not the bare /v1/chat/completions
    (calling the wrong one returns a confusing permission-denied error)."""
    from app.contracts import Contract

    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_text(
        json.dumps({"google.gemma-4-31b": {"contract": "openai", "openai_path_prefix": True}})
    )
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)

    calls = []
    monkeypatch.setattr(
        mantle_client,
        "call",
        _fake_call(
            Contract.OPENAI,
            {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "usage": {}},
            calls,
        ),
    )

    resp = client.post(
        "/openai/v1/chat/completions",
        json={"model": "google.gemma-4-31b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert calls[0]["openai_path_prefix"] is True


def test_openai_entry_defaults_no_prefix_for_unflagged_model(monkeypatch, tmp_path):
    from app.contracts import Contract

    monkeypatch.setattr(model_registry, "_overrides_path", lambda: tmp_path / "does-not-exist.json")

    calls = []
    monkeypatch.setattr(
        mantle_client,
        "call",
        _fake_call(
            Contract.OPENAI,
            {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "usage": {}},
            calls,
        ),
    )

    resp = client.post(
        "/openai/v1/chat/completions",
        json={"model": "qwen.qwen3-coder-30b-a3b-instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert calls[0]["openai_path_prefix"] is False
