import json

import pytest

from app.translation.converters import (
    anthropic_error_to_openai,
    anthropic_request_to_openai,
    anthropic_response_to_openai,
    openai_error_to_anthropic,
    openai_request_to_anthropic,
    openai_response_to_anthropic,
    translate_anthropic_stream_to_openai,
    translate_openai_stream_to_anthropic,
)


async def _achunks(chunks):
    for c in chunks:
        yield c


# ---------------------------------------------------------------------------
# Anthropic request -> OpenAI request
# ---------------------------------------------------------------------------


def test_anthropic_request_to_openai_simple_text():
    body = {
        "model": "m",
        "max_tokens": 100,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    out = anthropic_request_to_openai(body)
    assert out["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert out["messages"][1] == {"role": "user", "content": "Hello"}


def test_anthropic_request_to_openai_tool_use_round_trip():
    body = {
        "model": "m",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "Paris"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}],
            },
        ],
        "tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto"},
    }
    out = anthropic_request_to_openai(body)
    msgs = out["messages"]
    assert msgs[1]["tool_calls"][0]["id"] == "toolu_1"
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {"location": "Paris"}
    assert msgs[2] == {"role": "tool", "tool_call_id": "toolu_1", "content": "72F"}
    assert out["tools"][0]["function"]["name"] == "get_weather"
    assert out["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# OpenAI request -> Anthropic request
# ---------------------------------------------------------------------------


def test_openai_request_to_anthropic_simple_text():
    body = {
        "model": "anthropic.claude-sonnet-4-6-v1",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    out = openai_request_to_anthropic(body)
    assert out["system"] == "You are helpful."
    assert out["messages"] == [{"role": "user", "content": "Hello"}]
    assert out["max_tokens"] == 4096  # default applied since OpenAI body omitted it


def test_openai_request_to_anthropic_respects_explicit_max_tokens():
    body = {"model": "m", "max_tokens": 256, "messages": [{"role": "user", "content": "hi"}]}
    out = openai_request_to_anthropic(body)
    assert out["max_tokens"] == 256


def test_openai_request_to_anthropic_tool_use_round_trip():
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object"}}}
        ],
        "tool_choice": "auto",
    }
    out = openai_request_to_anthropic(body)
    msgs = out["messages"]
    assert msgs[1]["content"][0] == {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"location": "Paris"}}
    assert msgs[2]["content"][0] == {"type": "tool_result", "tool_use_id": "call_1", "content": "72F"}
    assert out["tools"][0]["name"] == "get_weather"
    assert out["tool_choice"] == {"type": "auto"}


def test_openai_request_to_anthropic_image_data_url():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                ],
            }
        ],
    }
    out = openai_request_to_anthropic(body)
    blocks = out["messages"][0]["content"]
    assert blocks[1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}


# ---------------------------------------------------------------------------
# Non-streaming responses, round trip both ways
# ---------------------------------------------------------------------------


def test_openai_response_to_anthropic_text():
    resp = {
        "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    out = openai_response_to_anthropic(resp, model="m")
    assert out["content"] == [{"type": "text", "text": "Hi!"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_anthropic_response_to_openai_text():
    resp = {
        "content": [{"type": "text", "text": "Hi!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    out = anthropic_response_to_openai(resp, model="m")
    assert out["choices"][0]["message"]["content"] == "Hi!"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_anthropic_response_to_openai_tool_use():
    resp = {
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "Paris"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    out = anthropic_response_to_openai(resp, model="m")
    msg = out["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["id"] == "toolu_1"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"location": "Paris"}
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_full_round_trip_preserves_text():
    original_anthropic = {
        "content": [{"type": "text", "text": "round trip"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    as_openai = anthropic_response_to_openai(original_anthropic, model="m")
    back_to_anthropic = openai_response_to_anthropic(as_openai, model="m")
    assert back_to_anthropic["content"] == [{"type": "text", "text": "round trip"}]
    assert back_to_anthropic["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_openai_error_to_anthropic():
    out = openai_error_to_anthropic({"error": {"message": "bad request", "type": "invalid_request_error"}})
    assert out == {"type": "error", "error": {"type": "api_error", "message": "bad request"}}


def test_anthropic_error_to_openai():
    out = anthropic_error_to_openai({"type": "error", "error": {"type": "api_error", "message": "bad request"}})
    assert out == {"error": {"message": "bad request", "type": "api_error"}}


# ---------------------------------------------------------------------------
# Streaming, both directions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_to_anthropic_text():
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}},
    ]
    events = [e async for e in translate_openai_stream_to_anthropic(_achunks(chunks), model="m")]
    joined = "".join(events)
    assert "message_start" in joined
    assert '"text": "Hel"' in joined
    assert '"stop_reason": "end_turn"' in joined
    assert "message_stop" in joined


@pytest.mark.asyncio
async def test_anthropic_stream_to_openai_text():
    events_in = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 7}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    events = [e async for e in translate_anthropic_stream_to_openai(_achunks(events_in), model="m")]
    joined = "".join(events)
    assert '"content": "Hel"' in joined
    assert '"content": "lo"' in joined
    assert '"finish_reason": "stop"' in joined
    assert '"prompt_tokens": 7' in joined
    assert '"completion_tokens": 2' in joined
    assert "data: [DONE]" in joined


@pytest.mark.asyncio
async def test_anthropic_stream_to_openai_tool_use():
    events_in = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"location":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
    ]
    events = [e async for e in translate_anthropic_stream_to_openai(_achunks(events_in), model="m")]
    joined = "".join(events)
    assert '"name": "get_weather"' in joined
    assert '\\"location\\":' in joined
    assert '"finish_reason": "tool_calls"' in joined


# ---------------------------------------------------------------------------
# Mid-stream errors — Mantle (and OpenAI-compatible backends generally) can
# send a failure as a data chunk over an otherwise-200 connection. Regression
# tests for the Gemma-4-through-Claude-Code report: generation failed on
# Mantle's side, but the error chunk had no "choices"/wasn't a recognized
# Anthropic event type, so it was silently skipped and the stream ended up
# looking like an empty, successful completion instead of a failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_mid_stream_error_is_surfaced_and_stops_the_stream():
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"error": {"code": "validation_error", "message": "Task submission failed: Generation failed"}},
        # Should never be reached — the stream must stop at the error.
        {"choices": [{"delta": {"content": "should not appear"}, "finish_reason": "stop"}]},
    ]
    events = [e async for e in translate_openai_stream_to_anthropic(_achunks(chunks), model="m")]
    joined = "".join(events)

    assert "event: error" in joined
    assert "Task submission failed: Generation failed" in joined
    assert "should not appear" not in joined
    # The stream ends at the error — no fake successful completion follows.
    assert "message_stop" not in joined


@pytest.mark.asyncio
async def test_anthropic_stream_mid_stream_error_is_surfaced_and_stops_the_stream():
    events_in = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "error", "error": {"type": "api_error", "message": "overloaded_error: upstream failed"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "should not appear"}},
    ]
    events = [e async for e in translate_anthropic_stream_to_openai(_achunks(events_in), model="m")]
    joined = "".join(events)

    assert "overloaded_error: upstream failed" in joined
    assert '"type": "api_error"' in joined
    assert "should not appear" not in joined
    assert "[DONE]" not in joined
