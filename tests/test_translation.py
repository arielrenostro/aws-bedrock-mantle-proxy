import json

import pytest

from app.translation.anthropic_to_openai import anthropic_request_to_openai
from app.translation.openai_to_anthropic import (
    openai_response_to_anthropic,
    translate_openai_stream_to_anthropic,
)


def test_simple_text_request():
    body = {
        "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "max_tokens": 100,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    out = anthropic_request_to_openai(body)
    assert out["model"] == body["model"]
    assert out["max_tokens"] == 100
    assert out["stream"] is False
    assert out["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert out["messages"][1] == {"role": "user", "content": "Hello"}


def test_tool_use_round_trip_request():
    body = {
        "model": "m",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"location": "Paris"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "72F and sunny",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
        "tool_choice": {"type": "auto"},
    }
    out = anthropic_request_to_openai(body)
    msgs = out["messages"]

    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Let me check."
    assert msgs[1]["tool_calls"][0]["id"] == "toolu_1"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {"location": "Paris"}

    assert msgs[2] == {"role": "tool", "tool_call_id": "toolu_1", "content": "72F and sunny"}

    assert out["tools"][0]["type"] == "function"
    assert out["tools"][0]["function"]["name"] == "get_weather"
    assert out["tool_choice"] == "auto"


def test_non_streaming_response_text():
    openai_resp = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    out = openai_response_to_anthropic(openai_resp, model="m")
    assert out["content"] == [{"type": "text", "text": "Hi there!"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_non_streaming_response_tool_calls():
    openai_resp = {
        "choices": [
            {
                "message": {
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
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    out = openai_response_to_anthropic(openai_resp, model="m")
    assert out["stop_reason"] == "tool_use"
    assert out["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"location": "Paris"}}
    ]


async def _achunks(chunks):
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_streaming_text_translation():
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}},
    ]
    events = [e async for e in translate_openai_stream_to_anthropic(_achunks(chunks), model="m")]
    joined = "".join(events)

    assert "message_start" in joined
    assert '"type": "text_delta", "text": "Hel"' in joined
    assert '"type": "text_delta", "text": "lo"' in joined
    assert "content_block_stop" in joined
    assert '"stop_reason": "end_turn"' in joined
    assert "message_stop" in joined


@pytest.mark.asyncio
async def test_streaming_tool_call_translation():
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"location":'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    events = [e async for e in translate_openai_stream_to_anthropic(_achunks(chunks), model="m")]
    joined = "".join(events)

    assert '"type": "tool_use"' in joined
    assert '"name": "get_weather"' in joined
    assert '"partial_json": "{\\"location\\":"' in joined
    assert '"stop_reason": "tool_use"' in joined
