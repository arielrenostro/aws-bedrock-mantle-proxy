"""Pure, I/O-free translation between the Anthropic Messages API wire format
and the OpenAI Chat Completions wire format.

Every function here takes plain dicts (or an async iterator of already
-parsed SSE event dicts) and returns plain dicts / strings. Nothing in this
module talks to the network — that keeps it trivially unit-testable and
keeps the "domain" layer of the proxy free of infrastructure concerns.
"""

import json
import re
import time
import uuid
from typing import AsyncIterator, Any

from ..contracts import Contract

DEFAULT_MAX_TOKENS = 4096

_DATA_URL_RE = re.compile(r"^data:(?P<media_type>[^;]+);base64,(?P<data>.+)$", re.DOTALL)

FINISH_REASON_TO_ANTHROPIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
    None: "end_turn",
}

STOP_REASON_TO_OPENAI = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
    "refusal": "content_filter",
    None: "stop",
}


# ---------------------------------------------------------------------------
# Anthropic request -> OpenAI request
# ---------------------------------------------------------------------------


def _flatten_text_blocks(blocks: list) -> str:
    parts = []
    for b in blocks:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "".join(parts)


def _anthropic_image_block_to_openai_part(block: dict) -> dict | None:
    source = block.get("source", {})
    if source.get("type") == "base64":
        data_url = f"data:{source.get('media_type')};base64,{source.get('data')}"
        return {"type": "image_url", "image_url": {"url": data_url}}
    if source.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url")}}
    return None


def _anthropic_messages_to_openai(anthropic_messages: list[dict], system: Any) -> list[dict]:
    openai_messages: list[dict] = []

    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            openai_messages.append({"role": "system", "content": _flatten_text_blocks(system)})

    for msg in anthropic_messages:
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        image_parts: list[dict] = []

        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "image":
                part = _anthropic_image_block_to_openai_part(block)
                if part:
                    image_parts.append(part)
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif btype == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_text = _flatten_text_blocks(result_content)
                else:
                    result_text = str(result_content)
                if block.get("is_error"):
                    result_text = f"Error: {result_text}"
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": result_text,
                    }
                )

        if tool_results:
            # Anthropic carries tool_result blocks inside a "user" message;
            # OpenAI expects one "tool" message per result.
            openai_messages.extend(tool_results)
            if text_parts:
                openai_messages.append({"role": "user", "content": "".join(text_parts)})
            continue

        if role == "assistant":
            msg_out: dict = {"role": "assistant"}
            if text_parts:
                msg_out["content"] = "".join(text_parts)
            if tool_calls:
                msg_out["tool_calls"] = tool_calls
            if "content" not in msg_out and "tool_calls" not in msg_out:
                msg_out["content"] = ""
            openai_messages.append(msg_out)
        elif image_parts:
            parts: list[dict] = []
            if text_parts:
                parts.append({"type": "text", "text": "".join(text_parts)})
            parts.extend(image_parts)
            openai_messages.append({"role": "user", "content": parts})
        else:
            openai_messages.append({"role": role, "content": "".join(text_parts)})

    return openai_messages


def _anthropic_tools_to_openai(anthropic_tools: list[dict] | None) -> list[dict] | None:
    if not anthropic_tools:
        return None
    out = []
    for t in anthropic_tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return out


def _anthropic_tool_choice_to_openai(tool_choice: dict | None):
    if not tool_choice:
        return None
    t = tool_choice.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return "auto"


def anthropic_request_to_openai(body: dict) -> dict:
    openai_body: dict = {
        "model": body["model"],
        "messages": _anthropic_messages_to_openai(body.get("messages", []), body.get("system")),
        "stream": bool(body.get("stream", False)),
    }
    if "max_tokens" in body:
        openai_body["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        openai_body["temperature"] = body["temperature"]
    if "top_p" in body:
        openai_body["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        openai_body["stop"] = body["stop_sequences"]

    tools = _anthropic_tools_to_openai(body.get("tools"))
    if tools:
        openai_body["tools"] = tools
        tool_choice = _anthropic_tool_choice_to_openai(body.get("tool_choice"))
        if tool_choice is not None:
            openai_body["tool_choice"] = tool_choice

    return openai_body


# ---------------------------------------------------------------------------
# OpenAI request -> Anthropic request
# ---------------------------------------------------------------------------


def _flatten_openai_text_parts(parts: list) -> str:
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict) and p.get("type") == "text":
            out.append(p.get("text", ""))
    return "".join(out)


def _openai_image_part_to_anthropic(part: dict) -> dict:
    url = part.get("image_url", {}).get("url", "")
    m = _DATA_URL_RE.match(url)
    if m:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": m.group("media_type"),
                "data": m.group("data"),
            },
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _openai_messages_to_anthropic(openai_messages: list[dict]) -> tuple[list[str], list[dict]]:
    system_parts: list[str] = []
    anthropic_messages: list[dict] = []

    for msg in openai_messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.append(_flatten_openai_text_parts(content))
            continue

        if role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content if isinstance(content, str) else json.dumps(content),
                        }
                    ],
                }
            )
            continue

        if role == "assistant":
            blocks: list[dict] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                text = _flatten_openai_text_parts(content)
                if text:
                    blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                try:
                    tool_input = json.loads(func.get("arguments") or "{}")
                except json.JSONDecodeError:
                    tool_input = {"_raw_arguments": func.get("arguments")}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex}"),
                        "name": func.get("name", ""),
                        "input": tool_input,
                    }
                )
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            anthropic_messages.append({"role": "assistant", "content": blocks})
            continue

        # role == "user" (or unrecognized, treated as user)
        if isinstance(content, str):
            anthropic_messages.append({"role": "user", "content": content})
        elif isinstance(content, list):
            blocks = []
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    blocks.append(_openai_image_part_to_anthropic(part))
            anthropic_messages.append({"role": "user", "content": blocks})
        else:
            anthropic_messages.append({"role": "user", "content": ""})

    return system_parts, anthropic_messages


def _openai_tools_to_anthropic(openai_tools: list[dict] | None) -> list[dict] | None:
    if not openai_tools:
        return None
    out = []
    for t in openai_tools:
        func = t.get("function", {})
        out.append(
            {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


def _openai_tool_choice_to_anthropic(tool_choice):
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "tool", "name": tool_choice.get("function", {}).get("name", "")}
    return {"type": "auto"}


def openai_request_to_anthropic(body: dict) -> dict:
    system_parts, anthropic_messages = _openai_messages_to_anthropic(body.get("messages", []))

    anthropic_body: dict = {
        "model": body["model"],
        "max_tokens": body.get("max_tokens") or DEFAULT_MAX_TOKENS,
        "messages": anthropic_messages,
        "stream": bool(body.get("stream", False)),
    }
    if system_parts:
        anthropic_body["system"] = "\n\n".join(system_parts)
    if "temperature" in body:
        anthropic_body["temperature"] = body["temperature"]
    if "top_p" in body:
        anthropic_body["top_p"] = body["top_p"]
    stop = body.get("stop")
    if stop:
        anthropic_body["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    tools = _openai_tools_to_anthropic(body.get("tools"))
    if tools:
        anthropic_body["tools"] = tools
        tool_choice = _openai_tool_choice_to_anthropic(body.get("tool_choice"))
        if tool_choice is not None:
            anthropic_body["tool_choice"] = tool_choice

    return anthropic_body


# ---------------------------------------------------------------------------
# OpenAI response -> Anthropic response (non-streaming)
# ---------------------------------------------------------------------------


def openai_response_to_anthropic(openai_resp: dict, model: str) -> dict:
    choice = openai_resp["choices"][0]
    message = choice.get("message", {})
    content_blocks: list[dict] = []

    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        func = tool_call.get("function", {})
        try:
            tool_input = json.loads(func.get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_input = {"_raw_arguments": func.get("arguments")}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex}",
                "name": func.get("name", ""),
                "input": tool_input,
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop_reason = FINISH_REASON_TO_ANTHROPIC.get(choice.get("finish_reason"), "end_turn")
    usage = openai_resp.get("usage") or {}

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Anthropic response -> OpenAI response (non-streaming)
# ---------------------------------------------------------------------------


def anthropic_response_to_openai(anthropic_resp: dict, model: str) -> dict:
    content_blocks = anthropic_resp.get("content", [])
    text_parts = []
    tool_calls = []

    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )

    text = "".join(text_parts)
    message: dict = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = STOP_REASON_TO_OPENAI.get(anthropic_resp.get("stop_reason"), "stop")
    usage = anthropic_resp.get("usage") or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Error bodies
# ---------------------------------------------------------------------------


def _extract_error_message(body: dict) -> str:
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("message", json.dumps(body)))
    if isinstance(err, str):
        return err
    return json.dumps(body)


def openai_error_to_anthropic(openai_error_body: dict) -> dict:
    return {
        "type": "error",
        "error": {"type": "api_error", "message": _extract_error_message(openai_error_body)},
    }


def anthropic_error_to_openai(anthropic_error_body: dict) -> dict:
    return {
        "error": {"message": _extract_error_message(anthropic_error_body), "type": "api_error"},
    }


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE events -> Anthropic SSE text
# ---------------------------------------------------------------------------


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class _AnthropicStreamState:
    def __init__(self) -> None:
        self.next_index = 0
        self.text_index: int | None = None
        self.text_open = False
        self.tool_index_map: dict[int, int] = {}
        self.tool_open: dict[int, bool] = {}
        self.finish_reason: str | None = None
        self.completion_tokens = 0
        self.prompt_tokens = 0


async def translate_openai_stream_to_anthropic(
    chunks: AsyncIterator[dict], model: str
) -> AsyncIterator[str]:
    """Consume parsed OpenAI chat.completion.chunk dicts and yield Anthropic
    Messages API SSE event strings (message_start .. message_stop)."""
    message_id = f"msg_{uuid.uuid4().hex}"
    state = _AnthropicStreamState()

    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    async for chunk in chunks:
        if chunk.get("usage"):
            u = chunk["usage"]
            state.prompt_tokens = u.get("prompt_tokens", state.prompt_tokens)
            state.completion_tokens = u.get("completion_tokens", state.completion_tokens)

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        if choice.get("finish_reason"):
            state.finish_reason = choice["finish_reason"]

        text_delta = delta.get("content")
        if text_delta:
            if state.text_index is None:
                state.text_index = state.next_index
                state.next_index += 1
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": state.text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                state.text_open = True
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": state.text_index,
                    "delta": {"type": "text_delta", "text": text_delta},
                },
            )

        for tc in delta.get("tool_calls") or []:
            openai_index = tc.get("index", 0)
            if openai_index not in state.tool_index_map:
                if state.text_open:
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": state.text_index},
                    )
                    state.text_open = False

                anth_index = state.next_index
                state.next_index += 1
                state.tool_index_map[openai_index] = anth_index
                state.tool_open[anth_index] = True
                func = tc.get("function", {})
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": anth_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex}",
                            "name": func.get("name", ""),
                            "input": {},
                        },
                    },
                )

            anth_index = state.tool_index_map[openai_index]
            func = tc.get("function", {})
            args_delta = func.get("arguments")
            if args_delta:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": anth_index,
                        "delta": {"type": "input_json_delta", "partial_json": args_delta},
                    },
                )

    if state.text_open:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": state.text_index})
    for anth_index in state.tool_index_map.values():
        if state.tool_open.get(anth_index):
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": anth_index})

    stop_reason = FINISH_REASON_TO_ANTHROPIC.get(state.finish_reason, "end_turn")
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": state.completion_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Streaming: Anthropic SSE events -> OpenAI SSE text
# ---------------------------------------------------------------------------


async def translate_anthropic_stream_to_openai(
    events: AsyncIterator[dict], model: str
) -> AsyncIterator[str]:
    """Consume parsed Anthropic Messages API SSE event dicts and yield
    OpenAI chat.completion.chunk SSE text, ending with `data: [DONE]`."""
    chat_id = f"chatcmpl_{uuid.uuid4().hex}"
    created = int(time.time())
    tool_openai_index: dict[int, int] = {}
    next_tool_index = 0
    stop_reason: str | None = None
    input_tokens = 0
    output_tokens = 0

    def _chunk(delta: dict, finish_reason: str | None = None) -> str:
        payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield _chunk({"role": "assistant", "content": ""})

    async for event in events:
        etype = event.get("type")

        if etype == "message_start":
            usage = event.get("message", {}).get("usage", {})
            input_tokens = usage.get("input_tokens", input_tokens)

        elif etype == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                anth_index = event.get("index", 0)
                openai_index = next_tool_index
                next_tool_index += 1
                tool_openai_index[anth_index] = openai_index
                yield _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": openai_index,
                                "id": block.get("id") or f"call_{uuid.uuid4().hex}",
                                "type": "function",
                                "function": {"name": block.get("name", ""), "arguments": ""},
                            }
                        ]
                    }
                )

        elif etype == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                yield _chunk({"content": delta.get("text", "")})
            elif delta.get("type") == "input_json_delta":
                anth_index = event.get("index", 0)
                openai_index = tool_openai_index.get(anth_index, 0)
                yield _chunk(
                    {
                        "tool_calls": [
                            {"index": openai_index, "function": {"arguments": delta.get("partial_json", "")}}
                        ]
                    }
                )

        elif etype == "message_delta":
            delta = event.get("delta", {})
            if delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]
            usage = event.get("usage") or {}
            if "output_tokens" in usage:
                output_tokens = usage["output_tokens"]

        elif etype == "message_stop":
            break

    finish_reason = STOP_REASON_TO_OPENAI.get(stop_reason, "stop")
    yield _chunk({}, finish_reason=finish_reason)

    usage_payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    yield f"data: {json.dumps(usage_payload)}\n\n"
    yield "data: [DONE]\n\n"


def format_error_sse(entry: Contract, raw_error_body: bytes) -> bytes:
    """Format a raw (upstream) error body as an SSE chunk in the entry
    contract's error shape."""
    try:
        parsed = json.loads(raw_error_body)
        if not isinstance(parsed, dict):
            parsed = {"message": str(parsed)}
    except (json.JSONDecodeError, UnicodeDecodeError):
        parsed = {"message": raw_error_body.decode(errors="replace")}

    message = _extract_error_message(parsed) if "error" in parsed else parsed.get("message", json.dumps(parsed))

    if entry == Contract.ANTHROPIC:
        payload = {"type": "error", "error": {"type": "api_error", "message": message}}
        return f"event: error\ndata: {json.dumps(payload)}\n\n".encode()

    payload = {"error": {"message": message, "type": "api_error"}}
    return f"data: {json.dumps(payload)}\n\n".encode()
