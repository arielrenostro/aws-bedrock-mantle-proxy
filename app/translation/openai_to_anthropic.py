import json
import uuid
from typing import AsyncIterator

FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
    None: "end_turn",
}


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


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

    stop_reason = FINISH_REASON_MAP.get(choice.get("finish_reason"), "end_turn")
    usage = openai_resp.get("usage") or {}

    return {
        "id": _new_message_id(),
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


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def iter_openai_sse_json(response) -> AsyncIterator[dict]:
    """Parse an httpx streaming response of `data: {...}` lines into dicts."""
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


class _StreamState:
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
    message_id = _new_message_id()
    state = _StreamState()

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

    stop_reason = FINISH_REASON_MAP.get(state.finish_reason, "end_turn")
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": state.completion_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})
