import json
from typing import Any


def _flatten_text_blocks(blocks: list) -> str:
    parts = []
    for b in blocks:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "".join(parts)


def _image_block_to_openai_part(block: dict) -> dict | None:
    source = block.get("source", {})
    if source.get("type") == "base64":
        data_url = f"data:{source.get('media_type')};base64,{source.get('data')}"
        return {"type": "image_url", "image_url": {"url": data_url}}
    if source.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url")}}
    return None


def convert_messages(anthropic_messages: list[dict], system: Any) -> list[dict]:
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
                part = _image_block_to_openai_part(block)
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


def convert_tools(anthropic_tools: list[dict] | None) -> list[dict] | None:
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


def convert_tool_choice(anthropic_tool_choice: dict | None):
    if not anthropic_tool_choice:
        return None
    t = anthropic_tool_choice.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        return {"type": "function", "function": {"name": anthropic_tool_choice["name"]}}
    return "auto"


def anthropic_request_to_openai(body: dict) -> dict:
    openai_body: dict = {
        "model": body["model"],
        "messages": convert_messages(body.get("messages", []), body.get("system")),
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

    tools = convert_tools(body.get("tools"))
    if tools:
        openai_body["tools"] = tools
        tool_choice = convert_tool_choice(body.get("tool_choice"))
        if tool_choice is not None:
            openai_body["tool_choice"] = tool_choice

    return openai_body
