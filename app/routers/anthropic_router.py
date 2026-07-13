import json

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import get_bedrock_token
from ..config import settings
from ..translation.anthropic_to_openai import anthropic_request_to_openai
from ..translation.openai_to_anthropic import (
    iter_openai_sse_json,
    openai_response_to_anthropic,
    translate_openai_stream_to_anthropic,
)

router = APIRouter()


@router.post("/v1/messages")
async def create_message(request: Request):
    body = await request.json()
    model = body["model"]
    openai_payload = anthropic_request_to_openai(body)
    url = f"{settings.mantle_base_url}/chat/completions"

    if openai_payload.get("stream"):

        async def event_stream():
            token = await get_bedrock_token()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
                async with client.stream("POST", url, json=openai_payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        error_body = await resp.aread()
                        payload = {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": error_body.decode(errors="replace"),
                            },
                        }
                        yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                        return
                    async for event in translate_openai_stream_to_anthropic(
                        iter_openai_sse_json(resp), model
                    ):
                        yield event

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    token = await get_bedrock_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.post(url, json=openai_payload, headers=headers)

    if resp.status_code >= 400:
        return JSONResponse(
            status_code=resp.status_code,
            content={"type": "error", "error": {"type": "api_error", "message": resp.text}},
        )

    return JSONResponse(content=openai_response_to_anthropic(resp.json(), model))
