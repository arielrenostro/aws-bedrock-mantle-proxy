import json
import logging

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
logger = logging.getLogger(__name__)


@router.post("/v1/messages")
async def create_message(request: Request):
    body = await request.json()
    model = body["model"]
    openai_payload = anthropic_request_to_openai(body)
    url = f"{settings.mantle_base_url}/chat/completions"

    if openai_payload.get("stream"):

        async def event_stream():
            token = await get_bedrock_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                # Avoid any compressed transfer-encoding wrinkles while we
                # read the stream line-by-line and re-emit translated events.
                "Accept-Encoding": "identity",
            }
            try:
                async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
                    async with client.stream(
                        "POST", url, json=openai_payload, headers=headers
                    ) as resp:
                        logger.info(
                            "Mantle chat/completions stream status=%s headers=%s",
                            resp.status_code,
                            dict(resp.headers),
                        )
                        if resp.status_code >= 400:
                            error_body = await resp.aread()
                            logger.error(
                                "Mantle returned an error for a streaming request: %s %s",
                                resp.status_code,
                                error_body[:2000],
                            )
                            payload = {
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": error_body.decode(errors="replace"),
                                },
                            }
                            yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                            return
                        event_count = 0
                        async for event in translate_openai_stream_to_anthropic(
                            iter_openai_sse_json(resp), model
                        ):
                            event_count += 1
                            yield event
                        logger.info("Mantle stream translated into %d event(s)", event_count)
            except httpx.HTTPError as exc:
                logger.exception("Mantle stream request failed: %s", exc)
                payload = {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"proxy: upstream stream failed: {exc}",
                    },
                }
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"

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
