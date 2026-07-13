import json
import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from ..auth import get_bedrock_token
from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/v1/models")
async def list_models():
    token = await get_bedrock_token()
    url = f"{settings.mantle_base_url}/models"
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.get(url, headers=headers)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    url = f"{settings.mantle_base_url}/chat/completions"

    if body.get("stream"):

        async def event_stream():
            token = await get_bedrock_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                # Force an uncompressed response so no decoding step (gzip/
                # deflate/br) can ever desync mid-stream while we relay bytes.
                "Accept-Encoding": "identity",
            }
            try:
                async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
                    async with client.stream("POST", url, json=body, headers=headers) as resp:
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
                            yield error_body
                            return

                        chunk_count = 0
                        async for chunk in resp.aiter_bytes():
                            chunk_count += 1
                            yield chunk
                        logger.info("Mantle stream ended normally after %d chunk(s)", chunk_count)
            except httpx.HTTPError as exc:
                logger.exception("Mantle stream request failed: %s", exc)
                error_payload = {"error": {"message": f"proxy: upstream stream failed: {exc}"}}
                yield f"data: {json.dumps(error_payload)}\n\n".encode()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    token = await get_bedrock_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.post(url, json=body, headers=headers)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )
