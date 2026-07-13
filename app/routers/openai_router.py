import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from ..auth import get_bedrock_token
from ..config import settings

router = APIRouter()


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
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    async for chunk in resp.aiter_raw():
                        yield chunk

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
