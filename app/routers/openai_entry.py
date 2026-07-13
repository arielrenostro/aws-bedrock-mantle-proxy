import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import mantle_client, orchestrator
from ..contracts import Contract

router = APIRouter(prefix="/openai")
logger = logging.getLogger(__name__)


@router.get("/v1/models")
async def list_models():
    status, body, _ = await mantle_client.list_models()
    return JSONResponse(status_code=status, content=body)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    extra_headers = dict(request.headers)

    if body.get("stream"):
        return StreamingResponse(
            orchestrator.handle_stream(Contract.OPENAI, body, extra_headers),
            media_type="text/event-stream",
        )

    try:
        status, resp_body = await orchestrator.handle_request(Contract.OPENAI, body, extra_headers)
    except httpx.HTTPError as exc:
        logger.exception("Mantle request failed: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "api_error"}},
        )
    return JSONResponse(status_code=status, content=resp_body)
