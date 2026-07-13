import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import orchestrator
from ..contracts import Contract

router = APIRouter(prefix="/anthropic")
logger = logging.getLogger(__name__)


@router.post("/v1/messages")
async def create_message(request: Request):
    body = await request.json()
    extra_headers = dict(request.headers)

    if body.get("stream"):
        return StreamingResponse(
            orchestrator.handle_stream(Contract.ANTHROPIC, body, extra_headers),
            media_type="text/event-stream",
        )

    try:
        status, resp_body = await orchestrator.handle_request(Contract.ANTHROPIC, body, extra_headers)
    except httpx.HTTPError as exc:
        logger.exception("Mantle request failed: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": str(exc)}},
        )
    return JSONResponse(status_code=status, content=resp_body)
