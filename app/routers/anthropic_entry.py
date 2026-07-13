import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import orchestrator
from ..contracts import Contract
from ..logging_context import current_model

router = APIRouter(prefix="/anthropic")
logger = logging.getLogger(__name__)


@router.post("/v1/messages")
async def create_message(request: Request):
    body = await request.json()
    extra_headers = dict(request.headers)

    if body.get("stream"):
        # handle_stream sets/resets current_model itself, for the lifetime
        # of the generator (which runs after this function has returned).
        return StreamingResponse(
            orchestrator.handle_stream(Contract.ANTHROPIC, body, extra_headers),
            media_type="text/event-stream",
        )

    token = current_model.set(body.get("model") or "-")
    try:
        status, resp_body = await orchestrator.handle_request(Contract.ANTHROPIC, body, extra_headers)
    except httpx.HTTPError as exc:
        logger.exception("Mantle request failed: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": str(exc)}},
        )
    finally:
        current_model.reset(token)
    return JSONResponse(status_code=status, content=resp_body)
