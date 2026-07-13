import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import mantle_client, orchestrator
from ..contracts import Contract
from ..logging_context import current_model

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
        # handle_stream sets/resets current_model itself, for the lifetime
        # of the generator (which runs after this function has returned).
        return StreamingResponse(
            orchestrator.handle_stream(Contract.OPENAI, body, extra_headers),
            media_type="text/event-stream",
        )

    token = current_model.set(body.get("model") or "-")
    try:
        status, resp_body = await orchestrator.handle_request(Contract.OPENAI, body, extra_headers)
    except httpx.HTTPError as exc:
        logger.exception("Mantle request failed: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "api_error"}},
        )
    finally:
        current_model.reset(token)
    return JSONResponse(status_code=status, content=resp_body)
