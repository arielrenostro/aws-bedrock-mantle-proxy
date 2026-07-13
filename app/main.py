import logging

from fastapi import FastAPI

from .logging_context import ModelLogFilter
from .routers import anthropic_entry, openai_entry


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s [model=%(model)s]: %(message)s")
    )
    # Filters on a Handler run for every record that reaches it, no matter
    # which logger emitted it — this is what puts the model into httpx's own
    # "HTTP Request: ..." lines, not just our own app.* log calls.
    handler.addFilter(ModelLogFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])


_configure_logging()

app = FastAPI(title="aws-bedrock-mantle-proxy")

app.include_router(anthropic_entry.router)
app.include_router(openai_entry.router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
