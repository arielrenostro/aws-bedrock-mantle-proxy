import logging

from fastapi import FastAPI

from .routers import anthropic_entry, openai_entry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="aws-bedrock-mantle-proxy")

app.include_router(anthropic_entry.router)
app.include_router(openai_entry.router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
