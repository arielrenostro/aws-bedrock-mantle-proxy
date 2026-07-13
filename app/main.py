import logging

from fastapi import FastAPI

from .routers import anthropic_router, openai_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="aws-bedrock-mantle-proxy")

app.include_router(anthropic_router.router)
app.include_router(openai_router.router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
