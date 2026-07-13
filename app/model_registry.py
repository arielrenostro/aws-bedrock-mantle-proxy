"""Resolves which Mantle contract (Anthropic-native vs OpenAI-compatible) a
given model ID must be called through.

Resolution order:
1. Mantle's own `GET /v1/models` listing, best-effort — if a model entry
   ever carries an explicit field naming its supported API(s), use that.
   As of this writing Mantle's listing doesn't expose this, so in practice
   this step rarely resolves anything, which is exactly why step 2 exists.
2. `model_contracts.json` — an explicit, hand-maintained override file.
3. A prefix heuristic (`anthropic.*` -> Anthropic contract, everything else
   -> OpenAI contract) as a last-resort default so unknown models still get
   routed somewhere sensible instead of erroring out.
"""

import json
import logging
import time
from pathlib import Path

import httpx

from . import mantle_client
from .config import settings
from .contracts import Contract

logger = logging.getLogger(__name__)

_MODEL_LIST_CACHE_TTL_SECONDS = 300
_model_list_cache: dict = {"data": None, "fetched_at": 0.0}


def _overrides_path() -> Path:
    return settings.model_contracts_path


def _load_overrides() -> dict[str, str]:
    path = _overrides_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read %s; ignoring contract overrides", path, exc_info=True)
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _heuristic_contract(model_id: str) -> Contract:
    # On Bedrock, Anthropic model IDs are always prefixed "anthropic." and
    # are only reachable via Mantle's native Anthropic Messages contract;
    # every other provider prefix (qwen., meta., amazon., openai., etc.) is
    # served through the OpenAI-compatible contract.
    if model_id.startswith("anthropic."):
        return Contract.ANTHROPIC
    return Contract.OPENAI


async def _fetch_model_list() -> list[dict]:
    now = time.monotonic()
    if _model_list_cache["data"] is not None and now - _model_list_cache["fetched_at"] < _MODEL_LIST_CACHE_TTL_SECONDS:
        return _model_list_cache["data"]
    try:
        status, body, _ = await mantle_client.list_models()
        if status == 200 and isinstance(body, dict):
            data = body.get("data", [])
            _model_list_cache["data"] = data
            _model_list_cache["fetched_at"] = now
            return data
    except httpx.HTTPError:
        logger.warning("Could not fetch model list from Mantle for contract resolution", exc_info=True)
    return _model_list_cache["data"] or []


def _contract_from_model_info(info: dict) -> Contract | None:
    for key in ("api", "contract", "protocol"):
        value = info.get(key)
        if value in ("anthropic", "openai"):
            return Contract(value)
    endpoints = info.get("endpoints") or info.get("supported_apis")
    if isinstance(endpoints, list):
        joined = " ".join(str(e).lower() for e in endpoints)
        if "anthropic" in joined:
            return Contract.ANTHROPIC
        if "chat/completions" in joined or "openai" in joined:
            return Contract.OPENAI
    return None


async def resolve_contract(model_id: str) -> Contract:
    for m in await _fetch_model_list():
        if m.get("id") == model_id:
            contract = _contract_from_model_info(m)
            if contract is not None:
                return contract
            break

    overrides = _load_overrides()
    if model_id in overrides:
        return Contract(overrides[model_id])

    logger.warning(
        "Model '%s' isn't in Mantle's listing or in %s; falling back to the "
        "prefix heuristic. Add it to the overrides file for certainty.",
        model_id,
        _overrides_path(),
    )
    return _heuristic_contract(model_id)
