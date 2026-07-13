"""Resolves which Mantle contract (Anthropic-native vs OpenAI-compatible) a
given model ID must be called through, and two more per-model quirks needed
to actually get a working request to Mantle:
- whether it's served on the standard `/v1/...` path or the `/openai/v1/...`
  prefixed path (some models, e.g. Gemma 4 and GPT-5.x, are only reachable
  on the latter; calling the wrong one returns a confusing permission-denied
  error rather than a clean 404);
- whether it supports parallel tool calls (Gemma 4 doesn't — requesting one
  anyway can fail generation server-side).

Contract resolution order:
1. Mantle's own `GET /v1/models` listing, best-effort — if a model entry
   ever carries an explicit field naming its supported API(s), use that.
   As of this writing Mantle's listing doesn't expose this, so in practice
   this step rarely resolves anything, which is exactly why step 2 exists.
2. `model_contracts.json` — an explicit, hand-maintained override file.
3. A prefix heuristic (`anthropic.*` -> Anthropic contract, everything else
   -> OpenAI contract) as a last-resort default so unknown models still get
   routed somewhere sensible instead of erroring out.

The `/openai/v1/...` path decision has no equivalent to step 1 or 3 above —
it can't be derived from Mantle's model catalog, and there's no reliable
naming heuristic — so it's resolved purely from `model_contracts.json`
(see `needs_openai_v1_prefix`), defaulting to the standard path when a
model isn't listed there.
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


def _load_overrides() -> dict[str, str | dict]:
    path = _overrides_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read %s; ignoring contract overrides", path, exc_info=True)
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _normalize_override(value: str | dict) -> tuple[Contract | None, bool, bool]:
    """Parse one model_contracts.json entry into (contract, openai_path_prefix,
    disable_parallel_tool_calls).

    Accepts the legacy bare-string form ("anthropic"/"openai") or the
    extended object form ({"contract": ..., "openai_path_prefix": ...,
    "disable_parallel_tool_calls": ...}). Returns (None, False, False) for
    anything malformed so callers fall back to other resolution steps
    instead of crashing on a bad override.
    """
    if isinstance(value, str):
        try:
            return Contract(value), False, False
        except ValueError:
            return None, False, False
    if isinstance(value, dict):
        try:
            contract = Contract(value["contract"])
        except (KeyError, ValueError):
            contract = None
        # Strict `is True` so a stray non-boolean JSON value (a string, a
        # number) never gets silently coerced into "true".
        prefix = value.get("openai_path_prefix") is True
        disable_parallel = value.get("disable_parallel_tool_calls") is True
        return contract, prefix, disable_parallel
    return None, False, False


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
        contract, _, _ = _normalize_override(overrides[model_id])
        if contract is not None:
            return contract
        logger.warning(
            "Override for model '%s' in %s has no valid 'contract'; falling back to the prefix heuristic.",
            model_id,
            _overrides_path(),
        )
        return _heuristic_contract(model_id)

    logger.warning(
        "Model '%s' isn't in Mantle's listing or in %s; falling back to the "
        "prefix heuristic. Add it to the overrides file for certainty.",
        model_id,
        _overrides_path(),
    )
    return _heuristic_contract(model_id)


def needs_openai_v1_prefix(model_id: str) -> bool:
    """Whether Mantle serves this model at '/openai/v1/...' instead of the
    standard '/v1/...' path, when called through the OpenAI contract.

    Purely static — unlike contract resolution, this can't be derived from
    Mantle's /v1/models listing, so only model_contracts.json is consulted.
    A model absent from the overrides file defaults to False (the standard
    path). Meaningless when the resolved target contract is Anthropic —
    callers should only check this for an OpenAI target.
    """
    overrides = _load_overrides()
    if model_id not in overrides:
        return False
    _, prefix, _ = _normalize_override(overrides[model_id])
    return prefix


def disable_parallel_tool_calls(model_id: str) -> bool:
    """Whether the request translated for this model should force
    `parallel_tool_calls: false`.

    Anthropic models support parallel tool use and Claude Code relies on it
    by default, so a request translated from the Anthropic contract has no
    signal against it otherwise. Some OpenAI-contract models don't support
    multiple tool calls in one turn at all (confirmed for the Gemma 4
    family per its Bedrock model card: "Requesting more than one tool call
    in a single turn is not currently supported") — asking anyway can fail
    the whole generation server-side rather than just returning calls one
    at a time. Purely static, like needs_openai_v1_prefix; a model absent
    from the overrides file defaults to False (no restriction). Meaningless
    when the resolved target contract is Anthropic.
    """
    overrides = _load_overrides()
    if model_id not in overrides:
        return False
    _, _, disable_parallel = _normalize_override(overrides[model_id])
    return disable_parallel
