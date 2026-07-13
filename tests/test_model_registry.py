import json

import pytest

import app.model_registry as model_registry
from app.contracts import Contract


@pytest.fixture(autouse=True)
def _reset_cache():
    model_registry._model_list_cache["data"] = None
    model_registry._model_list_cache["fetched_at"] = 0.0
    yield
    model_registry._model_list_cache["data"] = None
    model_registry._model_list_cache["fetched_at"] = 0.0


@pytest.mark.asyncio
async def test_resolves_from_mantle_models_listing_when_it_names_the_contract(monkeypatch):
    async def fake_list_models():
        return 200, {"data": [{"id": "some-model", "api": "anthropic"}]}, {}

    monkeypatch.setattr(model_registry.mantle_client, "list_models", fake_list_models)
    contract = await model_registry.resolve_contract("some-model")
    assert contract == Contract.ANTHROPIC


@pytest.mark.asyncio
async def test_falls_back_to_json_overrides_when_listing_is_uninformative(monkeypatch, tmp_path):
    async def fake_list_models():
        return 200, {"data": [{"id": "custom.my-model"}]}, {}  # no contract-identifying field

    monkeypatch.setattr(model_registry.mantle_client, "list_models", fake_list_models)

    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_text(json.dumps({"custom.my-model": "anthropic"}))
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)

    contract = await model_registry.resolve_contract("custom.my-model")
    assert contract == Contract.ANTHROPIC


@pytest.mark.asyncio
async def test_falls_back_to_prefix_heuristic_when_nothing_else_resolves(monkeypatch, tmp_path):
    async def fake_list_models():
        return 200, {"data": []}, {}

    monkeypatch.setattr(model_registry.mantle_client, "list_models", fake_list_models)
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: tmp_path / "does-not-exist.json")

    assert await model_registry.resolve_contract("anthropic.claude-sonnet-4-6-v1") == Contract.ANTHROPIC
    assert await model_registry.resolve_contract("qwen.qwen3-coder-30b-a3b-instruct") == Contract.OPENAI


@pytest.mark.asyncio
async def test_json_override_takes_precedence_over_heuristic(monkeypatch, tmp_path):
    async def fake_list_models():
        return 200, {"data": []}, {}

    monkeypatch.setattr(model_registry.mantle_client, "list_models", fake_list_models)

    overrides_file = tmp_path / "overrides.json"
    # Deliberately contradicts the "anthropic." prefix heuristic.
    overrides_file.write_text(json.dumps({"anthropic.some-quirky-model": "openai"}))
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)

    contract = await model_registry.resolve_contract("anthropic.some-quirky-model")
    assert contract == Contract.OPENAI


@pytest.mark.asyncio
async def test_mantle_listing_failure_does_not_crash_resolution(monkeypatch, tmp_path):
    import httpx

    async def fake_list_models():
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(model_registry.mantle_client, "list_models", fake_list_models)
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: tmp_path / "does-not-exist.json")

    contract = await model_registry.resolve_contract("anthropic.claude-sonnet-4-6-v1")
    assert contract == Contract.ANTHROPIC


# ---------------------------------------------------------------------------
# needs_openai_v1_prefix — the /openai/v1/... path variant some OpenAI-
# contract models (Gemma 4, GPT-5.x, ...) are exclusively served on.
# ---------------------------------------------------------------------------


def test_needs_openai_v1_prefix_true_for_flagged_override(monkeypatch, tmp_path):
    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_text(
        json.dumps({"google.gemma-4-31b": {"contract": "openai", "openai_path_prefix": True}})
    )
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)
    assert model_registry.needs_openai_v1_prefix("google.gemma-4-31b") is True


def test_needs_openai_v1_prefix_false_for_bare_string_override(monkeypatch, tmp_path):
    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_text(json.dumps({"anthropic.claude-sonnet-4-6-v1": "anthropic"}))
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)
    assert model_registry.needs_openai_v1_prefix("anthropic.claude-sonnet-4-6-v1") is False


def test_needs_openai_v1_prefix_false_when_model_not_in_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: tmp_path / "does-not-exist.json")
    assert model_registry.needs_openai_v1_prefix("qwen.qwen3-coder-30b-a3b-instruct") is False


def test_needs_openai_v1_prefix_false_when_prefix_key_absent(monkeypatch, tmp_path):
    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_text(json.dumps({"custom.model": {"contract": "openai"}}))
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)
    assert model_registry.needs_openai_v1_prefix("custom.model") is False


@pytest.mark.asyncio
async def test_resolve_contract_works_with_object_form_override(monkeypatch, tmp_path):
    async def fake_list_models():
        return 200, {"data": []}, {}

    monkeypatch.setattr(model_registry.mantle_client, "list_models", fake_list_models)

    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_text(
        json.dumps({"google.gemma-4-31b": {"contract": "openai", "openai_path_prefix": True}})
    )
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)

    contract = await model_registry.resolve_contract("google.gemma-4-31b")
    assert contract == Contract.OPENAI


@pytest.mark.asyncio
async def test_resolve_contract_falls_back_to_heuristic_on_malformed_object_override(monkeypatch, tmp_path):
    async def fake_list_models():
        return 200, {"data": []}, {}

    monkeypatch.setattr(model_registry.mantle_client, "list_models", fake_list_models)

    overrides_file = tmp_path / "overrides.json"
    # Missing the required "contract" key.
    overrides_file.write_text(json.dumps({"custom.weird-model": {"openai_path_prefix": True}}))
    monkeypatch.setattr(model_registry, "_overrides_path", lambda: overrides_file)

    contract = await model_registry.resolve_contract("custom.weird-model")
    assert contract == Contract.OPENAI  # heuristic default, no "anthropic." prefix


def test_real_shipped_overrides_file_flags_gemma_4():
    """Cheap regression guard against a typo in the actual seed file — the
    only test that exercises app/model_contracts.json as shipped, without
    monkeypatching _overrides_path."""
    assert model_registry.needs_openai_v1_prefix("google.gemma-4-31b") is True
