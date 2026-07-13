# aws-bedrock-mantle-proxy

Reverse proxy that exposes an **Anthropic**-compatible API (`/anthropic/v1/messages`) and an **OpenAI**-compatible API (`/openai/v1/chat/completions`, `/openai/v1/models`), forwarding requests to **Amazon Bedrock Mantle** behind the scenes.

This lets you plug tools like **Claude Code**, **QwenCode**, or **Pi Harness** into this local proxy — they still "think" they're talking to Anthropic or OpenAI, but requests are authenticated and forwarded to Mantle.

**You can freely mix entry format and target model.** Mantle serves each model through exactly one of two contracts — its native Anthropic Messages API or its OpenAI-compatible Chat Completions API — but the proxy doesn't force you to speak whichever contract a given model happens to require. Point Claude Code at a Qwen model, or an OpenAI-SDK tool at a Claude model, and the proxy translates transparently in whichever direction is needed.

## Why this exists

Mantle doesn't allow issuing a static API key due to security requirements. Access must go through the AWS SDK, generating a **short-lived bearer token** on every call with the [`aws-bedrock-token-generator`](https://pypi.org/project/aws-bedrock-token-generator/) library, derived from AWS credentials already configured on the machine (profile, SSO, assumed role, environment variables, etc.).

Mantle natively exposes **two separate contracts** at the same host (`https://bedrock-mantle.{region}.api.aws`), and a given model is only reachable through one of them:

| Contract | Mantle path | Model families | Auth header |
|---|---|---|---|
| OpenAI-compatible | `/v1/chat/completions`, `/v1/models` | GPT and open-weight models (Qwen, Llama, etc.) | `Authorization: Bearer <token>` |
| Native Anthropic Messages API | `/anthropic/v1/messages` | Claude models | `x-api-key: <token>` |

## Architecture (onion-style layers)

```
app/
  contracts.py                    # Contract enum (ANTHROPIC | OPENAI) — the shared vocabulary
  config.py                       # environment variables / configuration
  auth.py                         # mints a fresh Mantle token on every request
  main.py                         # FastAPI app, registers the routers, configures logging

  routers/                        # ── outermost layer: HTTP adapters ──
    anthropic_entry.py            # POST /anthropic/v1/messages
    openai_entry.py                # GET /openai/v1/models, POST /openai/v1/chat/completions

  orchestrator.py                 # ── application layer ──
                                   # resolves the target contract, decides whether to
                                   # translate, calls Mantle, translates the response back

  translation/converters.py       # ── domain layer, pure functions, no I/O ──
                                   # bidirectional request/response/stream/error translation
                                   # between the Anthropic and OpenAI wire formats

  mantle_client.py                # ── infrastructure layer ──
                                   # the only module that speaks HTTP to Mantle: URLs,
                                   # per-contract auth headers, streaming, decompression

  model_registry.py               # resolves model_id -> Contract, and -> openai_path_prefix (see below)
  model_contracts.json            # explicit model_id -> contract (+ optional openai_path_prefix) overrides

tests/
  test_converters.py              # translation layer, no network/AWS
  test_model_registry.py          # contract resolution, no network/AWS
  test_orchestrator_stream.py     # streaming routing (raw relay vs. translated), no network/AWS
  test_routing.py                 # HTTP layer end-to-end, mantle_client monkeypatched

main.py                           # entrypoint (uvicorn)
```

Dependencies only point inward: routers depend on the orchestrator; the orchestrator depends on `mantle_client`, `model_registry`, and `translation.converters`; those depend only on `auth`/`config`/`contracts`. `translation/converters.py` has zero I/O, so every translation rule is unit-tested directly with plain dicts.

### Request flow

1. **Router** (`/anthropic/...` or `/openai/...`) parses the incoming body and hands it to the **orchestrator** along with which contract the client is speaking (the *entry* contract).
2. **Orchestrator** asks `model_registry.resolve_contract(model_id)` which contract Mantle actually needs for that model (the *target* contract).
3. If entry == target, the body is forwarded unmodified (no translation risk, no overhead).
4. If they differ, the **translation layer** converts the request into the target's wire format.
5. **`mantle_client`** calls Mantle with the correct URL/auth header for the target contract (streaming or not).
6. If entry == target, the response (or SSE stream) is relayed back as-is. If they differ, the **translation layer** converts the response (or SSE stream) back into the entry contract's shape.

### How the target contract is resolved

`model_registry.resolve_contract(model_id)`, in order:

1. **Mantle's own `GET /v1/models` listing** (best-effort, cached for 5 minutes) — if a model entry ever carries an explicit field naming its supported API, that wins. In practice Mantle's listing doesn't expose this today, which is exactly why step 2 exists.
2. **`app/model_contracts.json`** — an explicit, hand-maintained mapping of `model_id` to either a contract string (`"anthropic"` or `"openai"`) or, for OpenAI-contract models that need the path prefix described below, an object `{"contract": "openai", "openai_path_prefix": true}`. This is the place to correct or pin any model the automatic resolution gets wrong. Override the file location with `MODEL_CONTRACTS_FILE`.
3. **Prefix heuristic** as a last resort: `anthropic.*` model IDs → Anthropic contract, everything else → OpenAI contract (this matches how Bedrock actually names Claude models today). A warning is logged whenever this fallback is used, so you know which models are worth adding to the JSON file.

#### The `/openai/v1/...` path prefix

Within the OpenAI-compatible contract itself, Mantle actually has **two** URL paths, and a given model is only reachable on one of them:

- `/v1/chat/completions` — the standard path, used by default.
- `/openai/v1/chat/completions` — a second path that some models (currently `openai.gpt-5.5`, `openai.gpt-5.4`, `google.gemma-4-31b`, `google.gemma-4-e2b`, `google.gemma-4-26b-a4b`, `xai.grok-4.3`) are served on **exclusively**.

Calling the wrong path doesn't 404 — it returns a confusing `access_denied` / `permission_denied_error` response instead, which is easy to mistake for a real permissions problem. Which path a model needs can't be derived from the model catalog either, so — exactly like contract resolution — it's tracked purely via `model_contracts.json`:

```json
"google.gemma-4-31b": {"contract": "openai", "openai_path_prefix": true}
```

A model absent from the overrides file defaults to the standard (no-prefix) path. `model_registry.needs_openai_v1_prefix(model_id)` is only consulted when the resolved target contract is OpenAI — it's meaningless (and ignored) for the Anthropic contract. Add new model IDs here as AWS/Mantle exposes them.

## Prerequisites

- Python 3.11+
- AWS credentials configured locally with access to Bedrock Mantle (profile in `~/.aws/credentials`, SSO, environment variables, or an assumed role) — resolution uses boto3's default credential chain.

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate # Linux/macOS

pip install -r requirements.txt
```

## Configuration

Copy the example file and adjust as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Region where Bedrock Mantle is available |
| `MANTLE_BASE_URL` | `https://bedrock-mantle.{AWS_REGION}.api.aws` | Override for the Mantle host root (the proxy appends `/v1/...` or `/anthropic/v1/...` itself) |
| `BEDROCK_TOKEN_TTL_SECONDS` | `3600` | Requested lifetime for each generated token (max 12h / 43200s) |
| `MANTLE_REQUEST_TIMEOUT_SECONDS` | `300` | Timeout for requests forwarded to Mantle |
| `PROXY_HOST` | `0.0.0.0` | Bind address for the local server |
| `PROXY_PORT` | `8000` | Port for the local server |
| `MODEL_CONTRACTS_FILE` | `app/model_contracts.json` | Path to the model_id -> contract override file |

## Running

```bash
python main.py
```

The server starts at `http://localhost:8000` (or the configured port). Check it with:

```bash
curl http://localhost:8000/healthz
```

The console logs, for every request: the resolved entry/target contract, the status code and headers Mantle returned for streaming requests, and how many chunks/events were relayed or translated — useful for diagnosing upstream or routing issues.

Every log line — including the ones `httpx` itself emits for each request it sends to Mantle (`HTTP Request: POST ... "HTTP/1.1 200 OK"`) — is tagged with `[model=...]` for the model that request was made for. This works via a `contextvars`-backed `logging.Filter` (`app/logging_context.py`) attached to the root log handler, so it applies to any logger, not just the app's own, without needing to touch `httpx`'s internals:

```
2026-07-13 17:31:06 INFO httpx [model=google.gemma-4-31b]: HTTP Request: POST https://bedrock-mantle.../openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-13 17:31:06 INFO app.orchestrator [model=google.gemma-4-31b]: Routing entry=openai target=openai openai_path_prefix=True
```

## Connecting your tools

### Claude Code (Anthropic)

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000/anthropic
export ANTHROPIC_API_KEY=any-value   # ignored by the proxy; real auth happens against Mantle
```

Works with any model Mantle serves, Claude or not — e.g. point it at `qwen.qwen3-coder-30b-a3b-instruct` and the proxy translates the Anthropic-shaped request into OpenAI shape before calling Mantle.

### OpenAI-compatible tools (QwenCode, Pi Harness, OpenAI SDK)

```bash
export OPENAI_BASE_URL=http://localhost:8000/openai/v1
export OPENAI_API_KEY=any-value      # ignored by the proxy; real auth happens against Mantle
```

Same in reverse — point it at `anthropic.claude-sonnet-4-6-v1` and the proxy translates into the native Anthropic contract before calling Mantle, then translates the response back into an OpenAI-shaped `chat.completion`.

List the models available on Bedrock:

```bash
curl http://localhost:8000/openai/v1/models
```

## Tests

All tests are unit/integration tests with `mantle_client` monkeypatched — no real network calls or AWS credentials required:

```bash
pytest tests/ -v
```

## Known limitations

- `/v1/messages/count_tokens` (Anthropic) is not implemented.
- Mantle's `/v1/models` listing doesn't currently expose which contract a model supports, so contract resolution normally falls through to `model_contracts.json` / the prefix heuristic — keep the JSON file up to date for any model that doesn't follow the `anthropic.*` naming convention. Likewise, whether an OpenAI-contract model needs the `/openai/v1/...` path prefix instead of the standard `/v1/...` path can't be derived from the catalog either — it's tracked via `openai_path_prefix` in the same file and must be updated by hand as AWS adds more such models.
- Translated requests/responses cover text, images, tool use, and streaming; less common fields (e.g. `logprobs`, `n`, prompt caching hints) are not translated and are dropped when crossing contracts. Same-contract requests are always forwarded byte-for-byte, so nothing is ever lost when the client already speaks the target model's native contract.
- Mantle's native Anthropic Messages API endpoint rejects several request fields recent Claude Code / Anthropic SDK versions send by default, each with `400 <field>: Extra inputs are not permitted` — currently `output_config.format` (structured outputs) and `context_management` (compaction / context-editing). `mantle_client._strip_unsupported_anthropic_fields` drops just those fields before forwarding to the Anthropic contract (logging a warning each time), leaving everything else — e.g. `output_config.effort` — untouched. This list is expected to grow as Claude Code adopts newer Anthropic API surface faster than Mantle's Anthropic-compatible endpoint catches up; add new entries to `_UNSUPPORTED_TOP_LEVEL_FIELDS` / `_UNSUPPORTED_NESTED_FIELDS` in `app/mantle_client.py` as they're found.
- The proxy never forwards the client's `anthropic-beta` header to Mantle's Anthropic contract — Mantle rejects requests outright with `400 invalid beta flag` for beta values Claude Code sends by default (e.g. `context-management-2025-06-27`), since it doesn't recognize the same beta surface as the real Anthropic API. This is deliberate, not an oversight: since the request fields those betas would gate are already stripped (previous bullet), forwarding the header would only trade one hard failure for another.
