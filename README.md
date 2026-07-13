# aws-bedrock-mantle-proxy

Reverse proxy that exposes an **Anthropic**-compatible API (`/v1/messages`) and an **OpenAI**-compatible API (`/v1/chat/completions`, `/v1/models`), forwarding requests to **Amazon Bedrock Mantle** behind the scenes.

This lets you plug tools like **Claude Code**, **QwenCode**, or **Pi Harness** into this local proxy — they still "think" they're talking to Anthropic or OpenAI, but requests are authenticated and forwarded to Mantle.

## Why this exists

Mantle doesn't allow issuing a static API key due to security requirements. Access must go through the AWS SDK, generating a **short-lived bearer token** on every call with the [`aws-bedrock-token-generator`](https://pypi.org/project/aws-bedrock-token-generator/) library, derived from AWS credentials already configured on the machine (profile, SSO, assumed role, environment variables, etc.).

Mantle already natively exposes an OpenAI-compatible API at `https://bedrock-mantle.{region}.api.aws/v1`. This proxy:

1. Forwards OpenAI-shaped calls almost directly (injecting the token on every request).
2. Translates Anthropic Messages API calls into OpenAI format before sending them to Mantle, and translates the response back — including streaming (SSE).

## Architecture

```
app/
  config.py                       # environment variables / configuration
  auth.py                         # mints a fresh Mantle token on every request
  main.py                         # FastAPI app, registers the routers
  routers/
    anthropic_router.py           # POST /v1/messages  (translates Anthropic <-> OpenAI)
    openai_router.py              # GET /v1/models, POST /v1/chat/completions (passthrough)
  translation/
    anthropic_to_openai.py        # request: Anthropic Messages -> OpenAI Chat Completions
    openai_to_anthropic.py        # response: OpenAI (JSON and streaming SSE) -> Anthropic Messages
tests/
  test_translation.py             # unit tests for the translation layer (no network/AWS)
main.py                           # entrypoint (uvicorn)
```

### Flow

- **OpenAI-compatible tools** (QwenCode, Pi Harness, generic OpenAI SDK) → `POST /v1/chat/completions` or `GET /v1/models` → proxy injects the Bearer token → forwards to Mantle without modifying the body.
- **Claude Code** (speaks the Anthropic format) → `POST /v1/messages` → proxy translates the request into OpenAI format, calls Mantle, translates the response (or stream) back into Anthropic format.

The model name (`model`) is passed through exactly as the client sends it — **there is no friendly-name mapping**. Use `GET /v1/models` to list the real IDs available on Bedrock and point your tools directly at them.

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
| `MANTLE_BASE_URL` | `https://bedrock-mantle.{AWS_REGION}.api.aws/v1` | Override for the Mantle endpoint |
| `BEDROCK_TOKEN_TTL_SECONDS` | `3600` | Requested lifetime for each generated token (max 12h / 43200s) |
| `MANTLE_REQUEST_TIMEOUT_SECONDS` | `300` | Timeout for requests forwarded to Mantle |
| `PROXY_HOST` | `0.0.0.0` | Bind address for the local server |
| `PROXY_PORT` | `8000` | Port for the local server |

## Running

```bash
python main.py
```

The server starts at `http://localhost:8000` (or the configured port). Check it with:

```bash
curl http://localhost:8000/healthz
```

## Connecting your tools

### Claude Code (Anthropic)

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=any-value   # ignored by the proxy; real auth happens against Mantle
```

### OpenAI-compatible tools (QwenCode, Pi Harness, OpenAI SDK)

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=any-value      # ignored by the proxy; real auth happens against Mantle
```

List the models available on Bedrock:

```bash
curl http://localhost:8000/v1/models
```

## Tests

The translation tests are unit tests and make no network calls or require AWS credentials:

```bash
pytest tests/ -v
```

## Known limitations

- `/v1/messages/count_tokens` (Anthropic) is not implemented.
- On **streaming** responses from `/v1/messages`, the `usage` field may come back zeroed if Mantle doesn't send a final chunk with usage information. Non-streaming responses report real usage.
- There is no friendly model-name mapping — the `model` value is passed through exactly as sent by the client tool.
