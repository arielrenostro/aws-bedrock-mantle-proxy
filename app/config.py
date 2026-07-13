import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_APP_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Settings:
    aws_region: str
    mantle_base_url: str
    token_ttl_seconds: int
    request_timeout: float
    host: str
    port: int
    model_contracts_path: Path


def load_settings() -> Settings:
    region = os.getenv("AWS_REGION", "us-east-1")
    # Host root only — the OpenAI-compatible surface lives under /v1/... and
    # the native Anthropic Messages API lives under /anthropic/v1/... ; these
    # are sibling paths, not nested under a shared /v1 prefix.
    base_url = os.getenv("MANTLE_BASE_URL", f"https://bedrock-mantle.{region}.api.aws")
    contracts_path = os.getenv("MODEL_CONTRACTS_FILE")
    return Settings(
        aws_region=region,
        mantle_base_url=base_url.rstrip("/"),
        token_ttl_seconds=int(os.getenv("BEDROCK_TOKEN_TTL_SECONDS", "3600")),
        request_timeout=float(os.getenv("MANTLE_REQUEST_TIMEOUT_SECONDS", "300")),
        host=os.getenv("PROXY_HOST", "0.0.0.0"),
        port=int(os.getenv("PROXY_PORT", "8000")),
        model_contracts_path=Path(contracts_path) if contracts_path else _APP_DIR / "model_contracts.json",
    )


settings = load_settings()
