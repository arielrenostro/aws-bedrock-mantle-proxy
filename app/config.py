import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    aws_region: str
    mantle_base_url: str
    token_ttl_seconds: int
    request_timeout: float
    host: str
    port: int


def load_settings() -> Settings:
    region = os.getenv("AWS_REGION", "us-east-1")
    base_url = os.getenv("MANTLE_BASE_URL", f"https://bedrock-mantle.{region}.api.aws/v1")
    return Settings(
        aws_region=region,
        mantle_base_url=base_url.rstrip("/"),
        token_ttl_seconds=int(os.getenv("BEDROCK_TOKEN_TTL_SECONDS", "3600")),
        request_timeout=float(os.getenv("MANTLE_REQUEST_TIMEOUT_SECONDS", "300")),
        host=os.getenv("PROXY_HOST", "0.0.0.0"),
        port=int(os.getenv("PROXY_PORT", "8000")),
    )


settings = load_settings()
