import asyncio
from datetime import timedelta

from aws_bedrock_token_generator import provide_token

from .config import settings


async def get_bedrock_token() -> str:
    """Generate a fresh Bedrock Mantle bearer token for the current request.

    A new token is minted on every call (per the security requirement that a
    static Mantle API key cannot be issued). Token generation is a local
    SigV4 signing operation, not a network round trip, so calling it once per
    proxied request is cheap. It runs in a thread because it resolves AWS
    credentials (env/profile/SSO/role) via botocore, which can block.
    """
    return await asyncio.to_thread(
        provide_token,
        region=settings.aws_region,
        expiry=timedelta(seconds=settings.token_ttl_seconds),
    )
