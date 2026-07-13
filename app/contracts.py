from enum import Enum


class Contract(str, Enum):
    """Which wire format an API surface speaks — either the client's entry
    format, or the format Mantle expects for a given model."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
