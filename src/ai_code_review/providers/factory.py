from __future__ import annotations

from ..config import Settings
from ..errors import ConfigError
from .anthropic import AnthropicProvider
from .base import LLMProvider
from .openai import OpenAIProvider


def build_provider(settings: Settings, *, api_key: str | None = None) -> LLMProvider:
    """Construct an LLM provider from the configured backend.

    A single OpenAI-compatible client covers two cases: hosted OpenAI and any
    OpenAI-compatible local server. Anthropic gets its own client because its
    API differs enough (system prompt handling, content blocks) to warrant a
    dedicated implementation. Azure OpenAI is intentionally NOT supported
    here - see OpenAIProvider's docstring.
    """
    name = settings.ai_provider
    if name in {"openai", "local"}:
        return OpenAIProvider(settings, api_key=api_key)
    if name == "anthropic":
        return AnthropicProvider(settings, api_key=api_key)
    raise ConfigError(f"unsupported ai_provider: {name!r}")


__all__ = [
    "AnthropicProvider",
    "LLMProvider",
    "OpenAIProvider",
    "build_provider",
]
