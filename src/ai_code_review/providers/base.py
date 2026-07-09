from __future__ import annotations

from typing import Protocol

from ..errors import ProviderError


class LLMProvider(Protocol):
    """Minimal contract every provider implements.

    `complete` takes a list of chat messages and returns the assistant's text
    response. Structured-output parsing happens upstream — providers stay
    agnostic to the review domain.
    """

    async def complete(self, messages: list[dict[str, str]]) -> str: ...

    async def aclose(self) -> None: ...


class RetryPolicy:
    """Decides whether a failed HTTP response is worth retrying."""

    def __init__(self, max_retries: int, base_delay: float = 1.0) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay

    def should_retry(self, status: int) -> bool:
        # 408/429 and any 5xx are worth another shot; 4xx is the caller's
        # fault (bad key, bad request) and must surface immediately.
        return status in {408, 429} or status >= 500

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and retry_after > 0:
            return float(min(retry_after, 30.0))
        return float(min(self.base_delay * (2**attempt), 30.0))


def raise_for_status(status: int, body: str, *, provider: str) -> None:
    if 200 <= status < 300:
        return
    if status in {401, 403}:
        raise ProviderError(f"{provider}: auth failed (HTTP {status})")
    if status == 404:
        raise ProviderError(f"{provider}: model or endpoint not found (HTTP 404)")
    raise ProviderError(f"{provider}: HTTP {status}: {body[:200]}")


__all__ = ["LLMProvider", "ProviderError", "RetryPolicy", "raise_for_status"]
