from __future__ import annotations

import asyncio
import json
from types import TracebackType
from typing import Any

import httpx

from ..config import Settings
from ..errors import ProviderError, ProviderTimeout
from ..redact import redact
from .base import RetryPolicy, raise_for_status
from .http_util import parse_retry_after

_USER_AGENT = (
    "ai-code-review-agent/0.1 (+https://github.com/assasino15555-star/ai-code-review-agent)"
)
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    """Anthropic Messages API client.

    Anthropic splits the prompt into a `system` string plus a list of user /
    assistant turns. We translate our generic chat-message list into that
    shape, keeping the first system message separate from the turn list.
    """

    def __init__(self, settings: Settings, *, api_key: str | None = None) -> None:
        self.settings = settings
        self.api_key = api_key or settings.require_api_key()
        self.base_url = (settings.ai_base_url or "https://api.anthropic.com").rstrip("/")
        self.model = settings.default_model()
        self.retry = RetryPolicy(settings.ai_retries)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AnthropicProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.settings.ai_timeout, connect=10.0),
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "Content-Type": "application/json",
                    "User-Agent": _USER_AGENT,
                    "Accept": "application/json",
                },
            )
        return self._client

    def _split(self, messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        system = ""
        turns: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "system":
                if not system:
                    system = m.get("content", "")
                else:
                    system = f"{system}\n\n{m.get('content', '')}"
            else:
                turns.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        if not turns:
            turns = [{"role": "user", "content": system or "review the following diff"}]
            system = ""
        return system, turns

    async def complete(self, messages: list[dict[str, str]]) -> str:
        system, turns = self._split(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
        }
        if system:
            payload["system"] = system

        client = await self._ensure_client()
        last_err: ProviderError | None = None
        for attempt in range(self.retry.max_retries + 1):
            try:
                resp = await client.post("/v1/messages", json=payload)
            except httpx.TimeoutException as exc:
                last_err = ProviderTimeout(f"anthropic: request timed out: {exc}")
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(self.retry.delay(attempt))
                    continue
                raise last_err from exc
            except httpx.HTTPError as exc:
                last_err = ProviderError(f"anthropic: transport error: {exc}")
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(self.retry.delay(attempt))
                    continue
                raise last_err from exc

            if self.retry.should_retry(resp.status_code):
                retry_after = parse_retry_after(resp.headers.get("retry-after"))
                last_err = ProviderError(
                    f"anthropic: HTTP {resp.status_code}: {redact(resp.text)[:200]}"
                )
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(self.retry.delay(attempt, retry_after))
                    continue
                raise last_err

            if resp.status_code >= 400:
                raise_for_status(resp.status_code, resp.text, provider="anthropic")

            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                raise ProviderError("anthropic: response is not JSON") from exc

            try:
                blocks = data["content"]
                if not isinstance(blocks, list) or not blocks:
                    raise ProviderError("anthropic: empty content array")
                text_parts: list[str] = []
                for b in blocks:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text_parts.append(str(b.get("text", "")))
                return "".join(text_parts)
            except (KeyError, TypeError) as exc:
                raise ProviderError(
                    f"anthropic: malformed response: {redact(json.dumps(data))[:200]}"
                ) from exc

        if last_err is None:
            raise ProviderError("anthropic: retries exhausted without error")
        raise last_err

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
