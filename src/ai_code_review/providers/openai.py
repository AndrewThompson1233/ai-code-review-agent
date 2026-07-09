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


class OpenAIProvider:
    """OpenAI-compatible chat completions client.

    Works with the public OpenAI API and any OpenAI-compatible local server
    (llama.cpp, vLLM, Ollama, LM Studio). All of them speak the same
    `/chat/completions` schema, so a single implementation covers them.

    Note: Azure OpenAI is NOT supported by this client. Azure uses a
    deployment-based URL scheme and a different auth header; point an
    Azure-compatible proxy at this client's `AI_BASE_URL` if you need it.
    """

    def __init__(self, settings: Settings, *, api_key: str | None = None) -> None:
        self.settings = settings
        self.api_key = api_key or settings.require_api_key()
        self.base_url = (settings.ai_base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = settings.default_model()
        self.retry = RetryPolicy(settings.ai_retries)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OpenAIProvider:
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
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": _USER_AGENT,
                    "Accept": "application/json",
                },
            )
        return self._client

    async def complete(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "stream": False,
        }
        client = await self._ensure_client()
        last_err: ProviderError | None = None
        for attempt in range(self.retry.max_retries + 1):
            try:
                resp = await client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                last_err = ProviderTimeout(f"openai: request timed out: {exc}")
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(self.retry.delay(attempt))
                    continue
                raise last_err from exc
            except httpx.HTTPError as exc:
                last_err = ProviderError(f"openai: transport error: {exc}")
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(self.retry.delay(attempt))
                    continue
                raise last_err from exc

            if self.retry.should_retry(resp.status_code):
                retry_after = parse_retry_after(resp.headers.get("retry-after"))
                last_err = ProviderError(
                    f"openai: HTTP {resp.status_code}: {redact(resp.text)[:200]}"
                )
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(self.retry.delay(attempt, retry_after))
                    continue
                raise last_err

            if resp.status_code >= 400:
                raise_for_status(resp.status_code, resp.text, provider="openai")

            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                raise ProviderError("openai: response is not JSON") from exc

            try:
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError(
                    f"openai: malformed response: {redact(json.dumps(data))[:200]}"
                ) from exc

        if last_err is None:
            raise ProviderError("openai: retries exhausted without error")
        raise last_err

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
