from __future__ import annotations

from typing import Any

import httpx
import pytest

from ai_code_review.config import Settings
from ai_code_review.errors import ProviderError
from ai_code_review.providers import OpenAIProvider, build_provider
from ai_code_review.providers.base import RetryPolicy


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "ai_provider": "openai",
        "ai_api_key": "sk-test",
        "ai_model": "gpt-4o-mini",
        "ai_retries": 2,
        "ai_timeout": 5.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _make_resp(status: int, body: Any, headers: dict[str, str] | None = None) -> httpx.Response:
    import json

    if isinstance(body, (dict, list)):
        content = json.dumps(body).encode()
        ct = "application/json"
    else:
        content = body.encode() if isinstance(body, str) else b""
        ct = "text/plain"
    return httpx.Response(status, content=content, headers={"content-type": ct, **(headers or {})})


async def test_openai_success(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings()
    p = OpenAIProvider(s)

    async def fake_post(self: OpenAIProvider, path: str, **kwargs: Any) -> httpx.Response:
        return _make_resp(
            200,
            {"choices": [{"message": {"content": "hello world"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    out = await p.complete([{"role": "user", "content": "hi"}])
    assert out == "hello world"
    await p.aclose()


async def test_openai_429_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(ai_retries=2)
    p = OpenAIProvider(s)
    calls: list[int] = []

    async def fake_post(self: OpenAIProvider, path: str, **kwargs: Any) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return _make_resp(429, "rate limit", headers={"retry-after": "0"})
        return _make_resp(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr("asyncio.sleep", lambda *a, **kw: _noop())  # type: ignore[arg-type]
    out = await p.complete([{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert len(calls) == 3
    await p.aclose()


async def test_openai_500_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(ai_retries=1)
    p = OpenAIProvider(s)

    async def fake_post(self: OpenAIProvider, path: str, **kwargs: Any) -> httpx.Response:
        return _make_resp(503, "down")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr("asyncio.sleep", lambda *a, **kw: _noop())  # type: ignore[arg-type]
    with pytest.raises(ProviderError):
        await p.complete([{"role": "user", "content": "hi"}])
    await p.aclose()


async def test_openai_400_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(ai_retries=3)
    p = OpenAIProvider(s)
    calls: list[int] = []

    async def fake_post(self: OpenAIProvider, path: str, **kwargs: Any) -> httpx.Response:
        calls.append(1)
        return _make_resp(401, "bad key")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(ProviderError, match="auth failed"):
        await p.complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    await p.aclose()


async def test_openai_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings()
    p = OpenAIProvider(s)

    async def fake_post(self: OpenAIProvider, path: str, **kwargs: Any) -> httpx.Response:
        return _make_resp(200, {"unexpected": "shape"})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(ProviderError, match="malformed"):
        await p.complete([{"role": "user", "content": "hi"}])
    await p.aclose()


async def test_openai_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(ai_retries=0, ai_timeout=0.1)
    p = OpenAIProvider(s)

    async def fake_post(self: OpenAIProvider, path: str, **kwargs: Any) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(ProviderError):
        await p.complete([{"role": "user", "content": "hi"}])
    await p.aclose()


def test_retry_policy_logic() -> None:
    rp = RetryPolicy(3)
    assert rp.should_retry(429)
    assert rp.should_retry(503)
    assert rp.should_retry(500)
    assert not rp.should_retry(400)
    assert not rp.should_retry(404)
    assert rp.delay(0) > 0
    assert rp.delay(0, retry_after=2.0) == 2.0


def test_build_provider_openai() -> None:
    s = _settings(ai_provider="openai")
    p = build_provider(s)
    assert isinstance(p, OpenAIProvider)


def test_build_provider_local_uses_openai_client() -> None:
    s = _settings(ai_provider="local", ai_base_url="http://localhost:8000/v1")
    p = build_provider(s)
    assert isinstance(p, OpenAIProvider)
    assert p.base_url == "http://localhost:8000/v1"


def test_build_provider_unknown_raises() -> None:
    s = _settings()
    s.__dict__["ai_provider"] = "bogus"  # bypass validation
    from ai_code_review.errors import ConfigError

    with pytest.raises(ConfigError):
        build_provider(s)


async def _noop() -> None:
    return None
