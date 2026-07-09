from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from ai_code_review.config import Settings
from ai_code_review.errors import ConfigError


def test_defaults_with_test_env() -> None:
    s = Settings()
    assert s.ai_provider in {"openai", "anthropic", "local"}
    assert 0.0 <= s.review_min_confidence <= 1.0
    assert s.ai_retries >= 0


def test_provider_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(ai_provider="bogus")


def test_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Settings(review_min_confidence=1.5)
    with pytest.raises(ValidationError):
        Settings(review_min_confidence=-0.1)


def test_retries_bounds() -> None:
    with pytest.raises(ValidationError):
        Settings(ai_retries=99)


def test_require_api_key_for_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    s = Settings(ai_provider="openai", ai_api_key=SecretStr(""))
    with pytest.raises(ConfigError):
        s.require_api_key()


def test_local_provider_does_not_require_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    s = Settings(ai_provider="local", ai_api_key=SecretStr(""))
    # Should not raise even though the key is empty.
    assert s.require_api_key() == ""


def test_require_github_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    s = Settings(github_token=SecretStr(""))
    with pytest.raises(ConfigError):
        s.require_github_token()


def test_default_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(ai_provider="openai", ai_model="")
    assert s.default_model() == "gpt-4o-mini"
    s2 = Settings(ai_provider="anthropic", ai_model="")
    assert "claude" in s2.default_model()
    s3 = Settings(ai_provider="local", ai_model="")
    assert s3.default_model()


def test_secret_str_not_revealed_in_repr() -> None:
    s = Settings(ai_api_key="sk-supersecret")
    rep = repr(s)
    assert "sk-supersecret" not in rep
