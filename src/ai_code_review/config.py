from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums import LogLevel
from .errors import ConfigError

_DEFAULT_GITHUB_API = "https://api.github.com"


class Settings(BaseSettings):
    """Runtime configuration loaded from env vars and an optional .env file.

    Every secret is wrapped in SecretStr so it is never accidentally logged by
    pydantic's repr or by an exception that captures the model.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM provider ---
    ai_provider: str = "openai"
    ai_model: str = ""
    ai_api_key: SecretStr = SecretStr("")
    ai_base_url: str | None = None
    ai_timeout: float = 120.0
    ai_retries: int = 3
    ai_temperature: float = 0.1
    ai_max_tokens: int = 4096

    # --- GitHub ---
    github_token: SecretStr = SecretStr("")
    github_api_url: str = _DEFAULT_GITHUB_API

    # --- Review behavior ---
    review_min_confidence: float = 0.7
    review_max_files: int = 50
    review_max_file_size: int = 512 * 1024
    review_max_context_size: int = 64 * 1024
    review_timeout: float = 300.0
    review_concurrency: int = 3

    # --- Output ---
    log_level: LogLevel = LogLevel.INFO

    @field_validator("ai_provider")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"openai", "anthropic", "local"}:
            raise ValueError(f"unsupported ai_provider: {v!r}")
        return v

    @field_validator("review_min_confidence")
    @classmethod
    def _check_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("review_min_confidence must be in [0.0, 1.0]")
        return v

    @field_validator("ai_retries")
    @classmethod
    def _check_retries(cls, v: int) -> int:
        if not 0 <= v <= 10:
            raise ValueError("ai_retries must be in [0, 10]")
        return v

    @field_validator("review_concurrency")
    @classmethod
    def _check_concurrency(cls, v: int) -> int:
        if not 1 <= v <= 16:
            raise ValueError("review_concurrency must be in [1, 16]")
        return v

    @field_validator("review_timeout")
    @classmethod
    def _check_timeout(cls, v: float) -> float:
        if v <= 0 or v > 3600:
            raise ValueError("review_timeout must be in (0, 3600]")
        return v

    def require_api_key(self) -> str:
        if self.ai_provider == "local":
            return self.ai_api_key.get_secret_value()
        key = self.ai_api_key.get_secret_value()
        if not key:
            raise ConfigError(f"AI_API_KEY is required for provider {self.ai_provider!r}")
        return key

    def require_github_token(self) -> str:
        token = self.github_token.get_secret_value()
        if not token:
            raise ConfigError("GITHUB_TOKEN is required for GitHub integration")
        return token

    def default_model(self) -> str:
        if self.ai_model:
            return self.ai_model
        return {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
            "local": "qwen2.5-coder-7b-instruct",
        }[self.ai_provider]


def load_settings(
    env_file: Path | str | None = None,
    *,
    ai_provider: str | None = None,
    ai_model: str | None = None,
    review_min_confidence: float | None = None,
    review_max_files: int | None = None,
    review_concurrency: int | None = None,
    review_timeout: float | None = None,
) -> Settings:
    """Load settings, optionally pointing at a specific .env path.

    Only the overrides the CLI actually needs are exposed — passing a dict of
    arbitrary kwargs would defeat pydantic-settings' field validation.
    """
    kwargs: dict[str, object] = {}
    if ai_provider is not None:
        kwargs["ai_provider"] = ai_provider
    if ai_model is not None:
        kwargs["ai_model"] = ai_model
    if review_min_confidence is not None:
        kwargs["review_min_confidence"] = review_min_confidence
    if review_max_files is not None:
        kwargs["review_max_files"] = review_max_files
    if review_concurrency is not None:
        kwargs["review_concurrency"] = review_concurrency
    if review_timeout is not None:
        kwargs["review_timeout"] = review_timeout
    if env_file is not None:
        kwargs["_env_file"] = str(env_file)
    return Settings(**kwargs)  # type: ignore[arg-type]
