from __future__ import annotations

from ai_code_review.redact import redact


def test_redact_authorization_header() -> None:
    out = redact("authorization: Bearer sk-abc1234567890")
    assert "sk-abc1234567890" not in out
    assert "<redacted>" in out


def test_redact_bearer_token() -> None:
    out = redact("Bearer ghp_1234567890abcdef")
    assert "ghp_1234567890abcdef" not in out


def test_redact_openai_key() -> None:
    out = redact("config: sk-abcdefghijklmnopqrstuvwxyz")
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in out


def test_redact_github_token() -> None:
    out = redact("token: ghp_1234567890abcdefghijklmnopqrstuvwxyz")
    assert "ghp_1234567890" not in out


def test_redact_password() -> None:
    out = redact("password=supersecret123")
    assert "supersecret123" not in out


def test_redact_preserves_normal_text() -> None:
    text = "Found 3 issues in src/app.py at line 42"
    assert redact(text) == text


def test_redact_handles_empty() -> None:
    assert redact("") == ""
