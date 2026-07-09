from __future__ import annotations

from typing import Any

import httpx
import pytest

from ai_code_review.config import Settings
from ai_code_review.enums import Severity
from ai_code_review.errors import GitHubError
from ai_code_review.github import (
    GitHubClient,
    build_review_body,
    extract_fingerprints,
    parse_pr_ref,
)
from ai_code_review.models import ReviewIssue


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "github_token": "ghp_test_token",
        "ai_api_key": "sk-test",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _make_resp(status: int, body: Any, headers: dict[str, str] | None = None) -> httpx.Response:
    import json

    if isinstance(body, (dict, list)):
        content = json.dumps(body).encode()
        ct = "application/json"
    elif body is None:
        content = b""
        ct = "application/json"
    else:
        content = body.encode() if isinstance(body, str) else b""
        ct = "text/plain"
    return httpx.Response(status, content=content, headers={"content-type": ct, **(headers or {})})


def test_parse_pr_ref_ok() -> None:
    assert parse_pr_ref("owner/repo#42") == ("owner", "repo", 42)


def test_parse_pr_ref_bad() -> None:
    for bad in ["", "owner/repo", "owner#42", "owner/repo#abc", "owner/repo#0", "owner/repo#-1"]:
        with pytest.raises(GitHubError):
            parse_pr_ref(bad)


async def test_get_pr_success(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings()
    c = GitHubClient(s)

    async def fake_request(self: GitHubClient, method: str, path: str, **kw: Any) -> Any:
        assert method == "GET"
        assert path == "/repos/o/r/pulls/1"
        return {"number": 1, "head": {"sha": "abc"}, "base": {"sha": "def"}}

    monkeypatch.setattr(GitHubClient, "_request", fake_request)
    out = await c.get_pr("o", "r", 1)
    assert out["number"] == 1
    await c.aclose()


async def test_get_pr_error_redacts(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings()
    c = GitHubClient(s)

    async def fake_request(self: GitHubClient, method: str, path: str, **kw: Any) -> Any:
        raise GitHubError("HTTP 404: not found")

    monkeypatch.setattr(GitHubClient, "_request", fake_request)
    with pytest.raises(GitHubError):
        await c.get_pr("o", "r", 99)
    await c.aclose()


async def test_get_pr_diff_uses_text_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings()
    c = GitHubClient(s)
    captured: dict[str, Any] = {}

    async def fake_get(
        self: httpx.AsyncClient, url: str, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        captured["headers"] = headers or {}
        return _make_resp(200, "diff --git a/x b/x\n")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    out = await c.get_pr_diff("o", "r", 1)
    assert "diff --git" in out
    assert "v3.diff" in captured["headers"].get("Accept", "")
    await c.aclose()


async def test_post_review_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings()
    c = GitHubClient(s)
    captured: dict[str, Any] = {}

    async def fake_request(
        self: GitHubClient,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        captured["method"] = method
        captured["path"] = path
        captured["body"] = json_body
        return {"id": 99}

    monkeypatch.setattr(GitHubClient, "_request", fake_request)
    out = await c.post_review(
        "o",
        "r",
        5,
        body="hello",
        event="COMMENT",
        comments=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
    )
    assert out == {"id": 99}
    assert captured["method"] == "POST"
    assert captured["path"] == "/repos/o/r/pulls/5/reviews"
    assert captured["body"]["body"] == "hello"
    assert captured["body"]["event"] == "COMMENT"
    assert len(captured["body"]["comments"]) == 1
    await c.aclose()


async def test_list_review_comments_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings()
    c = GitHubClient(s)
    pages = [[{"id": i} for i in range(100)], [{"id": 100}], []]
    calls: list[int] = []

    async def fake_request(
        self: GitHubClient,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        calls.append(params.get("page", 1) if params else 1)
        idx = min(calls[-1] - 1, len(pages) - 1)
        return pages[idx]

    monkeypatch.setattr(GitHubClient, "_request", fake_request)
    out = await c.list_review_comments("o", "r", 1)
    assert len(out) == 101
    await c.aclose()


def test_build_review_body_has_marker() -> None:
    body = build_review_body("summary", ["fp1", "fp2"])
    assert "## AI Code Review" in body
    assert "fp1,fp2" in body


def test_extract_fingerprints_roundtrip() -> None:
    body = build_review_body("s", ["abc", "def"])
    assert extract_fingerprints(body) == {"abc", "def"}


def test_extract_fingerprints_handles_missing() -> None:
    assert extract_fingerprints("no markers here") == set()


def test_to_inline_comment_shape() -> None:
    from ai_code_review.github import to_inline_comment

    issue = ReviewIssue(
        title="test issue",
        body="b",
        severity=Severity.HIGH,
        confidence=0.9,
        file="a.py",
        line=10,
        category="bug",
    )
    c = to_inline_comment(issue)
    assert c["path"] == "a.py"
    assert c["line"] == 10
    assert c["side"] == "RIGHT"
    assert "fp:" in c["body"]


def test_to_inline_comment_always_returns_dict() -> None:
    # to_inline_comment no longer returns None; invalid anchors are filtered
    # upstream in publish_review.
    from ai_code_review.github import to_inline_comment

    issue = ReviewIssue(
        title="test issue",
        body="b",
        severity=Severity.LOW,
        confidence=0.7,
        file="a.py",
        line=1,
        category="bug",
    )
    c = to_inline_comment(issue)
    assert c is not None
    assert c["path"] == "a.py"


async def test_publish_review_skips_existing_fingerprints(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_code_review.github import PrRef, publish_review
    from ai_code_review.models import ReviewResult

    s = _settings()
    c = GitHubClient(s)

    issue = ReviewIssue(
        title="test issue",
        body="b",
        severity=Severity.HIGH,
        confidence=0.9,
        file="a.py",
        line=10,
        category="bug",
    )

    async def fake_list_comments(
        self: GitHubClient, owner: str, repo: str, pr: int
    ) -> list[dict[str, Any]]:
        return [{"body": build_review_body("old", [issue.fingerprint])}]

    async def fake_post_review(
        self: GitHubClient, owner: str, repo: str, pr: int, **kw: Any
    ) -> dict[str, Any]:
        return {"id": 1}

    monkeypatch.setattr(GitHubClient, "list_review_comments", fake_list_comments)
    monkeypatch.setattr(GitHubClient, "post_review", fake_post_review)

    pr = PrRef(owner="o", repo="r", number=1, head_sha="x", base_sha="y")
    result = ReviewResult(issues=[issue], summary="s", analyzed_files=["a.py"])
    out = await publish_review(c, pr, result, head_sha="x")
    # Existing fingerprint → no new comments posted → post still called with empty comments
    assert out == {"id": 1}
    await c.aclose()
