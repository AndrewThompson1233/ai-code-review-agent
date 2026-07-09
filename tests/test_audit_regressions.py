"""Regression tests for the audit fixes.

Each test targets one confirmed bug from the audit:
  - parse retry actually retries on malformed JSON
  - whole-pipeline timeout caps slow providers
  - concurrency is bounded by the semaphore
  - max_files produces a single source of truth and correct reporting
  - resilient GitHub publishing falls back to summary
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_code_review.agent.pipeline import PipelineDeps, PipelineFailure, ReviewPipeline
from ai_code_review.config import Settings
from ai_code_review.enums import Severity
from ai_code_review.errors import GitHubError, ProviderError, ReviewError
from ai_code_review.github import PrRef, build_review_body, publish_review
from ai_code_review.models import FileDiff, ReviewIssue, ReviewResult


class FakeProvider:
    def __init__(self, responses: list[str] | None = None, *, delay: float = 0.0) -> None:
        self._responses = list(responses or [])
        self._delay = delay
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._responses:
            return self._responses.pop(0)
        return '{"issues": []}'

    async def aclose(self) -> None:
        pass


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "ai_provider": "openai",
        "ai_api_key": "sk-test",
        "ai_model": "gpt-4o-mini",
        "ai_retries": 0,
        "review_min_confidence": 0.5,
        "review_concurrency": 2,
        "review_timeout": 10.0,
    }
    base.update(overrides)
    return Settings(**base)


def _file(path: str, added_lines: list[int], patch: str = "") -> FileDiff:
    return FileDiff(path=path, added_lines=added_lines, patch=patch or "@@ -1 +1 @@\n+x\n")


def _issue(line: int, conf: float = 0.9, sev: str = "high", file: str = "a.py") -> str:
    import json

    return json.dumps(
        {
            "issues": [
                {
                    "title": "test issue here",
                    "body": "test body content",
                    "severity": sev,
                    "confidence": conf,
                    "file": file,
                    "line": line,
                    "category": "bug",
                }
            ]
        }
    )


# ---------- P3: real parse retry ----------


async def test_parse_retry_succeeds_on_second_attempt() -> None:
    # First response malformed, second valid. Should NOT lose the chunk.
    provider = FakeProvider(["not json at all", _issue(line=5)])
    files = [_file("a.py", [5])]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    out = await p.run(files)
    assert len(out.issues) == 1
    assert out.issues[0].line == 5
    # Two calls: initial + one repair retry.
    assert len(provider.calls) == 2


async def test_parse_retry_exhausted_raises_critical_failure() -> None:
    # Both responses malformed. Single file, all chunks fail → critical.
    provider = FakeProvider(["not json", "also not json"])
    files = [_file("a.py", [5])]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    with pytest.raises(PipelineFailure):
        await p.run(files)


async def test_parse_retry_valid_first_response_single_call() -> None:
    provider = FakeProvider([_issue(line=5)])
    files = [_file("a.py", [5])]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    await p.run(files)
    assert len(provider.calls) == 1


# ---------- P5: whole-pipeline timeout ----------


async def test_pipeline_timeout_caps_slow_provider() -> None:
    provider = FakeProvider(delay=5.0)
    files = [_file("a.py", [5])]
    p = ReviewPipeline(
        PipelineDeps(
            provider=provider,
            settings=_settings(review_timeout=0.2, ai_timeout=0.2),
        )
    )
    with pytest.raises(ReviewError, match="timed out"):
        await p.run(files)


# ---------- P7: bounded concurrency ----------


async def test_concurrency_is_bounded_by_semaphore() -> None:
    # Track concurrent in-flight calls; semaphore should cap them.
    in_flight = {"current": 0, "peak": 0}
    lock = asyncio.Lock()

    class TrackingProvider(FakeProvider):
        async def complete(self, messages: list[dict[str, str]]) -> str:
            async with lock:
                in_flight["current"] += 1
                in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
            await asyncio.sleep(0.05)
            async with lock:
                in_flight["current"] -= 1
            return '{"issues": []}'

    provider = TrackingProvider()
    files = [_file(f"f{i}.py", [1]) for i in range(6)]
    p = ReviewPipeline(
        PipelineDeps(
            provider=provider,
            settings=_settings(review_concurrency=2),
        )
    )
    await p.run(files)
    assert in_flight["peak"] <= 2, f"concurrency exceeded cap: peak={in_flight['peak']}"


async def test_concurrency_preserves_input_order() -> None:
    # Each file's provider returns a different delay; results must still be
    # returned in input file order.
    delays = [0.08, 0.01, 0.05, 0.02, 0.07]

    class OrderingProvider(FakeProvider):
        def __init__(self) -> None:
            self._idx = 0

        async def complete(self, messages: list[dict[str, str]]) -> str:
            idx = self._idx
            self._idx += 1
            await asyncio.sleep(delays[idx % len(delays)])
            return _issue(line=1, file=f"f{idx}.py")

    provider = OrderingProvider()
    files = [_file(f"f{i}.py", [1]) for i in range(5)]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    out = await p.run(files)
    # Issues should be sorted by severity then confidence, but since all are
    # identical they retain input order.
    assert [i.file for i in out.issues] == [f"f{i}.py" for i in range(5)]


# ---------- P6: single source of truth for max_files ----------


async def test_max_files_reports_omitted_not_duplicated() -> None:
    provider = FakeProvider(['{"issues": []}'] * 10)
    files = [_file(f"f{i}.py", [1]) for i in range(10)]
    p = ReviewPipeline(
        PipelineDeps(
            provider=provider,
            settings=_settings(review_max_files=3),
        )
    )
    out = await p.run(files)
    assert out.limited is True
    assert out.limit_reason is not None
    assert "capping at 3" in out.limit_reason
    assert len(out.analyzed_files) == 3
    assert len(out.omitted_files) == 7
    assert out.total_changed_files == 10


# ---------- P4: critical failure raises ----------


async def test_all_files_failed_raises_pipeline_failure() -> None:
    class FailingProvider(FakeProvider):
        async def complete(self, messages: list[dict[str, str]]) -> str:
            raise ProviderError("down")

    provider = FailingProvider()
    files = [_file(f"f{i}.py", [1]) for i in range(3)]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    with pytest.raises(PipelineFailure):
        await p.run(files)


async def test_partial_failure_does_not_raise() -> None:
    state = {"n": 0}

    class MixedProvider(FakeProvider):
        async def complete(self, messages: list[dict[str, str]]) -> str:
            state["n"] += 1
            if state["n"] % 2 == 0:
                raise ProviderError("every other fails")
            return _issue(
                line=1,
                file=messages[-1]["content"].split("`")[1]
                if "`" in messages[-1]["content"]
                else "x.py",
            )

    provider = MixedProvider()
    files = [_file(f"f{i}.py", [1]) for i in range(4)]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    out = await p.run(files)
    assert len(out.failed_files) > 0
    assert len(out.analyzed_files) > 0
    assert "failed" in out.summary.lower() or "incomplete" in out.summary.lower()


# ---------- P2: context from git tree (uses FakeRepo) ----------


class FakeRepo:
    def __init__(self, files: dict[str, str]) -> None:
        self._files = files

    def read_file_from_tree(self, path: str, ref: str = "HEAD") -> str:
        if path not in self._files:
            from ai_code_review.errors import GitError

            raise GitError(f"file not found in {ref}: {path}")
        return self._files[path]


async def test_context_read_from_repo_not_worktree() -> None:
    # The pipeline should call read_file_from_tree, not read from disk.
    fake_repo = FakeRepo({"a.py": "committed = 1\n"})
    provider = FakeProvider(['{"issues": []}'])
    files = [FileDiff(path="a.py", added_lines=[1], patch="@@ -1 +1 @@\n+x\n")]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings(), repo=fake_repo))
    out = await p.run(files)
    assert out.analyzed_files == ["a.py"]
    # The provider received the context block with the committed content.
    assert "committed = 1" in provider.calls[0][1]["content"]


async def test_context_missing_file_does_not_abort() -> None:
    fake_repo = FakeRepo({})  # nothing in tree
    provider = FakeProvider(['{"issues": []}'])
    files = [FileDiff(path="missing.py", added_lines=[1], patch="@@ -1 +1 @@\n+x\n")]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings(), repo=fake_repo))
    out = await p.run(files)
    assert out.analyzed_files == ["missing.py"]


# ---------- P8: resilient GitHub publishing ----------


class FakeGitHubClient:
    def __init__(
        self,
        *,
        list_comments: list[dict[str, Any]] | None = None,
        post_results: list[Exception | dict[str, Any]] | None = None,
    ) -> None:
        self._list_comments = list_comments or []
        self._post_results = list(post_results or [])
        self.post_calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> FakeGitHubClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def list_review_comments(self, owner: str, repo: str, pr: int) -> list[dict[str, Any]]:
        return self._list_comments

    async def post_review(
        self,
        owner: str,
        repo: str,
        pr: int,
        *,
        body: str,
        event: str = "COMMENT",
        comments: list[dict[str, Any]] | None = None,
        commit_id: str | None = None,
    ) -> dict[str, Any]:
        self.post_calls.append({"body": body, "event": event, "comments": comments})
        if self._post_results:
            r = self._post_results.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        return {"id": 1}


async def test_publish_review_falls_back_to_summary_on_inline_rejection() -> None:
    issue = ReviewIssue(
        title="test issue here",
        body="b",
        severity=Severity.HIGH,
        confidence=0.9,
        file="a.py",
        line=10,
        category="bug",
    )
    result = ReviewResult(issues=[issue], summary="found 1 issue", analyzed_files=["a.py"])
    # First post (with inline) fails; second post (summary-only) succeeds.
    client = FakeGitHubClient(post_results=[GitHubError("invalid line"), {"id": 99}])
    pr = PrRef(owner="o", repo="r", number=1, head_sha="sha")
    out = await publish_review(client, pr, result, head_sha="sha")
    assert out == {"id": 99}
    assert len(client.post_calls) == 2
    # First call had inline comments; second call had none.
    assert client.post_calls[0]["comments"] is not None
    assert len(client.post_calls[0]["comments"]) == 1
    assert client.post_calls[1]["comments"] is None


async def test_publish_review_idempotent_skips_existing() -> None:
    issue = ReviewIssue(
        title="test issue here",
        body="b",
        severity=Severity.HIGH,
        confidence=0.9,
        file="a.py",
        line=10,
        category="bug",
    )
    result = ReviewResult(issues=[issue], summary="found 1 issue", analyzed_files=["a.py"])
    # Existing comment already has this fingerprint.
    existing_body = build_review_body("old", [issue.fingerprint])
    client = FakeGitHubClient(list_comments=[{"body": existing_body}])
    pr = PrRef(owner="o", repo="r", number=1, head_sha="sha")
    await publish_review(client, pr, result, head_sha="sha")
    # No new issues to post → summary-only fallback still runs (with empty
    # inline list). Verify no inline comments were sent.
    if client.post_calls:
        assert (
            client.post_calls[0]["comments"] is None or len(client.post_calls[0]["comments"]) == 0
        )


async def test_publish_review_both_attempts_fail_raises() -> None:
    issue = ReviewIssue(
        title="test issue here",
        body="b",
        severity=Severity.HIGH,
        confidence=0.9,
        file="a.py",
        line=10,
        category="bug",
    )
    result = ReviewResult(issues=[issue], summary="found 1 issue", analyzed_files=["a.py"])
    client = FakeGitHubClient(
        post_results=[
            GitHubError("inline rejected"),
            GitHubError("summary also rejected"),
        ]
    )
    pr = PrRef(owner="o", repo="r", number=1, head_sha="sha")
    with pytest.raises(GitHubError):
        await publish_review(client, pr, result, head_sha="sha")


# ---------- HTTP Retry-After HTTP-date support ----------


def test_parse_retry_after_numeric() -> None:
    from ai_code_review.providers.http_util import parse_retry_after

    assert parse_retry_after("42") == 42.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("not a number") is None


def test_parse_retry_after_http_date() -> None:
    from datetime import UTC, datetime, timedelta

    from ai_code_review.providers.http_util import parse_retry_after

    future = datetime.now(UTC) + timedelta(seconds=30)
    from email.utils import format_datetime

    val = format_datetime(future, usegmt=True)
    result = parse_retry_after(val)
    assert result is not None
    assert 20 < result < 35  # roughly 30s, with scheduling slack


def test_parse_retry_after_clamps_large_values() -> None:
    from ai_code_review.providers.http_util import parse_retry_after

    assert parse_retry_after("99999") == 300.0
