from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..errors import GitHubError
from ..models import ReviewIssue, ReviewResult
from .client import GitHubClient, build_review_body, extract_fingerprints

log = logging.getLogger("ai_code_review.github")


@dataclass
class PrRef:
    owner: str
    repo: str
    number: int
    head_sha: str = ""
    base_sha: str = ""


def parse_pr_ref(ref: str) -> tuple[str, str, int]:
    """Parse `owner/repo#123` into a triple. Raises on bad input.

    This is the single source of truth for PR-ref parsing. The GitHubClient
    method of the same name was removed to avoid the duplication.
    """
    ref = ref.strip()
    if "/" not in ref or "#" not in ref:
        raise GitHubError(f"invalid PR ref {ref!r}; expected owner/repo#N")
    owner_repo, num = ref.rsplit("#", 1)
    if "/" not in owner_repo:
        raise GitHubError(f"invalid PR ref {ref!r}; expected owner/repo#N")
    owner, repo = owner_repo.split("/", 1)
    try:
        n = int(num)
    except ValueError as exc:
        raise GitHubError(f"invalid PR number {num!r}") from exc
    if n < 1:
        raise GitHubError(f"invalid PR number {n}")
    return owner, repo, n


def to_inline_comment(issue: ReviewIssue) -> dict[str, Any]:
    """Convert an issue into a GitHub inline review comment payload.

    `head_sha` is supplied separately to `publish_review` and is not needed
    here: the inline comment anchors to `line`/`side=RIGHT` against the PR's
    head, which GitHub infers from the review's `commit_id` field.
    """
    body = (
        f"**[{issue.severity.value.upper()}] {issue.title}**\n\n"
        f"{issue.body}\n\n"
        f"_category: {issue.category.value} · confidence: {issue.confidence:.2f}_\n"
        f"<!-- ai-code-review-agent:fp:{issue.fingerprint} -->"
    )
    return {
        "path": issue.file,
        "line": issue.line,
        "side": "RIGHT",
        "body": body,
    }


async def publish_review(
    client: GitHubClient,
    pr: PrRef,
    result: ReviewResult,
    *,
    head_sha: str,
    dedup_against_existing: bool = True,
) -> dict[str, Any] | None:
    """Publish the review summary plus inline comments, idempotently.

    Resilience strategy:
      1. Try to post the review with all inline comments at once.
      2. If GitHub rejects (e.g. one comment anchors to a line not in the
         diff's RIGHT side), retry with inline comments stripped - move the
         lost findings into the summary body instead of dropping the whole
         review.
      3. If that also fails, post a summary-only review so the user still
         sees the findings.

    Idempotency: each comment body carries a hidden fingerprint marker.
    Before posting we list existing review comments and skip fingerprints
    that have already appeared.
    """
    if not result.issues and not result.summary:
        log.info("no issues and no summary - skipping review post")
        return None

    seen_fps: set[str] = set()
    if dedup_against_existing:
        try:
            existing = await client.list_review_comments(pr.owner, pr.repo, pr.number)
            for c in existing:
                body = c.get("body") or ""
                seen_fps |= extract_fingerprints(body)
        except GitHubError as exc:
            log.warning("could not list existing comments for dedup: %s", exc)

    new_issues = [i for i in result.issues if i.fingerprint not in seen_fps]
    if not new_issues and not result.summary:
        log.info("all findings already posted - nothing new to publish")
        return None

    inline_issues = [i for i in new_issues if i.line >= 1 and i.file]
    summary_issues = [i for i in new_issues if i not in inline_issues]

    fps = [i.fingerprint for i in new_issues]
    summary = _compose_summary(result.summary, summary_issues)
    body = build_review_body(summary, fps)

    # First attempt: full review with inline comments.
    if inline_issues:
        comments = [to_inline_comment(i) for i in inline_issues]
        try:
            return await client.post_review(
                pr.owner,
                pr.repo,
                pr.number,
                body=body,
                event="COMMENT",
                comments=comments,
                commit_id=head_sha or None,
            )
        except GitHubError as exc:
            log.warning(
                "full review rejected (%s); retrying summary-only with inline "
                "findings folded into the summary",
                exc,
            )
            # Fold inline findings into the summary and retry without comments.
            summary = _compose_summary(result.summary, new_issues)
            body = build_review_body(summary, fps)

    # Fallback: summary-only review.
    try:
        return await client.post_review(
            pr.owner,
            pr.repo,
            pr.number,
            body=body,
            event="COMMENT",
            comments=None,
            commit_id=head_sha or None,
        )
    except GitHubError as exc:
        log.error("summary-only review also failed: %s", exc)
        raise


def _compose_summary(base_summary: str, fallback_issues: list[ReviewIssue]) -> str:
    """Append inline-failed findings to the summary as a bulleted list."""
    if not fallback_issues:
        return base_summary
    lines = [base_summary, "", "### Findings (could not be posted inline)", ""]
    for i in fallback_issues:
        lines.append(
            f"- **[{i.severity.value.upper()}] {i.title}** - `{i.file}:{i.line}` "
            f"(confidence: {i.confidence:.2f})"
        )
    return "\n".join(lines)


__all__ = [
    "PrRef",
    "build_review_body",
    "extract_fingerprints",
    "parse_pr_ref",
    "publish_review",
    "to_inline_comment",
]
