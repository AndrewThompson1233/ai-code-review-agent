from __future__ import annotations

import asyncio
import json
import re
from types import TracebackType
from typing import Any

import httpx

from ..config import Settings
from ..errors import GitHubError
from ..providers.http_util import parse_retry_after
from ..redact import redact

_USER_AGENT = (
    "ai-code-review-agent/0.1 (+https://github.com/assasino15555-star/ai-code-review-agent)"
)
_HIDDEN_MARKER_PREFIX = "<!-- ai-code-review-agent:fp:"


class GitHubClient:
    """Thin async wrapper around the GitHub REST API.

    Only the endpoints the agent actually needs are implemented. Tokens are
    sent via the Authorization header and never logged; every error path runs
    the response body through `redact` in case the API echoes the header back.
    """

    def __init__(self, settings: Settings, *, token: str | None = None) -> None:
        self.settings = settings
        self.token = token or settings.require_github_token()
        self.base_url = settings.github_api_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> GitHubClient:
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
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": _USER_AGENT,
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        client = await self._ensure_client()
        last_exc: GitHubError | None = None
        for attempt in range(3):
            try:
                resp = await client.request(method, path, json=json_body, params=params)
            except httpx.HTTPError as exc:
                last_exc = GitHubError(f"github transport error: {exc}")
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise last_exc from exc

            if resp.status_code in (403, 429) or resp.status_code >= 500:
                retry_after = parse_retry_after(resp.headers.get("retry-after"))
                last_exc = GitHubError(f"github HTTP {resp.status_code}: {redact(resp.text)[:200]}")
                if attempt < 2:
                    delay = retry_after if retry_after is not None else (2.0 * (attempt + 1))
                    await asyncio.sleep(delay)
                    continue
                raise last_exc

            if resp.status_code >= 400:
                raise GitHubError(
                    f"github HTTP {resp.status_code} on {method} {path}: {redact(resp.text)[:200]}"
                )

            if resp.status_code == 204 or not resp.content:
                return None
            try:
                return resp.json()
            except json.JSONDecodeError as exc:
                raise GitHubError(f"github: invalid JSON response: {exc}") from exc

        if last_exc is None:
            raise GitHubError("github: retries exhausted without error")
        raise last_exc

    async def get_pr(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        result = await self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")
        return result if isinstance(result, dict) else {}

    async def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        all_files: list[dict[str, Any]] = []
        page = 1
        while True:
            batch = await self._request(
                "GET",
                f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
                params={"per_page": 100, "page": page},
            )
            if not isinstance(batch, list) or not batch:
                break
            all_files.extend(batch)
            if len(batch) < 100:
                break
            page += 1
            if page > 20:  # hard cap at 2000 files
                break
        return all_files

    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetch the unified diff for a PR using the Accept: text/plain hack."""
        client = await self._ensure_client()
        resp = await client.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        if resp.status_code >= 400:
            raise GitHubError(
                f"github HTTP {resp.status_code} fetching diff: {redact(resp.text)[:200]}"
            )
        return resp.text

    async def list_reviews(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        return await _paginate(self._request, f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews")

    async def list_review_comments(
        self, owner: str, repo: str, pr_number: int
    ) -> list[dict[str, Any]]:
        return await _paginate(self._request, f"/repos/{owner}/{repo}/pulls/{pr_number}/comments")

    async def post_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        *,
        body: str,
        event: str = "COMMENT",
        comments: list[dict[str, Any]] | None = None,
        commit_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"body": body, "event": event}
        if comments:
            payload["comments"] = comments
        if commit_id:
            payload["commit_id"] = commit_id
        result = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
            json_body=payload,
        )
        return result if isinstance(result, dict) else {}


async def _paginate(
    request_fn: Any, path: str, *, per_page: int = 100, max_pages: int = 10
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = await request_fn("GET", path, params={"per_page": per_page, "page": page})
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
        if page > max_pages:
            break
    return out


def build_review_body(summary: str, fingerprints: list[str]) -> str:
    """Compose the review summary, embedding hidden fingerprint markers.

    The marker is what makes the review idempotent: on a re-run we read
    existing comments and skip fingerprints that have already been posted.
    """
    lines = ["## AI Code Review", "", summary, ""]
    if fingerprints:
        joined = ",".join(fingerprints)
        lines.append(f"{_HIDDEN_MARKER_PREFIX}{joined} -->")
    return "\n".join(lines)


def extract_fingerprints(comment_body: str) -> set[str]:
    """Recover fingerprints previously embedded by `build_review_body`."""
    pattern = re.compile(re.escape(_HIDDEN_MARKER_PREFIX) + r"([^>]+)-->")
    out: set[str] = set()
    for m in pattern.finditer(comment_body):
        for raw_fp in m.group(1).split(","):
            fp = raw_fp.strip()
            if fp:
                out.add(fp)
    return out
