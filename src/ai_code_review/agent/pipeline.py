from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..config import Settings
from ..enums import Severity
from ..errors import GitError, ProviderError, ProviderTimeout, ReviewError
from ..models import FileDiff, FileFailure, ReviewIssue, ReviewResult
from ..prompts import SYSTEM_PROMPT, user_prompt
from ..providers import LLMProvider
from .chunking import chunk_file
from .context import gather_context
from .filters import anchor_to_diff, dedup, filter_by_confidence, sort_by_severity
from .parser import ParseError, parse_issues
from .types import RepoLike

log = logging.getLogger("ai_code_review.agent")

# How many times we retry a single (LLM call + parse) cycle when the model
# returns malformed JSON. The repair instruction is appended once, not grown
# on every retry, to keep the prompt bounded.
_MAX_PARSE_RETRIES = 1
_REPAIR_NOTE = (
    "Your previous response was not valid JSON matching the schema. "
    "Return ONLY a JSON object with an `issues` array, no prose, no fences."
)

# When this fraction of eligible files failed to analyze, treat the whole
# review as a failure rather than a misleading "no issues".
_CRITICAL_FAILURE_RATIO = 0.5


@dataclass
class PipelineDeps:
    """Injectable dependencies. Tests substitute a fake provider here."""

    provider: LLMProvider
    settings: Settings
    repo: RepoLike | None = None
    head_ref: str = "HEAD"


class PipelineFailure(ReviewError):
    """Raised when a critical fraction of files failed to analyze."""


class ReviewPipeline:
    """Orchestrates the full review pipeline.

    Flow per file: chunk → context (from git tree) → LLM call with parse
    retry → validate → collect. Files run concurrently under a bounded
    semaphore; results are gathered in input order so output stays stable.
    """

    def __init__(self, deps: PipelineDeps) -> None:
        self.deps = deps

    async def run(self, files: list[FileDiff]) -> ReviewResult:
        if not files:
            return ReviewResult(summary="No reviewable changes.", total_changed_files=0)

        settings = self.deps.settings
        timeout = settings.review_timeout
        try:
            async with asyncio.timeout(timeout):
                return await self._run_with_budget(files)
        except TimeoutError as exc:
            raise ReviewError(
                f"review timed out after {timeout}s; partial results discarded"
            ) from exc

    async def _run_with_budget(self, files: list[FileDiff]) -> ReviewResult:
        settings = self.deps.settings
        max_files = settings.review_max_files

        total_changed = len(files)
        limited = total_changed > max_files
        limit_reason: str | None = None
        if limited:
            limit_reason = f"PR has {total_changed} reviewable files; capping at {max_files}"
            omitted = [f.path for f in files[max_files:]]
            files = files[:max_files]
        else:
            omitted = []

        semaphore = asyncio.Semaphore(max(1, settings.review_concurrency))
        tasks = [
            asyncio.create_task(self._review_file_bounded(semaphore, f), name=f.path) for f in files
        ]

        analyzed: list[str] = []
        failed: list[FileFailure] = []
        all_issues: list[ReviewIssue] = []
        per_file_results: list[tuple[int, list[ReviewIssue]]] = []

        try:
            for idx, raw_result in enumerate(await asyncio.gather(*tasks, return_exceptions=True)):
                path = files[idx].path
                if isinstance(raw_result, BaseException):
                    if isinstance(raw_result, (ProviderError, ProviderTimeout)):
                        failed.append(FileFailure(path=path, reason=f"provider: {raw_result}"))
                    elif isinstance(raw_result, ReviewError):
                        failed.append(FileFailure(path=path, reason=str(raw_result)))
                    else:
                        # An unexpected internal error is recorded as a
                        # failure rather than swallowed into "no issues".
                        log.exception("unexpected error reviewing %s", path, exc_info=raw_result)
                        failed.append(
                            FileFailure(
                                path=path,
                                reason=f"internal error: {type(raw_result).__name__}: {raw_result}",
                            )
                        )
                    continue
                issues, ok = raw_result
                if not ok:
                    failed.append(FileFailure(path=path, reason="analysis incomplete"))
                    continue
                analyzed.append(path)
                per_file_results.append((idx, issues))
        finally:
            # Cancel anything still pending if we exit early (e.g. via
            # asyncio.timeout). gather(return_exceptions=True) above already
            # returned, so this is mostly belt-and-braces.
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Stable order by original file index.
        per_file_results.sort(key=lambda x: x[0])
        for _, issues in per_file_results:
            all_issues.extend(issues)

        filtered = filter_by_confidence(all_issues, min_confidence=settings.review_min_confidence)
        filtered = anchor_to_diff(filtered, files)
        filtered = dedup(filtered)
        filtered = sort_by_severity(filtered)

        summary = _build_summary(
            issues=filtered,
            analyzed=analyzed,
            failed=failed,
            omitted=omitted,
            total_changed=total_changed,
            limited=limited,
        )

        result = ReviewResult(
            issues=filtered,
            analyzed_files=analyzed,
            skipped_files=[],
            failed_files=failed,
            omitted_files=omitted,
            total_changed_files=total_changed,
            summary=summary,
            limited=limited,
            limit_reason=limit_reason,
        )

        _maybe_raise_on_critical_failure(result, eligible=len(files))
        return result

    async def _review_file_bounded(
        self, semaphore: asyncio.Semaphore, file_diff: FileDiff
    ) -> tuple[list[ReviewIssue], bool]:
        async with semaphore:
            return await self._review_file(file_diff)

    async def _review_file(self, file_diff: FileDiff) -> tuple[list[ReviewIssue], bool]:
        """Review one file. Returns (issues, success).

        `success=False` means the file's analysis is incomplete (e.g. every
        chunk failed to parse) - distinct from "analyzed, found nothing".
        """
        chunks = chunk_file(file_diff)
        if not chunks:
            return [], True

        context_block = ""
        if self.deps.repo is not None:
            try:
                context_block = gather_context(
                    self.deps.repo, file_diff, self.deps.settings, head_ref=self.deps.head_ref
                )
            except GitError as exc:
                log.debug("context gathering failed for %s: %s", file_diff.path, exc)
            except Exception as exc:
                log.debug("context gathering raised for %s: %s", file_diff.path, exc)

        allowed_files = {file_diff.path}
        results: list[ReviewIssue] = []
        chunk_failures = 0
        for chunk in chunks:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt(chunk, context_block, file_path=file_diff.path),
                },
            ]
            try:
                issues = await self._call_with_parse_retry(messages, allowed_files=allowed_files)
            except (ProviderError, ProviderTimeout):
                # Provider failure bubbles up to the file-level handler so
                # it is recorded as a failure, not silently dropped.
                raise
            except ParseError as exc:
                log.warning("unparseable response for %s: %s", file_diff.path, exc)
                chunk_failures += 1
                continue
            results.extend(issues)

        # If every chunk failed to parse, the file's analysis is incomplete.
        success = chunk_failures < len(chunks)
        return results, success

    async def _call_with_parse_retry(
        self, messages: list[dict[str, str]], *, allowed_files: set[str]
    ) -> list[ReviewIssue]:
        """Call the LLM and parse the response in one retry loop.

        Retry is triggered ONLY by a parse failure (malformed JSON, invalid
        schema). Provider transport errors (4xx/5xx/timeout) are handled
        inside the provider's own retry loop - they bubble up here only when
        unrecoverable, and we do NOT retry them at this layer.
        """
        last_err: ParseError | None = None
        for attempt in range(_MAX_PARSE_RETRIES + 1):
            extra = _REPAIR_NOTE if attempt > 0 else None
            current = [*messages, {"role": "user", "content": extra}] if extra else messages
            raw = await self.deps.provider.complete(current)
            try:
                return parse_issues(raw, allowed_files=allowed_files)
            except ParseError as exc:
                last_err = exc
                if attempt < _MAX_PARSE_RETRIES:
                    continue
                raise
        if last_err is None:
            raise ParseError("parse retry exhausted without error")
        raise last_err


def _build_summary(
    *,
    issues: list[ReviewIssue],
    analyzed: list[str],
    failed: list[FileFailure],
    omitted: list[str],
    total_changed: int,
    limited: bool,
) -> str:
    parts: list[str] = []
    if not issues:
        parts.append("No high-confidence defects found")
    else:
        sev_counts = dict.fromkeys(Severity, 0)
        for i in issues:
            sev_counts[i.severity] += 1
        sev_str = ", ".join(f"{sev_counts[s]} {s.value}" for s in Severity if sev_counts[s] > 0)
        parts.append(f"Found {len(issues)} issue(s): {sev_str}")

    parts.append(f"analyzed {len(analyzed)} of {total_changed} file(s)")
    if failed:
        parts.append(f"{len(failed)} file(s) failed analysis")
    if omitted:
        parts.append(f"{len(omitted)} file(s) omitted by max-files cap")
    if limited:
        parts.append("review was LIMITED")

    if failed and not analyzed:
        # Don't let a total failure masquerade as a clean review.
        parts.append("WARNING: no files were analyzed successfully")
    return ". ".join(parts) + "."


def _maybe_raise_on_critical_failure(result: ReviewResult, *, eligible: int) -> None:
    if eligible == 0:
        return
    if not result.analyzed_files and result.failed_count >= max(
        1, int(eligible * _CRITICAL_FAILURE_RATIO)
    ):
        reasons = "; ".join(f"{f.path}: {f.reason}" for f in result.failed_files[:3])
        raise PipelineFailure(
            f"critical: {result.failed_count} of {eligible} file(s) failed analysis. "
            f"First reasons: {reasons}"
        )
