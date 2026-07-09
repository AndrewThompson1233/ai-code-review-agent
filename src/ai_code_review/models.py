from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import Category, Severity

_TITLE_MAX = 120
_BODY_MAX = 4000
_SUGGESTION_MAX = 4000
_FILE_MAX = 512


def _normalize_path(path: str) -> str:
    cleaned = path.strip().strip('"').strip("'")
    # Reject absolute paths, drive letters and traversal segments before we ever
    # pass the value to a filesystem operation.
    if not cleaned or "\x00" in cleaned:
        raise ValueError("empty or null-containing path")
    if cleaned[0] in ("/", "\\"):
        raise ValueError("absolute paths are not allowed")
    if re.match(r"^[A-Za-z]:", cleaned):
        raise ValueError("drive-letter paths are not allowed")
    parts = PurePosixPath(cleaned).parts
    if ".." in parts or "." in parts:
        raise ValueError("path traversal segments are not allowed")
    return cleaned


class ReviewIssue(BaseModel):
    """A single finding produced by the agent or parsed from LLM output.

    Fields are deliberately strict: anything that fails validation becomes a
    parse error rather than silently dropped, so callers can decide whether to
    retry or surface the failure.
    """

    title: str = Field(..., min_length=3, max_length=_TITLE_MAX)
    body: str = Field(..., min_length=1, max_length=_BODY_MAX)
    severity: Severity
    confidence: float = Field(..., ge=0.0, le=1.0)
    file: str = Field(..., min_length=1, max_length=_FILE_MAX)
    line: int = Field(..., ge=1, le=2_000_000)
    category: Category
    suggestion: str | None = Field(default=None, max_length=_SUGGESTION_MAX)

    # Stable hash used for dedup and GitHub idempotency. Computed in
    # model_post_init so callers never need to set it themselves.
    fingerprint: str = Field(default="", repr=False)

    @field_validator("title", "body")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty after strip")
        return v

    @field_validator("file")
    @classmethod
    def _validate_file(cls, v: str) -> str:
        return _normalize_path(v)

    @model_validator(mode="after")
    def _compute_fingerprint(self) -> ReviewIssue:
        if not self.fingerprint:
            key = f"{self.file}:{self.line}:{self.category.value}:{self.title.lower()}"
            self.fingerprint = _short_hash(key)
        return self

    def matches_line(self, added_lines: set[int]) -> bool:
        """Return True if the issue's line falls on an added line of its file.

        Accepts an off-by-one window: a finding on the line immediately above
        or below an added hunk is still considered anchored, because multi-line
        hunks frequently push the real defect onto a neighbouring line.
        """
        if not added_lines:
            return True
        if self.line in added_lines:
            return True
        return any(abs(self.line - ln) == 1 for ln in added_lines)


def _short_hash(text: str) -> str:
    # Fingerprinting, not cryptography: a 12-char sha256 prefix is enough to
    # dedup findings without colliding in practice. Truncated hash of a
    # file:line:category:title key is not a security primitive.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class FileDiff(BaseModel):
    """A parsed file change extracted from a unified diff."""

    path: str
    old_path: str | None = None
    added: int = 0
    removed: int = 0
    added_lines: list[int] = Field(default_factory=list)
    is_binary: bool = False
    is_new: bool = False
    is_deleted: bool = False
    patch: str = ""

    @field_validator("path", "old_path")
    @classmethod
    def _validate_path(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _normalize_path(v)

    @property
    def added_lines_set(self) -> set[int]:
        return set(self.added_lines)


class FileFailure(BaseModel):
    """A file the pipeline could not analyze, with a reason.

    Distinct from `skipped_files` (intentionally skipped: binary, lock, etc.)
    so callers can tell "we chose not to" from "we tried and failed".
    """

    path: str
    reason: str


class ReviewResult(BaseModel):
    """Final structured output produced by the agent pipeline.

    The bookkeeping fields (analyzed / skipped / failed / total_changed)
    exist so a caller can distinguish four outcomes that all look like "no
    issues" otherwise:

      - the diff was empty
      - all files were skipped (binary / lockfile)
      - all files failed to analyze (provider down, parse error)
      - analysis actually succeeded and found nothing

    Only the last is a clean "code is clean" signal.
    """

    issues: list[ReviewIssue] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    analyzed_files: list[str] = Field(default_factory=list)
    failed_files: list[FileFailure] = Field(default_factory=list)
    omitted_files: list[str] = Field(default_factory=list)
    total_changed_files: int = 0
    summary: str = ""
    limited: bool = False
    limit_reason: str | None = None

    @property
    def total(self) -> int:
        return len(self.issues)

    @property
    def failed_count(self) -> int:
        return len(self.failed_files)

    @property
    def has_failures(self) -> bool:
        return bool(self.failed_files)

    @property
    def all_failed(self) -> bool:
        """True when every eligible file failed (so 0 issues is a lie)."""
        return self.failed_count > 0 and not self.analyzed_files
