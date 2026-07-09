from __future__ import annotations

from ..enums import SEVERITY_RANK
from ..models import FileDiff, ReviewIssue


def filter_by_confidence(issues: list[ReviewIssue], *, min_confidence: float) -> list[ReviewIssue]:
    """Drop findings below the configured confidence threshold."""
    return [i for i in issues if i.confidence >= min_confidence]


def dedup(issues: list[ReviewIssue]) -> list[ReviewIssue]:
    """Collapse near-duplicate findings using the issue fingerprint.

    Two findings with the same file/line/category/title-key are treated as the
    same defect and only the higher-confidence one is kept.
    """
    seen: dict[str, ReviewIssue] = {}
    for issue in issues:
        existing = seen.get(issue.fingerprint)
        if existing is None or issue.confidence > existing.confidence:
            seen[issue.fingerprint] = issue
    return list(seen.values())


def anchor_to_diff(issues: list[ReviewIssue], files: list[FileDiff]) -> list[ReviewIssue]:
    """Drop findings whose line doesn't fall on (or right next to) an added line.

    Without this filter the LLM happily reports line numbers that exist in the
    file but were never touched by the PR — those are useless noise for an
    inline GitHub review.
    """
    by_path: dict[str, set[int]] = {f.path: f.added_lines_set for f in files}
    out: list[ReviewIssue] = []
    for issue in issues:
        added = by_path.get(issue.file)
        if added is None:
            continue
        if issue.matches_line(added):
            out.append(issue)
    return out


def sort_by_severity(issues: list[ReviewIssue]) -> list[ReviewIssue]:
    """Stable sort by severity then confidence (highest first)."""
    return sorted(
        issues,
        key=lambda i: (SEVERITY_RANK[i.severity], -i.confidence),
    )
