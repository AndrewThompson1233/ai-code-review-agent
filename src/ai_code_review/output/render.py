from __future__ import annotations

import json
from typing import Any

from ..models import ReviewResult


def render_text(result: ReviewResult) -> str:
    """Render a ReviewResult as plain text (no Rich markup)."""
    lines: list[str] = [f"Summary: {result.summary}"]
    lines.append(
        f"Files: {result.total_changed_files} changed, "
        f"{len(result.analyzed_files)} analyzed, "
        f"{len(result.failed_files)} failed, "
        f"{len(result.skipped_files)} skipped, "
        f"{len(result.omitted_files)} omitted"
    )
    if result.limited and result.limit_reason:
        lines.append(f"LIMITED: {result.limit_reason}")
    lines.append("")
    if not result.issues:
        if result.failed_files and not result.analyzed_files:
            lines.append("WARNING: no files were analyzed successfully — review is incomplete.")
        else:
            lines.append("No issues above the confidence threshold.")
        return "\n".join(lines)

    for i, issue in enumerate(result.issues, 1):
        sev = issue.severity.value.upper()
        lines.append(f"[{i}] {sev} [{issue.category.value}] {issue.title}")
        lines.append(f"    file: {issue.file}:{issue.line}")
        lines.append(f"    confidence: {issue.confidence:.2f}")
        lines.append(f"    {issue.body}")
        if issue.suggestion:
            lines.append(f"    suggestion: {issue.suggestion}")
        lines.append("")
    return "\n".join(lines)


def render_json(result: ReviewResult) -> str:
    """Stable JSON serialization for automation."""
    payload: dict[str, Any] = {
        "summary": result.summary,
        "limited": result.limited,
        "limit_reason": result.limit_reason,
        "total_changed_files": result.total_changed_files,
        "analyzed_files": sorted(result.analyzed_files),
        "skipped_files": sorted(result.skipped_files),
        "omitted_files": sorted(result.omitted_files),
        "failed_files": sorted(
            [{"path": f.path, "reason": f.reason} for f in result.failed_files],
            key=lambda x: x["path"],
        ),
        "issues": [
            {
                "title": i.title,
                "body": i.body,
                "severity": i.severity.value,
                "confidence": round(i.confidence, 4),
                "file": i.file,
                "line": i.line,
                "category": i.category.value,
                "suggestion": i.suggestion,
                "fingerprint": i.fingerprint,
            }
            for i in result.issues
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)
