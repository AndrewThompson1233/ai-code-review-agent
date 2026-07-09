from __future__ import annotations

from ai_code_review.agent.filters import (
    anchor_to_diff,
    dedup,
    filter_by_confidence,
    sort_by_severity,
)
from ai_code_review.enums import Severity
from ai_code_review.models import FileDiff, ReviewIssue


def _issue(file: str, line: int, conf: float, sev: str = "high") -> ReviewIssue:
    return ReviewIssue(
        title=f"issue at {file}:{line}",
        body="something is wrong",
        severity=Severity(sev),
        confidence=conf,
        file=file,
        line=line,
        category="bug",
    )


def test_filter_by_confidence_drops_low() -> None:
    issues = [
        _issue("a.py", 1, 0.9),
        _issue("a.py", 2, 0.4),
        _issue("a.py", 3, 0.7),
    ]
    out = filter_by_confidence(issues, min_confidence=0.7)
    assert {i.line for i in out} == {1, 3}


def test_dedup_keeps_higher_confidence() -> None:
    a = _issue("a.py", 1, 0.7)
    b = _issue("a.py", 1, 0.9)
    out = dedup([a, b])
    assert len(out) == 1
    assert out[0].confidence == 0.9


def test_dedup_distinct_lines_kept() -> None:
    out = dedup([_issue("a.py", 1, 0.8), _issue("a.py", 2, 0.8)])
    assert len(out) == 2


def test_dedup_distinct_categories_kept() -> None:
    a = ReviewIssue(
        title="sql injection",
        body="b",
        severity=Severity.HIGH,
        confidence=0.8,
        file="a.py",
        line=1,
        category="security",
    )
    b = ReviewIssue(
        title="sql injection",
        body="b",
        severity=Severity.HIGH,
        confidence=0.8,
        file="a.py",
        line=1,
        category="bug",
    )
    out = dedup([a, b])
    assert len(out) == 2


def test_anchor_drops_offline_lines() -> None:
    files = [FileDiff(path="a.py", added_lines=[5, 6, 7])]
    issues = [
        _issue("a.py", 5, 0.9),
        _issue("a.py", 4, 0.9),  # neighbor of 5 — kept
        _issue("a.py", 100, 0.9),  # far away — dropped
        _issue("missing.py", 1, 0.9),  # not in diff — dropped
    ]
    out = anchor_to_diff(issues, files)
    assert {i.line for i in out} == {4, 5}


def test_sort_by_severity() -> None:
    issues = [
        _issue("a.py", 1, 0.5, sev="low"),
        _issue("a.py", 2, 0.95, sev="critical"),
        _issue("a.py", 3, 0.8, sev="high"),
    ]
    out = sort_by_severity(issues)
    assert [i.severity for i in out] == [Severity.CRITICAL, Severity.HIGH, Severity.LOW]
