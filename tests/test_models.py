from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_code_review.enums import Category, Severity
from ai_code_review.models import ReviewIssue


def _make(**overrides: object) -> ReviewIssue:
    base: dict[str, object] = {
        "title": "Possible NoneType dereference",
        "body": "x may be None when foo() returns 0.",
        "severity": "high",
        "confidence": 0.85,
        "file": "src/app.py",
        "line": 42,
        "category": "bug",
    }
    base.update(overrides)
    return ReviewIssue(**base)  # type: ignore[arg-type]


def test_valid_issue_has_fingerprint() -> None:
    i = _make()
    assert i.fingerprint
    assert len(i.fingerprint) == 12


def test_path_traversal_rejected() -> None:
    with pytest.raises(ValidationError):
        _make(file="../../etc/passwd")
    with pytest.raises(ValidationError):
        _make(file="/etc/passwd")
    with pytest.raises(ValidationError):
        _make(file="C:/windows/system32")


def test_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        _make(confidence=1.5)
    with pytest.raises(ValidationError):
        _make(confidence=-0.1)


def test_line_bounds() -> None:
    with pytest.raises(ValidationError):
        _make(line=0)
    with pytest.raises(ValidationError):
        _make(line=-5)


def test_severity_enum() -> None:
    i = _make(severity="critical")
    assert i.severity is Severity.CRITICAL


def test_category_enum() -> None:
    i = _make(category="security")
    assert i.category is Category.SECURITY


def test_title_too_long_rejected() -> None:
    with pytest.raises(ValidationError):
        _make(title="x" * 200)


def test_empty_body_rejected() -> None:
    with pytest.raises(ValidationError):
        _make(body="   ")


def test_matches_line_exact() -> None:
    i = _make(line=5)
    assert i.matches_line({5})


def test_matches_line_neighbor() -> None:
    i = _make(line=4)
    assert i.matches_line({5})
    assert i.matches_line({3})
    # Two lines away - not anchored.
    assert not i.matches_line({7})


def test_matches_line_empty_set_passes() -> None:
    i = _make(line=5)
    assert i.matches_line(set())


def test_dedup_fingerprint_stable() -> None:
    a = _make()
    b = _make()
    assert a.fingerprint == b.fingerprint
