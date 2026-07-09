"""Shared HTTP helpers used by every provider.

`parse_retry_after` honours both the numeric (seconds) and the HTTP-date
forms of the header, per RFC 7231 §7.1.3. Some providers (notably Anthropic
on burst limits) return an HTTP date.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime


def parse_retry_after(value: str | None) -> float | None:
    """Return the delay in seconds implied by a Retry-After header.

    Returns None when the header is absent or unparseable. Negative or
    implausibly large values are clamped to None so a buggy server can't
    stall the pipeline for hours.
    """
    if not value:
        return None
    value = value.strip()
    secs: float | None
    try:
        secs = float(value)
    except ValueError:
        secs = _from_http_date(value)
    if secs is None or secs < 0:
        return None
    return min(secs, 300.0)


def _from_http_date(value: str) -> float | None:
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = (dt - datetime.now(UTC)).total_seconds()
    return delta
