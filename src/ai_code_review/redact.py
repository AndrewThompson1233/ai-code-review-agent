"""Redaction helpers for logs and error surfaces.

Anything that might contain an Authorization header, an API key or a token is
funnelled through `redact` before being formatted into a log message or raised
as an exception. The patterns are deliberately conservative: false positives
(redacting the word "key" in prose) are far cheaper than false negatives.
"""

from __future__ import annotations

import re

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(authorization)\s*[:=]\s*(\S+)", re.IGNORECASE), r"\1: <redacted>"),
    (re.compile(r"(?i)(bearer)\s+(\S+)", re.IGNORECASE), r"\1 <redacted>"),
    (re.compile(r"(?i)(sk-[A-Za-z0-9_\-]{10,})"), "<redacted>"),
    (re.compile(r"(?i)(x-api-key)\s*[:=]\s*(\S+)", re.IGNORECASE), r"\1: <redacted>"),
    (re.compile(r"(?i)(token)\s*[:=]\s*([A-Za-z0-9_\-\.]{8,})", re.IGNORECASE), r"\1: <redacted>"),
    (re.compile(r"(?i)(password)\s*[:=]\s*(\S+)", re.IGNORECASE), r"\1: <redacted>"),
    (re.compile(r"(ghp_[A-Za-z0-9]{10,})"), "<redacted>"),
    (re.compile(r"(github_pat_[A-Za-z0-9_]{10,})"), "<redacted>"),
]


def redact(text: str) -> str:
    """Return a copy of *text* with secret-looking substrings masked."""
    if not text:
        return text
    out = text
    for pat, repl in _SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out
