from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from ..errors import ReviewError
from ..models import ReviewIssue
from ..redact import redact

_FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
_BARE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ParseError(ReviewError):
    """The LLM response could not be parsed into the expected schema."""


def extract_json(raw: str) -> str:
    """Pull a JSON object out of an LLM response that may include prose.

    Handles, in order:
      1. A single fenced ```json ... ``` block (returns its contents).
      2. Already-clean JSON (starts with `{`).
      3. The first valid JSON object found anywhere in the text, using
         `json.JSONDecoder.raw_decode` so nested braces and adjacent
         objects are handled correctly.

    Raises ParseError if nothing plausible is found.
    """
    if raw is None:
        raise ParseError("empty LLM response")
    text = raw.strip()
    if not text:
        raise ParseError("empty LLM response")

    if m := _FENCE.match(text):
        return m.group(1).strip()
    if m := _BARE_FENCE.search(text):
        return m.group(1).strip()

    # Find the first `{` and try to raw_decode from there. This handles
    # embedded JSON in prose AND adjacent JSON objects correctly.
    start = text.find("{")
    if start < 0:
        raise ParseError(f"no JSON object found in response: {redact(text)[:200]}")
    decoder = json.JSONDecoder()
    try:
        _data, end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc.msg}") from exc
    return text[start : start + end]


def parse_issues(raw: str, *, allowed_files: set[str] | None = None) -> list[ReviewIssue]:
    """Parse an LLM response into validated ReviewIssue objects.

    Validation is strict at the response level (must be a JSON object with an
    `issues` list) but lenient at the item level: a single malformed issue is
    dropped while the rest are kept, because one hallucinated line number
    should not waste a whole LLM call. Raises ParseError only when the
    response itself is not JSON or has the wrong top-level shape.
    """
    payload = extract_json(raw)
    try:
        data: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ParseError(f"expected JSON object, got {type(data).__name__}")

    raw_issues = data.get("issues")
    if raw_issues is None:
        # Some models wrap the array differently or forget the key.
        if isinstance(data, list):
            raw_issues = data
        else:
            return []
    if not isinstance(raw_issues, list):
        raise ParseError("`issues` must be a list")

    out: list[ReviewIssue] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        try:
            issue = ReviewIssue(**item)
        except (ValidationError, ValueError):
            # Drop the bad issue but keep going. Raising here would let one
            # hallucinated line number discard the entire batch.
            continue
        if allowed_files is not None and issue.file not in allowed_files:
            continue
        out.append(issue)
    return out


__all__ = ["ParseError", "extract_json", "parse_issues"]
