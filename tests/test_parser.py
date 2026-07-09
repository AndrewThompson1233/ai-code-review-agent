from __future__ import annotations

import pytest

from ai_code_review.agent.parser import ParseError, extract_json, parse_issues

_GOOD = """{
  "issues": [
    {
      "title": "SQL injection",
      "body": "User input concatenated into query.",
      "severity": "high",
      "confidence": 0.9,
      "file": "src/api.py",
      "line": 10,
      "category": "security"
    }
  ]
}"""


def test_extract_clean_json() -> None:
    assert extract_json('{"a":1}') == '{"a":1}'


def test_extract_fenced_json() -> None:
    raw = "```json\n" + _GOOD + "\n```"
    out = extract_json(raw)
    assert '"issues"' in out


def test_extract_embedded_json() -> None:
    raw = "Here is my review:\n" + _GOOD + "\nThanks!"
    out = extract_json(raw)
    assert out.startswith("{")


def test_extract_empty_raises() -> None:
    with pytest.raises(ParseError):
        extract_json("")
    with pytest.raises(ParseError):
        extract_json("no json here at all")


def test_parse_issues_valid() -> None:
    issues = parse_issues(_GOOD)
    assert len(issues) == 1
    assert issues[0].file == "src/api.py"
    assert issues[0].line == 10


def test_parse_issues_with_allowed_files_filter() -> None:
    issues = parse_issues(_GOOD, allowed_files={"src/other.py"})
    assert issues == []


def test_parse_issues_drops_malformed_keeps_rest() -> None:
    raw = """{
      "issues": [
        {"title":"valid issue","body":"b","severity":"low","confidence":0.7,"file":"a.py","line":1,"category":"bug"},
        {"title":"bad","body":"b","severity":"invalid","confidence":0.7,"file":"a.py","line":2,"category":"bug"},
        {"title":"bad2","body":"b","severity":"low","confidence":1.5,"file":"a.py","line":3,"category":"bug"}
      ]
    }"""
    issues = parse_issues(raw)
    assert len(issues) == 1
    assert issues[0].title == "valid issue"


def test_parse_issues_empty_issues_array() -> None:
    assert parse_issues('{"issues": []}') == []


def test_parse_issues_no_issues_key() -> None:
    assert parse_issues('{"foo": 1}') == []


def test_parse_issues_invalid_json_raises() -> None:
    with pytest.raises(ParseError):
        parse_issues("not json at all")


def test_parse_issues_issues_not_a_list_raises() -> None:
    with pytest.raises(ParseError):
        parse_issues('{"issues": "string"}')


def test_parse_issues_empty_raises() -> None:
    with pytest.raises(ParseError):
        parse_issues("")


def test_parse_issues_with_fenced_markdown() -> None:
    raw = "```json\n" + _GOOD + "\n```"
    issues = parse_issues(raw)
    assert len(issues) == 1


def test_parse_issues_drops_path_traversal() -> None:
    raw = """{
      "issues": [
        {"title":"valid issue","body":"b","severity":"low","confidence":0.7,
         "file":"../../etc/passwd","line":1,"category":"bug"}
      ]
    }"""
    issues = parse_issues(raw)
    assert issues == []


def test_parse_issues_handles_html_error_page() -> None:
    # Provider returns an HTML error page instead of JSON.
    raw = "<html><body>502 Bad Gateway</body></html>"
    with pytest.raises(ParseError):
        parse_issues(raw)


def test_parse_issues_handles_multiple_json_objects() -> None:
    # Some models emit two JSON objects back-to-back. We take the first.
    raw = '{"issues": []}{"issues": []}'
    issues = parse_issues(raw)
    assert issues == []


def test_parse_issues_handles_trailing_text() -> None:
    raw = _GOOD + "\n\nLet me know if you need anything else."
    issues = parse_issues(raw)
    assert len(issues) == 1
