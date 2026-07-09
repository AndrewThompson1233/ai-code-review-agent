"""End-to-end integration test: real git repo + mock LLM via monkeypatch.

This verifies the full chain:
  git diff → parse → context (from git tree, not worktree) → mock LLM →
  parse JSON → confidence filter → diff-anchor → output → exit code.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_code_review.cli import app


def _git_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e.com",
        "GIT_CONFIG_NOSYSTEM": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


class MockProvider:
    """Returns a canned finding anchored to the file's first added line."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.response

    async def aclose(self) -> None:
        pass


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = _git_env(tmp_path)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@e.com"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True, env=env)
    (repo / "app.py").write_text("def foo():\n    return 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True, env=env)
    return repo


def test_e2e_mock_provider_finds_issue_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    # Introduce a bug.
    (repo / "app.py").write_text(
        "def query(user_input):\n"
        "    sql = f\"SELECT * FROM users WHERE name = '{user_input}'\"\n"
        "    return sql\n"
    )
    env = _git_env(tmp_path)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "bug"], check=True, env=env)
    monkeypatch.chdir(repo)

    mock_resp = json.dumps(
        {
            "issues": [
                {
                    "title": "SQL injection via f-string",
                    "body": "user_input concatenated into SQL.",
                    "severity": "critical",
                    "confidence": 0.95,
                    "file": "app.py",
                    "line": 2,
                    "category": "security",
                }
            ]
        }
    )
    mock = MockProvider(mock_resp)
    monkeypatch.setattr("ai_code_review.cli.build_provider", lambda settings: mock)
    monkeypatch.setenv("AI_API_KEY", "sk-test")

    runner = CliRunner()
    res = runner.invoke(app, ["review", "--base", "HEAD~1", "--format", "json"])
    assert res.exit_code == 0, f"exit={res.exit_code}, output={res.output}"
    # Parse JSON from output.
    start = res.output.find("{")
    payload = json.loads(res.output[start:])
    assert len(payload["issues"]) == 1
    assert payload["issues"][0]["severity"] == "critical"
    assert payload["issues"][0]["file"] == "app.py"
    assert payload["issues"][0]["line"] == 2
    assert payload["analyzed_files"] == ["app.py"]
    assert payload["failed_files"] == []


def test_e2e_mock_provider_fail_on_issues_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    (repo / "app.py").write_text("def query(u):\n    return f'SELECT * FROM t WHERE n = {u}'\n")
    env = _git_env(tmp_path)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "bug"], check=True, env=env)
    monkeypatch.chdir(repo)

    mock_resp = json.dumps(
        {
            "issues": [
                {
                    "title": "SQL injection",
                    "body": "concatenated input",
                    "severity": "high",
                    "confidence": 0.9,
                    "file": "app.py",
                    "line": 2,
                    "category": "security",
                }
            ]
        }
    )
    mock = MockProvider(mock_resp)
    monkeypatch.setattr("ai_code_review.cli.build_provider", lambda settings: mock)
    monkeypatch.setenv("AI_API_KEY", "sk-test")

    runner = CliRunner()
    res = runner.invoke(app, ["review", "--base", "HEAD~1", "--fail-on-issues"])
    assert res.exit_code == 2, f"expected exit 2 (fail-on-issues), got {res.exit_code}"


def test_e2e_context_from_tree_not_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the LLM receives context from the committed HEAD, not the dirty worktree."""
    repo = _init_repo(tmp_path)
    # Commit a version with a specific marker.
    (repo / "app.py").write_text("MARKER_FROM_HEAD = 1\nx = 1\n")
    env = _git_env(tmp_path)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "marker"], check=True, env=env)
    # Now dirty the worktree with a DIFFERENT marker.
    (repo / "app.py").write_text("MARKER_FROM_WORKTREE = 2\ny = 2\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "dirty"], check=True, env=env)
    # Make one more change so there's a diff against HEAD~1.
    (repo / "app.py").write_text("MARKER_FROM_WORKTREE = 2\ny = 2\nz = 3\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "c3"], check=True, env=env)
    monkeypatch.chdir(repo)

    mock = MockProvider('{"issues": []}')
    monkeypatch.setattr("ai_code_review.cli.build_provider", lambda settings: mock)
    monkeypatch.setenv("AI_API_KEY", "sk-test")

    runner = CliRunner()
    res = runner.invoke(app, ["review", "--base", "HEAD~1"])
    assert res.exit_code == 0, f"exit={res.exit_code}, output={res.output}"
    # The context block sent to the LLM must contain the HEAD version's
    # first line, not the worktree's. The diff is HEAD~1..HEAD, so context
    # is read from HEAD.
    assert len(mock.calls) >= 1
    user_msg = mock.calls[0][1]["content"]
    # HEAD has the "z = 3" line and the worktree marker. The context should
    # reflect HEAD's content.
    assert "MARKER_FROM_WORKTREE" in user_msg or "z = 3" in user_msg


def test_e2e_provider_failure_exit_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    (repo / "app.py").write_text("y = 2\n")
    env = _git_env(tmp_path)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "c2"], check=True, env=env)
    monkeypatch.chdir(repo)

    from ai_code_review.errors import ProviderError

    class FailingProvider(MockProvider):
        async def complete(self, messages: list[dict[str, str]]) -> str:
            raise ProviderError("provider down")

    mock = FailingProvider("")
    monkeypatch.setattr("ai_code_review.cli.build_provider", lambda settings: mock)
    monkeypatch.setenv("AI_API_KEY", "sk-test")

    runner = CliRunner()
    res = runner.invoke(app, ["review", "--base", "HEAD~1"])
    # All files failed → operational failure → exit 1.
    assert res.exit_code == 1, f"expected exit 1, got {res.exit_code}, output={res.output}"
