from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_code_review.cli import app

runner = CliRunner()


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


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    env = _git_env(tmp_path)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@e.com"],
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "t"],
        check=True,
        env=env,
    )
    (repo / "f.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "init"],
        check=True,
        env=env,
    )
    return repo


def test_version_command() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "0.1.0" in res.output


def test_help_command() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "review" in res.output
    assert "version" in res.output


def test_review_help_lists_options() -> None:
    res = runner.invoke(app, ["review", "--help"], color=False)
    assert res.exit_code == 0

    for opt in [
        "--base",
        "--staged",
        "--format",
        "--pr",
        "--fail-on-issues",
        "--concurrency",
        "--timeout",
    ]:
        assert opt in res.output, f"missing {opt}"


def test_review_outside_git_repo_fails_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / ".git").exists()
    monkeypatch.setenv("AI_API_KEY", "sk-test")

    res = runner.invoke(app, ["review", "--staged"])

    assert res.exit_code == 1
    assert "git" in res.output.lower() or "error" in res.output.lower()


def test_review_no_changes_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("AI_API_KEY", "sk-test")

    res = runner.invoke(app, ["review", "--staged"])

    assert res.exit_code == 0
    assert "No changes" in res.output or "No reviewable" in res.output


def test_review_missing_api_key_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo(tmp_path)
    (repo / "f.py").write_text("y = 2\n")
    env = _git_env(tmp_path)

    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "c2"],
        check=True,
        env=env,
    )

    monkeypatch.chdir(repo)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    res = runner.invoke(app, ["review", "--base", "HEAD~1"])

    assert res.exit_code == 1


def test_review_local_provider_no_server_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo(tmp_path)
    (repo / "f.py").write_text("y = 2\n")
    env = _git_env(tmp_path)

    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "c2"],
        check=True,
        env=env,
    )

    monkeypatch.chdir(repo)
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("AI_MODEL", "test")
    monkeypatch.setenv("AI_BASE_URL", "http://localhost:9999/v1")

    res = runner.invoke(
        app,
        ["review", "--base", "HEAD~1", "--format", "json"],
    )

    assert res.exit_code == 1


def test_review_json_output_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json as _json

    repo = _make_repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("AI_BASE_URL", "http://localhost:9999/v1")

    res = runner.invoke(
        app,
        ["review", "--staged", "--format", "json"],
    )

    assert res.exit_code == 0

    out = res.output
    start = out.find("{")
    assert start >= 0, f"no JSON in output: {out!r}"

    payload = _json.loads(out[start:])

    assert "issues" in payload
    assert "failed_files" in payload
    assert "analyzed_files" in payload
    assert "omitted_files" in payload
    assert "total_changed_files" in payload


def test_parse_test_subcommand_reads_stdin() -> None:
    payload = (
        '{"issues":[{"title":"test issue","body":"b","severity":"low",'
        '"confidence":0.7,"file":"a.py","line":1,"category":"bug"}]}'
    )

    res = runner.invoke(app, ["parse-test", "-"], input=payload)

    assert res.exit_code == 0
    assert "test issue" in res.output


def test_parse_test_invalid_json_exits_one() -> None:
    res = runner.invoke(app, ["parse-test", "not json at all"])
    assert res.exit_code == 1