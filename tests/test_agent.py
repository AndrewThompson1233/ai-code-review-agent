from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import TracebackType

import pytest

from ai_code_review.agent import PipelineDeps, ReviewPipeline
from ai_code_review.config import Settings
from ai_code_review.git import GitRepo, parse_diff
from ai_code_review.models import FileDiff


class FakeProvider:
    """Records calls and returns a canned response per file."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return '{"issues": []}'

    async def aclose(self) -> None:
        pass

    async def __aenter__(self) -> FakeProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "ai_provider": "openai",
        "ai_api_key": "sk-test",
        "ai_model": "gpt-4o-mini",
        "ai_retries": 0,
        "review_min_confidence": 0.6,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _file(path: str, added_lines: list[int], patch: str = "") -> FileDiff:
    return FileDiff(
        path=path, added_lines=added_lines, patch=patch or f"diff --git a/{path} b/{path}\n"
    )


def _make_raw_issue(line: int, conf: float, sev: str = "high", file: str = "a.py") -> str:
    import json

    return json.dumps(
        {
            "issues": [
                {
                    "title": "test issue",
                    "body": "test body",
                    "severity": sev,
                    "confidence": conf,
                    "file": file,
                    "line": line,
                    "category": "bug",
                }
            ]
        }
    )


async def test_pipeline_no_files_returns_empty() -> None:
    p = ReviewPipeline(PipelineDeps(provider=FakeProvider([]), settings=_settings()))
    out = await p.run([])
    assert out.issues == []
    assert "No reviewable" in out.summary


async def test_pipeline_runs_full_flow() -> None:
    # Issue on line 5 of a.py - a.py has added_lines [5, 6].
    raw = _make_raw_issue(line=5, conf=0.9)
    provider = FakeProvider([raw])
    files = [_file("a.py", [5, 6], patch="@@ -1 +1,2 @@\n+x\n+y\n")]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    out = await p.run(files)
    assert len(out.issues) == 1
    assert out.issues[0].file == "a.py"
    assert out.issues[0].line == 5


async def test_pipeline_drops_low_confidence() -> None:
    raw = _make_raw_issue(line=5, conf=0.3)
    provider = FakeProvider([raw])
    files = [_file("a.py", [5, 6])]
    p = ReviewPipeline(
        PipelineDeps(provider=provider, settings=_settings(review_min_confidence=0.7))
    )
    out = await p.run(files)
    assert out.issues == []


async def test_pipeline_drops_unanchored() -> None:
    raw = _make_raw_issue(line=999, conf=0.95)
    provider = FakeProvider([raw])
    files = [_file("a.py", [5, 6])]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    out = await p.run(files)
    assert out.issues == []


async def test_pipeline_dedups_across_chunks() -> None:
    # Same fingerprint returned from two chunks.
    raw = _make_raw_issue(line=5, conf=0.9)
    provider = FakeProvider([raw, raw])
    files = [_file("a.py", [5])]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    out = await p.run(files)
    assert len(out.issues) == 1


async def test_pipeline_provider_error_recorded_as_failure() -> None:
    from ai_code_review.agent.pipeline import PipelineFailure
    from ai_code_review.errors import ProviderError

    class BadProvider(FakeProvider):
        async def complete(self, messages: list[dict[str, str]]) -> str:
            raise ProviderError("boom")

    provider = BadProvider([])
    files = [_file("a.py", [5])]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    # All files failed → critical failure raises.
    with pytest.raises(PipelineFailure):
        await p.run(files)


async def test_pipeline_partial_failure_does_not_abort() -> None:
    # Two files: one succeeds, one fails. Pipeline should not abort and
    # should report the failure in failed_files.
    from ai_code_review.errors import ProviderError

    good_resp = _make_raw_issue(line=5, conf=0.9)
    call_count = {"n": 0}

    class MixedProvider(FakeProvider):
        async def complete(self, messages: list[dict[str, str]]) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ProviderError("first file fails")
            return good_resp

    provider = MixedProvider([good_resp])
    files = [_file("fail.py", [5]), _file("ok.py", [5])]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    out = await p.run(files)
    assert len(out.failed_files) == 1
    assert len(out.analyzed_files) == 1
    assert out.failed_files[0].path in {"fail.py", "ok.py"}


async def test_pipeline_malformed_response_does_not_abort() -> None:
    provider = FakeProvider(["not json at all"])
    files = [_file("a.py", [5])]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings()))
    out = await p.run(files)
    assert out.issues == []
    # File was attempted but produced no issues; still counted as analyzed.
    assert out.analyzed_files == ["a.py"]


async def test_pipeline_limits_files() -> None:
    provider = FakeProvider(['{"issues": []}'] * 10)
    files = [_file(f"f{i}.py", [1]) for i in range(10)]
    p = ReviewPipeline(PipelineDeps(provider=provider, settings=_settings(review_max_files=3)))
    out = await p.run(files)
    assert out.limited is True
    assert out.limit_reason is not None
    assert len(out.analyzed_files) == 3


# ---- git integration tests against a real (temp) repo ----


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_CONFIG_NOSYSTEM": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True, env=env
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True, env=env)
    # Initial commit on main.
    (repo / "README.md").write_text("# initial\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True, env=env)
    return repo


def test_gitrepo_find_walks_up(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    found = GitRepo.find(sub)
    assert found.root == repo.resolve()


def test_gitrepo_find_outside_repo_raises(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(Exception):
        GitRepo.find(elsewhere)


def test_gitrepo_diff_against_base(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "foo.py").write_text("def foo():\n    return 1\n")
    env = _git_env(tmp_path)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "add foo"], check=True, env=env)

    g = GitRepo(repo)
    diff_text = g.diff("HEAD~1", "HEAD")
    files = parse_diff(diff_text)
    assert any(f.path == "foo.py" for f in files)


def test_gitrepo_staged_diff(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "bar.py").write_text("x = 1\n")
    env = _git_env(tmp_path)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    g = GitRepo(repo)
    diff_text = g.diff("", "", staged=True)
    assert "bar.py" in diff_text


def test_gitrepo_read_file_from_tree_refuses_traversal(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    g = GitRepo(repo)
    from ai_code_review.errors import GitError

    with pytest.raises(GitError):
        g.read_file_from_tree("../../etc/passwd")
    with pytest.raises(GitError):
        g.read_file_from_tree("/etc/passwd")
    with pytest.raises(GitError):
        g.read_file_from_tree("-malicious")


def test_gitrepo_read_file_from_tree_reads_head(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "tracked.py").write_text("x = 1\n")
    env = _git_env(tmp_path)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "add tracked"], check=True, env=env
    )

    # Now dirty the worktree - read_file_from_tree must still return the committed version.
    (repo / "tracked.py").write_text("x = 999  # dirty worktree\n")
    g = GitRepo(repo)
    body = g.read_file_from_tree("tracked.py", "HEAD")
    assert "x = 1" in body
    assert "999" not in body


def test_gitrepo_read_file_from_tree_missing_in_tree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    g = GitRepo(repo)
    from ai_code_review.errors import GitError

    with pytest.raises(GitError):
        g.read_file_from_tree("never_existed.py", "HEAD")


def _git_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_CONFIG_NOSYSTEM": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
