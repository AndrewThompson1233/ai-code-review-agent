from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..errors import GitError

# We always invoke git with an explicit argv list (never shell=True) and we
# force the working directory to the discovered repo root, which is the only
# way to keep `git` from walking up into a parent repository.


class GitRepo:
    """Thin wrapper around `git` invoked as a subprocess.

    The repo root is resolved once at construction time; every subsequent
    command runs with cwd=root so the agent never operates outside the
    repository by accident.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @classmethod
    def find(cls, start: Path | None = None) -> GitRepo:
        """Walk up from *start* until a `.git` entry is found."""
        cwd = (start or Path.cwd()).resolve()
        if not cwd.exists():
            raise GitError(f"path does not exist: {cwd}")
        cur = cwd
        while True:
            if (cur / ".git").exists():
                return cls(cur)
            if cur.parent == cur:
                raise GitError(f"not a git repository (or any parent): {cwd}")
            cur = cur.parent

    def _run(self, args: list[str], *, timeout: float = 30.0) -> str:
        env = {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": os.environ.get("HOME", "/nonexistent"),
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GitError("git executable not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"git {' '.join(args)} timed out") from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise GitError(f"git {' '.join(args)} failed: {stderr or 'unknown error'}")
        return proc.stdout

    def head_sha(self, ref: str = "HEAD") -> str:
        return self._run(["rev-parse", ref]).strip()

    def diff(self, base: str, head: str = "HEAD", *, staged: bool = False) -> str:
        if staged:
            return self._run(["diff", "--cached", "--no-color", "--no-ext-diff"])
        return self._run(["diff", "--no-color", "--no-ext-diff", f"{base}..{head}"])

    def merge_base_diff(self, base: str, head: str = "HEAD") -> str:
        # For PR-style "three-dot" diff: changes introduced on head since the
        # merge-base with base. This is what GitHub uses for PR review.
        return self._run(["diff", "--no-color", "--no-ext-diff", f"{base}...{head}"])

    def read_file_from_tree(self, path: str, ref: str = "HEAD") -> str:
        """Read a file from a specific git tree, never from the worktree.

        This is the only safe way to fetch context for a diff: the worktree
        may have moved past `ref` (or be dirty), so reading from disk would
        give the LLM context that does not match the diff under review.

        Path validation happens before the `git show` call: git itself rejects
        traversal, but defense in depth is cheap and gives a cleaner error.
        """
        _validate_repo_path(path)
        try:
            return self._run(["show", f"{ref}:{path}"])
        except GitError as exc:
            # Translate "does not exist in tree" to a clearer message so the
            # caller can decide whether the file is new/deleted vs broken.
            msg = str(exc)
            if "does not exist" in msg or "exists on disk, but not in" in msg:
                raise GitError(f"file not found in {ref}: {path}") from None
            raise

    def file_exists_in_tree(self, path: str, ref: str = "HEAD") -> bool:
        _validate_repo_path(path)
        try:
            self._run(["cat-file", "-e", f"{ref}:{path}"])
            return True
        except GitError:
            return False


def _validate_repo_path(path: str) -> None:
    """Reject anything that could escape the repo namespace in `git show`."""
    if not path or "\x00" in path:
        raise GitError(f"refused empty or null path: {path!r}")
    if path.startswith("/") or path.startswith("\\"):
        raise GitError(f"refused absolute path: {path!r}")
    if path[0] == "-":
        # A leading dash turns the path into a git flag. `--` separator
        # already protects the call, but refuse early to keep the error clear.
        raise GitError(f"refused dash-leading path: {path!r}")
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        raise GitError(f"refused path traversal: {path!r}")
