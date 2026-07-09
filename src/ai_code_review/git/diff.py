from __future__ import annotations

import re
from collections.abc import Iterable

from ..errors import DiffError
from ..models import FileDiff

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# `diff --git a/path b/path` — paths may be quoted when they contain spaces
# or special chars. We accept both quoted and unquoted forms.
_FILE_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_FILE_HEADER_QUOTED = re.compile(r'^diff --git "(.+?)" "(.+?)"$')
_OLD_NEW = re.compile(r"^--- (?:a/)?(.*)$")
_NEW_FILE = re.compile(r"^\+\+\+ (?:b/)?(.*)$")
_QUOTED_PATH = re.compile(r'^"(.+)"$')
_BINARY = re.compile(r"^Binary files")
_NEW_FILE_MODE = re.compile(r"^new file mode")
_DELETED_FILE_MODE = re.compile(r"^deleted file mode")
_RENAME = re.compile(r"^rename from (.+)$")


def _maybe_unquote(path: str) -> str:
    """Unquote a git-quoted path (`"path with space"` → `path with space`)."""
    if m := _QUOTED_PATH.match(path):
        return _unescape_git_path(m.group(1))
    return path


def _unescape_git_path(s: str) -> str:
    # git uses C-style escapes inside quoted paths: \\, \", \n, \t, etc.
    # Python's `unicode_escape` is close enough for the common cases.
    try:
        return s.encode("utf-8").decode("unicode_escape")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


def parse_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified git diff into structured FileDiff objects.

    The parser is intentionally strict: anything that doesn't match the
    expected grammar raises DiffError so the caller can decide to abort
    instead of silently producing an empty review.
    """
    if not diff_text or not diff_text.strip():
        return []

    files: list[FileDiff] = []
    current: FileDiff | None = None
    current_lines: list[tuple[int, int | None, str]] | None = None
    patch_lines: list[str] = []
    new_line = 0
    seen_added_in_hunk = False

    for raw in diff_text.splitlines():
        if raw.startswith("diff --git"):
            if current is not None:
                _flush(current, current_lines)
                current.patch = "\n".join(patch_lines)
                files.append(current)
            m = _FILE_HEADER_QUOTED.match(raw) or _FILE_HEADER.match(raw)
            if not m:
                raise DiffError(f"unparseable diff header: {raw!r}")
            old = _maybe_unquote(m.group(1))
            new = _maybe_unquote(m.group(2))
            current = FileDiff(path=new, old_path=old if old != new else None)
            current_lines = []
            patch_lines = [raw]
            seen_added_in_hunk = False
            continue

        if current is None:
            continue

        patch_lines.append(raw)

        if _NEW_FILE_MODE.match(raw):
            current.is_new = True
            continue
        if _DELETED_FILE_MODE.match(raw):
            current.is_deleted = True
            continue
        if m := _RENAME.match(raw):
            current.old_path = _maybe_unquote(m.group(1))
            continue
        if _BINARY.match(raw):
            current.is_binary = True
            continue
        if m := _OLD_NEW.match(raw):
            val = _maybe_unquote(m.group(1))
            if val != "/dev/null":
                current.old_path = val.rstrip("\t\r")
            continue
        if m := _NEW_FILE.match(raw):
            val = _maybe_unquote(m.group(1))
            if val != "/dev/null":
                current.path = val.rstrip("\t\r")
            continue

        if m := _HUNK_HEADER.match(raw):
            seen_added_in_hunk = True
            try:
                new_line = int(m.group(3))
            except ValueError as exc:
                raise DiffError(f"bad hunk header: {raw!r}") from exc
            # Don't reset current_lines: a file can have multiple hunks and
            # we accumulate added lines across all of them.
            if current_lines is None:
                current_lines = []
            continue

        if not seen_added_in_hunk or current.is_binary:
            continue

        if raw.startswith("+") and not raw.startswith("+++"):
            current.added += 1
            if current_lines is None:
                raise DiffError(f"added line before hunk header: {raw!r}")
            current_lines.append((new_line, None, raw[1:]))
            new_line += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            current.removed += 1
        elif raw.startswith("\\"):
            continue
        elif raw.startswith(" ") or raw == "":
            new_line += 1

    if current is not None:
        _flush(current, current_lines)
        current.patch = "\n".join(patch_lines)
        files.append(current)

    return files


def _flush(f: FileDiff, lines: Iterable[tuple[int, int | None, str]] | None) -> None:
    if lines is not None:
        f.added_lines = [ln for ln, _, _ in lines]


def filter_reviewable(files: list[FileDiff], *, max_files: int) -> tuple[list[FileDiff], list[str]]:
    """Split files into reviewable and skipped.

    Skips binary files, lock files, generated artifacts and vendored code.
    Enforces a per-PR file cap so a 500-file autogenerated PR cannot blow the
    context budget.
    """
    skipped: list[str] = []
    reviewable: list[FileDiff] = []
    for f in files:
        if f.is_binary:
            skipped.append(f.path)
            continue
        if _is_ignored(f.path):
            skipped.append(f.path)
            continue
        if len(reviewable) >= max_files:
            skipped.append(f.path)
            continue
        reviewable.append(f)
    return reviewable, skipped


_IGNORED_DIRS = (
    "node_modules/",
    "vendor/",
    "third_party/",
    ".venv/",
    "venv/",
    "dist/",
    "build/",
    "__pycache__/",
    ".git/",
    ".idea/",
    ".vscode/",
)
_IGNORED_SUFFIXES = (
    ".lock",
    ".map",
    ".min.js",
    ".min.css",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
)
_IGNORED_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "go.sum",
    "composer.lock",
    "Pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "Gemfile.lock",
    "mix.lock",
}


def _is_ignored(path: str) -> bool:
    lower = path.lower()
    if any(lower.startswith(d) or f"/{d}" in lower for d in _IGNORED_DIRS):
        return True
    if any(lower.endswith(suf) for suf in _IGNORED_SUFFIXES):
        return True
    name = lower.rsplit("/", 1)[-1]
    if name in _IGNORED_NAMES:
        return True
    return name.endswith(".generated.ts") or name.endswith(".gen.go") or ".pb." in name
