from __future__ import annotations

from ..config import Settings
from ..errors import GitError
from ..models import FileDiff
from .types import RepoLike


def gather_context(
    repo: RepoLike,
    file_diff: FileDiff,
    settings: Settings,
    *,
    head_ref: str = "HEAD",
) -> str:
    """Build a small context block around the changed file.

    Context is read from `head_ref` via `git show`, NOT from the worktree.
    The worktree may be dirty or checked out to a different commit, so reading
    from disk would give the LLM context that does not match the diff.

    Bounded by `review_max_context_size`: full file if it fits, otherwise a
    window of N lines around the first added hunk. We never pull in unrelated
    files - the cost/noise tradeoff is too poor for an automated reviewer.
    """
    if file_diff.is_new:
        return "(new file; no prior context)"
    if file_diff.is_deleted:
        return "(deleted file; no prior context)"

    try:
        body = repo.read_file_from_tree(file_diff.path, head_ref)
    except GitError:
        return f"(file not present in {head_ref}; no context)"

    if len(body) > settings.review_max_file_size:
        body = body[: settings.review_max_context_size]

    if len(body) <= settings.review_max_context_size:
        return f"### {file_diff.path} (full)\n```\n{body}\n```"

    first_added = file_diff.added_lines[0] if file_diff.added_lines else 1
    window_start = max(1, first_added - 30)
    lines = body.splitlines()
    end = min(len(lines), first_added + 30)
    window = "\n".join(lines[window_start - 1 : end])
    return f"### {file_diff.path} (lines {window_start}-{end} of {len(lines)})\n```\n{window}\n```"
