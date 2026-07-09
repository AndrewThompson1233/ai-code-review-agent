"""Shared structural types for the agent.

Kept in its own module so `pipeline` and `context` can both depend on it
without a circular import.
"""

from __future__ import annotations

from typing import Protocol


class RepoLike(Protocol):
    """Structural type for the repo dependency.

    PipelineDeps accepts any object with `read_file_from_tree` so tests can
    substitute a fake without inheriting from GitRepo.
    """

    def read_file_from_tree(self, path: str, ref: str = "HEAD") -> str: ...
