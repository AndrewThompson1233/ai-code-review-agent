"""Domain-specific exceptions.

The hierarchy is intentionally shallow: a single base class plus a few
narrow subclasses for places where callers actually need to branch.
"""

from __future__ import annotations


class ReviewError(Exception):
    """Base class for every error raised by the agent."""


class ConfigError(ReviewError):
    """Settings are missing or inconsistent."""


class GitError(ReviewError):
    """A git operation failed or the cwd is not a repository."""


class DiffError(ReviewError):
    """The diff text could not be parsed."""


class ProviderError(ReviewError):
    """An LLM provider returned an error or malformed payload."""


class ProviderTimeout(ProviderError):
    """The LLM provider did not answer in time."""


class GitHubError(ReviewError):
    """A GitHub REST API call failed."""


class ContextLimitExceeded(ReviewError):
    """The collected context exceeds the configured budget."""
