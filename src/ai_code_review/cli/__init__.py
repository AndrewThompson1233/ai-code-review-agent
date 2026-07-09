from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .. import __version__
from ..agent import PipelineDeps, PipelineFailure, ReviewPipeline
from ..agent.parser import ParseError, parse_issues
from ..config import Settings, load_settings
from ..errors import GitError, ReviewError
from ..git import GitRepo, filter_reviewable, parse_diff
from ..github import GitHubClient, PrRef, parse_pr_ref, publish_review
from ..models import ReviewResult
from ..output import render_json
from ..providers import build_provider
from ..redact import redact

app = typer.Typer(
    name="ai-review",
    help="AI agent that reviews Git diffs and GitHub Pull Requests.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
err_console = Console(stderr=True)
console = Console()

# Exit codes are deliberately distinct so CI can tell "operational failure"
# from "review found issues" from "clean run".
EXIT_OK = 0
EXIT_OPERATIONAL_FAILURE = 1
EXIT_ISSUES_FOUND = 2


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def _bail(msg: str, code: int = EXIT_OPERATIONAL_FAILURE) -> None:
    err_console.print(f"[red]error:[/red] {redact(msg)}")
    raise typer.Exit(code=code)


@app.command()
def review(
    base: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option("--base", "-b", help="Base ref (default: auto-detected main/master)."),
    ] = None,
    head: Annotated[
        str,
        typer.Option("--head", help="Head ref (default: HEAD)."),
    ] = "HEAD",
    staged: Annotated[
        bool,
        typer.Option("--staged", help="Review staged changes instead of base..head."),
    ] = False,
    provider: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option("--provider", help="Override AI_PROVIDER for this run."),
    ] = None,
    model: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option("--model", help="Override AI_MODEL for this run."),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text | json."),
    ] = "text",
    min_confidence: Annotated[
        Optional[float],  # noqa: UP045
        typer.Option("--min-confidence", help="Override REVIEW_MIN_CONFIDENCE."),
    ] = None,
    max_files: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option("--max-files", help="Override REVIEW_MAX_FILES."),
    ] = None,
    concurrency: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option("--concurrency", help="Override REVIEW_CONCURRENCY."),
    ] = None,
    timeout: Annotated[
        Optional[float],  # noqa: UP045
        typer.Option("--timeout", help="Override REVIEW_TIMEOUT (seconds, whole pipeline)."),
    ] = None,
    fail_on_issues: Annotated[
        bool,
        typer.Option(
            "--fail-on-issues",
            help="Exit non-zero when issues are found (for CI gates). "
            "Default: successful analysis exits 0 even with issues.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Run pipeline without posting to GitHub."),
    ] = False,
    pr: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option("--pr", help="GitHub PR ref (owner/repo#N). Posts review if set."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Review local Git changes or a GitHub PR and print findings.

    Exit codes:
      0 — analysis succeeded, no issues (or issues found without --fail-on-issues)
      1 — operational failure (config/git/provider/github error, or all files failed)
      2 — analysis succeeded AND found issues (only with --fail-on-issues)
    """
    _setup_logging(verbose, quiet)

    try:
        settings = load_settings(
            ai_provider=provider,
            ai_model=model,
            review_min_confidence=min_confidence,
            review_max_files=max_files,
            review_concurrency=concurrency,
            review_timeout=timeout,
        )
    except ReviewError as exc:
        _bail(str(exc))

    try:
        repo = GitRepo.find()
    except GitError as exc:
        _bail(str(exc))

    try:
        diff_text = _get_diff(repo, base=base, head=head, staged=staged)
    except GitError as exc:
        _bail(str(exc))

    if not diff_text.strip():
        console.print("[dim]No changes to review.[/dim]")
        _emit_empty(fmt)
        raise typer.Exit(code=EXIT_OK)

    files = parse_diff(diff_text)
    reviewable, skipped = filter_reviewable(files, max_files=settings.review_max_files)

    if not reviewable:
        console.print("[dim]No reviewable source files in the diff.[/dim]")
        _emit_empty(fmt, skipped=skipped, total_changed=len(files))
        raise typer.Exit(code=EXIT_OK)

    try:
        llm = build_provider(settings)
    except ReviewError as exc:
        _bail(str(exc))

    deps = PipelineDeps(provider=llm, settings=settings, repo=repo, head_ref=head)
    pipeline = ReviewPipeline(deps)

    async def _run_and_close() -> ReviewResult:
        try:
            return await pipeline.run(reviewable)
        finally:
            # Close the provider in the SAME event loop that created its
            # httpx client. Closing in a fresh loop would fail because the
            # client is bound to this loop.
            with contextlib.suppress(Exception):
                await _close_provider(llm)

    try:
        result = asyncio.run(_run_and_close())
    except PipelineFailure as exc:
        _bail(str(exc), code=EXIT_OPERATIONAL_FAILURE)
        raise  # unreachable; _bail raises typer.Exit, but keeps type checker happy
    except ReviewError as exc:
        _bail(str(exc), code=EXIT_OPERATIONAL_FAILURE)
        raise  # unreachable; _bail raises typer.Exit, but keeps type checker happy

    result.skipped_files = skipped
    result.total_changed_files = len(files)

    if fmt == "json":
        console.print(render_json(result))
    else:
        _render_rich(result)

    if pr and not dry_run:
        _publish_to_pr(pr, result, settings, head_sha=head)

    # Exit-code policy:
    #   - operational failures already bailed above
    #   - partial failure (some files failed, some analyzed) still exits 0
    #     so CI stays green, but the summary makes the partial failure loud
    #   - issues found → 0 by default, 2 with --fail-on-issues
    if result.failed_files and not result.analyzed_files:
        # Total failure: every eligible file failed. Bail with operational
        # failure so CI turns red — "no issues" here would be a lie.
        _bail(
            f"all {result.failed_count} file(s) failed analysis; see summary",
            code=EXIT_OPERATIONAL_FAILURE,
        )

    if result.issues and fail_on_issues:
        raise typer.Exit(code=EXIT_ISSUES_FOUND)
    raise typer.Exit(code=EXIT_OK)


@app.command()
def version() -> None:
    """Print the agent version."""
    console.print(__version__)


@app.command(name="parse-test")
def parse_test(
    raw: Annotated[
        str,
        typer.Argument(help="Raw LLM response to parse (or '-' to read from stdin)."),
    ],
) -> None:
    """Diagnostic helper: parse a raw LLM response and print the issues.

    Useful for prompt iteration without burning an LLM call.
    """
    text = sys.stdin.read() if raw == "-" else raw
    try:
        issues = parse_issues(text)
    except ParseError as exc:
        _bail(f"parse failed: {exc}")
    payload = [
        {
            "title": i.title,
            "severity": i.severity.value,
            "confidence": i.confidence,
            "file": i.file,
            "line": i.line,
            "category": i.category.value,
            "body": i.body,
        }
        for i in issues
    ]
    console.print(json.dumps(payload, indent=2))


def _get_diff(repo: GitRepo, *, base: str | None, head: str, staged: bool) -> str:
    if staged:
        return repo.diff("", "", staged=True)
    if base:
        return repo.diff(base, head)
    for candidate in ("main", "master"):
        try:
            return repo.merge_base_diff(candidate, head)
        except GitError:
            continue
    return repo.diff("HEAD~1", head)


def _emit_empty(fmt: str, *, skipped: list[str] | None = None, total_changed: int = 0) -> None:
    if fmt == "json":
        empty = {
            "summary": "No reviewable changes.",
            "limited": False,
            "limit_reason": None,
            "total_changed_files": total_changed,
            "analyzed_files": [],
            "skipped_files": sorted(skipped or []),
            "omitted_files": [],
            "failed_files": [],
            "issues": [],
        }
        console.print(json.dumps(empty, indent=2))


def _render_rich(result: ReviewResult) -> None:
    console.print(Panel(result.summary, title="AI Code Review", border_style="cyan"))
    if result.limited and result.limit_reason:
        console.print(f"[yellow]limited:[/yellow] {result.limit_reason}")
    console.print(f"analyzed: {len(result.analyzed_files)} / {result.total_changed_files} file(s)")
    if result.skipped_files:
        console.print(f"skipped: {len(result.skipped_files)} file(s)")
    if result.omitted_files:
        console.print(f"omitted (max-files): {len(result.omitted_files)} file(s)")
    if result.failed_files:
        console.print(f"[red]failed:[/red] {len(result.failed_files)} file(s)")
        for f in result.failed_files[:5]:
            console.print(f"  - {f.path}: {f.reason}")

    if not result.issues:
        if result.failed_files and not result.analyzed_files:
            console.print("[red]Review incomplete — no files analyzed successfully.[/red]")
        else:
            console.print("[green]No high-confidence issues found.[/green]")
        return

    table = Table(title="Findings", show_lines=True, border_style="dim")
    table.add_column("#", style="dim", width=3)
    table.add_column("Sev", width=8)
    table.add_column("Cat", width=14)
    table.add_column("File:Line", overflow="fold")
    table.add_column("Conf", width=5)
    table.add_column("Finding", overflow="fold")
    for idx, i in enumerate(result.issues, 1):
        sev = i.severity.value
        style = {
            "critical": "bold red",
            "high": "red",
            "medium": "yellow",
            "low": "dim",
        }.get(sev, "")
        table.add_row(
            str(idx),
            f"[{style}]{sev.upper()}[/{style}]" if style else sev.upper(),
            i.category.value,
            f"{i.file}:{i.line}",
            f"{i.confidence:.2f}",
            f"{i.title}\n[dim]{i.body[:240]}[/dim]",
        )
    console.print(table)


async def _close_provider(llm: object) -> None:
    close = getattr(llm, "aclose", None)
    if close is None:
        return
    with contextlib.suppress(Exception):
        await close()


def _publish_to_pr(pr_ref: str, result: ReviewResult, settings: Settings, *, head_sha: str) -> None:
    try:
        owner, repo, num = parse_pr_ref(pr_ref)
    except ReviewError as exc:
        _bail(str(exc))

    pr = PrRef(owner=owner, repo=repo, number=num, head_sha=head_sha)

    async def _go() -> None:
        async with GitHubClient(settings) as gh:
            await publish_review(gh, pr, result, head_sha=head_sha)

    try:
        asyncio.run(_go())
    except ReviewError as exc:
        _bail(f"github publish failed: {exc}")


if __name__ == "__main__":
    app()
