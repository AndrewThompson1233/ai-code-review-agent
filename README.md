# ai-code-review-agent

Automated code review tool for Git diffs and GitHub pull requests. Detects runtime crashes, security bugs, race conditions, and broken API contracts without generating linter noise.

## Core Design

- Review modes: Works on uncommitted working trees, staged diffs, commit ranges (`base..head`), or GitHub PRs.
- Model backend: OpenAI, Anthropic, or local OpenAI-compatible endpoints (vLLM, llama.cpp, Ollama). Handled through an async HTTP client.
- Structured output: Parses responses against a Pydantic schema, with a single retry on malformed JSON.
- Confidence cutoff: Drops findings below `REVIEW_MIN_CONFIDENCE` (default 0.7).
- Diff anchoring: Discards findings that do not touch or immediately neighbor modified lines, avoiding misaligned comments.
- Tree-isolated context: Reads file contents directly from the git object tree (`git show <ref>:<path>`) rather than disk, staying consistent even with dirty working trees.
- Idempotent comments: Adds invisible fingerprints to PR comments to prevent duplicate reports on rerun.
- Concurrency limit: Analyzes files concurrently with a bounded semaphore while preserving original file ordering.
- Hardened runner: Subprocesses use argument arrays (no `shell=True`), paths are verified against traversal, and tokens are redacted from logs.

## Pipeline

```
git diff -> parse -> filter -> tree context
                                   |
                                   v
                             LLM backend
                                   |
                                   v
             parse JSON -> schema validate -> confidence filter
                                   |
                                   v
                     anchor to diff -> deduplicate
                                   |
                 +-----------------+-----------------+
                 |                                   |
                 v                                   v
          CLI (stdout / JSON)                GitHub PR review
```

## Setup

```bash
git clone https://github.com/AndrewThompson1233/ai-code-review-agent
cd ai-code-review-agent
cp .env.example .env
pip install -e ".[dev]"
ai-review --help
```

Review staged changes:

```bash
ai-review review --staged
```

Review the latest commit:

```bash
ai-review review --commit HEAD
```

Review against main:

```bash
ai-review review origin/main..HEAD
```

## Configuration

Set via environment variables or CLI flags:

| Variable | Default | Description |
|---|---|---|
| `AI_PROVIDER` | `openai` | `openai`, `anthropic`, or `local` |
| `AI_MODEL` | provider default | e.g. `gpt-4o`, `claude-3-5-sonnet-20241022` |
| `AI_API_KEY` | - | API key for the selected provider |
| `AI_BASE_URL` | - | Custom base URL for local servers |
| `GITHUB_TOKEN` | - | GitHub token for posting PR comments |
| `REVIEW_MIN_CONFIDENCE` | `0.7` | Minimum threshold for reporting findings |
| `REVIEW_MAX_FILES` | `50` | Maximum files analyzed per run |
| `REVIEW_TIMEOUT` | `300` | Pipeline timeout in seconds |
| `REVIEW_CONCURRENCY` | `3` | Parallel file reviews (1 to 16) |

## CLI Output Example

```
Findings:
#  SEV      CATEGORY       FILE:LINE            CONF  SUMMARY
1  HIGH     SECURITY       src/api/auth.py:42   0.92  Unescaped parameter in raw SQL query
2  MEDIUM   CONCURRENCY    src/worker.py:88     0.78  Dict updated outside synchronization lock
```

Machine-readable output is available with `--format json`.

## GitHub Actions

Add `.github/workflows/review.yml` to run on incoming PRs:

```yaml
name: Review PR
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install .
      - run: ai-review review --pr ${{ github.event.pull_request.number }}
        env:
          AI_API_KEY: ${{ secrets.AI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

PRs from forks are ignored by default to prevent untrusted code execution.

## Testing

```bash
pip install -e ".[dev]"
ruff check src tests
mypy
pytest -q
```

## License

MIT (see [LICENSE](LICENSE)).
