# syntax=docker/dockerfile:1.7
# Build stage: install into a clean venv so we can copy only the runtime
# artifacts into the final image.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install build deps for any C-extension wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .


# --- runtime stage ---
FROM python:3.12-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Non-root user: the agent reviews untrusted code, so never run as root.
RUN groupadd --system --gid 1001 agent \
    && useradd --system --uid 1001 --gid 1001 --create-home --shell /usr/sbin/nologin agent

COPY --from=builder /opt/venv /opt/venv

USER agent
WORKDIR /home/agent/repo

# The repo to review is mounted at /home/agent/repo at run time.
ENTRYPOINT ["ai-review"]
CMD ["--help"]
