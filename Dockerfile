# syntax=docker/dockerfile:1

# Stage 1: Builder
FROM python:3.14-slim AS builder

# Bundle uv binary from official distribution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Compile bytecode and configure uv for standalone container builds
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install project dependencies with layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source and build virtual environment
COPY romutil/ ./romutil/
COPY README.md ./
RUN uv sync --frozen --no-dev --no-editable

# Stage 2: Runtime
FROM python:3.14-slim AS runtime

# Install Coin-OR CBC solver binary and clean package caches
RUN apt-get update && \
    apt-get install -y --no-install-recommends coinor-cbc && \
    rm -rf /var/lib/apt/lists/*

# Bundle uv binary into runtime environment
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy built application and virtual environment from builder stage
COPY --from=builder /app /app

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Setup volume mount point for area files and output maps
WORKDIR /data
VOLUME ["/data"]

# Configure container entrypoint to invoke romutil
ENTRYPOINT ["romutil"]
CMD ["--help"]
