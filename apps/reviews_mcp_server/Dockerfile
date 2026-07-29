FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

COPY apps/reviews_mcp_server ./apps/reviews_mcp_server/

ENV UV_COMPILE_BYTECODE=1

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package reviews_mcp_server

ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/apps/reviews_mcp_server/src:$PYTHONPATH"

RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app

USER app

EXPOSE 8002

WORKDIR /app/apps/items_mcp_server/src

CMD ["python", "-m", "reviews_mcp_server.main"]