# Dockerfile for the FinTech Compliance Document Simplifier application

# Base stage with uv and dependencies
FROM python:3.13-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Development stage
FROM base AS development

ENV ENVIRONMENT=development

EXPOSE 8000

# Command for development (with hot reload)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production

ENV ENVIRONMENT=production

# Copy application code
COPY app/ ./app/

EXPOSE 8000

# Command for production
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
