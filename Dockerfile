FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so application edits do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip && pip install .

COPY alembic.ini ./
COPY alembic ./alembic
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 tepsir && chown -R tepsir:tepsir /app
USER tepsir

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
