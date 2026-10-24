#!/usr/bin/env sh
set -eu

echo "Applying database migrations..."
alembic upgrade head

echo "Starting TEPSIR API..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
