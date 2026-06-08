#!/bin/bash
set -euo pipefail

echo "Waiting for PostgreSQL..."
until python -c "
import asyncio
from sqlalchemy import text
from core.database import engine

async def check():
    async with engine.connect() as conn:
        await conn.execute(text('SELECT 1'))

asyncio.run(check())
" 2>/dev/null; do
  sleep 2
done

echo "Initializing database..."
python -c "
import asyncio
from core.database import init_db
asyncio.run(init_db())
"

if [ "${AUTO_SEED_ON_STARTUP:-false}" = "true" ]; then
  echo "Generating mock data if missing..."
  python -m synthetic_data.generate_raw_mock || true

  echo "Starting API in background for seeding..."
  uvicorn api.main:app --host 127.0.0.1 --port 8000 &
  API_PID=$!
  sleep 5

  echo "Loading mock data..."
  python -m synthetic_data.load_mock_data --limit 100 --api-key "${API_KEY:-}" || true

  echo "Training models..."
  python -m models_ai.train || true

  kill $API_PID 2>/dev/null || true
  wait $API_PID 2>/dev/null || true
fi

echo "Starting Alt-Credit Engine..."
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
