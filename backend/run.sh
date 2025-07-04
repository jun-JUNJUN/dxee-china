#!/bin/bash
# Clear any conflicting virtual environment variables
unset VIRTUAL_ENV

# Set environment variables
export PYTHONUNBUFFERED=1
export MEILISEARCH_URL=http://localhost:7701

# Run the application using gunicorn with tornado worker
uv run gunicorn --bind 0.0.0.0:8100 --workers=1 --worker-class=tornado --log-level=debug --reload --timeout=300 --capture-output wsgi:application
