#!/bin/bash
# Activate the virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export MEILISEARCH_URL=http://localhost:7701

# Run the application using gunicorn with tornado worker
gunicorn --bind 0.0.0.0:8100 --workers=1 --worker-class=tornado --log-level=debug --reload --timeout=300 --capture-output wsgi:application
