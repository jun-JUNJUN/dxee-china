#!/bin/bash
# Activate the virtual environment
source venv/bin/activate

# Set environment variables
export FLASK_APP=apps
export FLASK_ENV=development
export MEILISEARCH_URL=http://localhost:7701

# Run the Flask application using gunicorn
gunicorn --bind 0.0.0.0:5101 --reload "apps:create_app()"
