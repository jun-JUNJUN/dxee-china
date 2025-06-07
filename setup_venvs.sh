#!/bin/bash
# This script sets up virtual environments using uv for both frontend and backend

echo "Setting up backend virtual environment with uv..."
cd backend
uv sync
chmod +x run.sh
echo "Backend virtual environment setup complete."

echo "Setting up frontend virtual environment..."
cd ../frontend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x run.sh
deactivate
echo "Frontend virtual environment setup complete."

echo "All virtual environments have been set up successfully."
echo "To run the backend: cd backend && uv run ./run.sh"
echo "To run the frontend: cd frontend && ./run.sh"
echo "To run Meilisearch: docker-compose up -d"
