#!/usr/bin/env fish
# This script sets up the backend environment for uv usage
# Usage: source activate_backend.sh

# Get the absolute path to the script directory
set SCRIPT_DIR (cd (dirname (status -f)); and pwd)

# Change to backend directory for uv context
cd "$SCRIPT_DIR/backend"

echo "Backend uv environment activated."
echo "You are now in: (pwd)"
echo "Use 'uv run <command>' to run tools with dependencies"
