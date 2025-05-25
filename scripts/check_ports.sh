#!/bin/bash

# Ports to check
PORTS=(8180 8143 5101 8101 7701)

echo "Checking port availability..."

for port in "${PORTS[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "❌ Port $port is already in use:"
        lsof -i :$port
    else
        echo "✅ Port $port is available"
    fi
done

echo -e "\nIf any ports are in use, please update docker-compose.yml with different port mappings." 
