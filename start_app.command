#!/bin/bash

# Track Gluer Togetherer Starter
# This .command file can be double-clicked in Finder

echo "🎵 Starting Track Gluer Togetherer..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working in: $SCRIPT_DIR"

# Kill any existing instances
pkill -f "trackgluer.py" 2>/dev/null || true
sleep 1

# Check if we're in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Using virtual environment: $VIRTUAL_ENV"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Activated venv"
else
    echo "⚠️ No virtual environment found, using system Python"
fi

# Quick dependency check
python3 -c "import flask" 2>/dev/null || pip3 install flask eyed3 musicbrainzngs

echo "🚀 Launching app..."
python3 trackgluer.py