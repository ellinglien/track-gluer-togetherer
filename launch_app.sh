#!/bin/bash

# Track Gluer Togetherer Launcher
# This script launches the app and opens it in the browser

echo "🎵 Starting Track Gluer Togetherer..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the app directory
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Activated virtual environment"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Using existing virtual environment: $VIRTUAL_ENV"
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Check if required packages are installed
python3 -c "import flask, eyed3, musicbrainzngs" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip3 install flask eyed3 musicbrainzngs
fi

# Launch the app
echo "🚀 Launching Track Gluer Togetherer..."
python3 trackgluer.py