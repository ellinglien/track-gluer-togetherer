#!/bin/bash
# Track Gluer Togetherer Installer

echo "🎵 Track Gluer Togetherer Installer"
echo "=================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)""
    exit 1
fi

# Install ffmpeg
echo "📦 Installing ffmpeg..."
brew install ffmpeg

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install flask eyed3 musicbrainzngs

echo "✅ Installation complete!"
echo "🚀 You can now run: python3 trackgluer.py"
