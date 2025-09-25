#!/bin/bash

# Script to rebuild the Track Gluer Togetherer.app bundle

echo "🔨 Rebuilding Track Gluer Togetherer.app..."

# Remove old app if it exists
if [ -d "Track Gluer Togetherer (New).app" ]; then
    rm -rf "Track Gluer Togetherer (New).app"
    echo "🗑️ Removed old app bundle"
fi

# Create app bundle structure
mkdir -p "Track Gluer Togetherer (New).app/Contents/MacOS"
mkdir -p "Track Gluer Togetherer (New).app/Contents/Resources"

echo "📁 Created app bundle structure"

# Create Info.plist
cat > "Track Gluer Togetherer (New).app/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Track Gluer Togetherer</string>
    <key>CFBundleIdentifier</key>
    <string>com.trackgluer.app</string>
    <key>CFBundleName</key>
    <string>Track Gluer Togetherer</string>
    <key>CFBundleVersion</key>
    <string>2.1</string>
    <key>CFBundleShortVersionString</key>
    <string>2.1</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.10</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo "📄 Created Info.plist"

# Create main executable
cat > "Track Gluer Togetherer (New).app/Contents/MacOS/Track Gluer Togetherer" << 'EOF'
#!/bin/bash

# Track Gluer Togetherer App Launcher
# This script runs from inside the .app bundle

echo "🎵 Starting Track Gluer Togetherer from App Bundle..."

# Get the path to the app bundle
APP_DIR="$(dirname "$(dirname "$(dirname "$0")")")"
SOURCE_DIR="$(dirname "$APP_DIR")"

echo "App Bundle: $APP_DIR"
echo "Source Directory: $SOURCE_DIR"

# Kill any existing instances of the app first
echo "🔍 Checking for existing instances..."
pkill -f "trackgluer.py" 2>/dev/null || true
pkill -f "Track Gluer Togetherer" 2>/dev/null || true

# Wait a moment for processes to close
sleep 2

# Check if ports are still in use and kill those processes
for port in 9876 9877 9878 9879 8000 8001; do
    pid=$(lsof -ti :$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "🔪 Killing process using port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
    fi
done

# Change to the source directory where the Python files are
cd "$SOURCE_DIR"

# Show current directory and git status to verify we're using latest code
echo "📁 Working directory: $(pwd)"
if [ -d ".git" ]; then
    echo "🔄 Git status: $(git log --oneline -1 2>/dev/null || echo 'No git info')"
fi

# Check if we're in a virtual environment, if not try to activate it
if [ -z "$VIRTUAL_ENV" ]; then
    # Try different virtual environment locations
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "✅ Activated virtual environment: venv"
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "✅ Activated virtual environment: .venv"
    else
        echo "⚠️ No virtual environment found, using system Python"
    fi
else
    echo "✅ Using existing virtual environment: $VIRTUAL_ENV"
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    echo "Press any key to exit..."
    read -n 1
    exit 1
fi

# Check if ffmpeg is installed
echo "🔍 Checking for ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found. Installing via Homebrew..."
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "   Then run the app again."
        echo "Press any key to exit..."
        read -n 1
        exit 1
    fi

    echo "📦 Installing ffmpeg..."
    brew install ffmpeg

    if [ $? -eq 0 ]; then
        echo "✅ ffmpeg installed successfully"
    else
        echo "❌ Failed to install ffmpeg. Please install it manually:"
        echo "   brew install ffmpeg"
        echo "Press any key to exit..."
        read -n 1
        exit 1
    fi
else
    echo "✅ ffmpeg found: $(which ffmpeg)"
fi

# Check if required Python packages are installed
echo "🔍 Checking Python dependencies..."
python3 -c "import flask, eyed3, musicbrainzngs" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required Python packages..."
    pip3 install flask eyed3 musicbrainzngs
fi

# Launch the app
echo "🚀 Launching Track Gluer Togetherer..."
echo "📍 Using Python: $(which python3)"
python3 trackgluer.py
EOF

# Make executable
chmod +x "Track Gluer Togetherer (New).app/Contents/MacOS/Track Gluer Togetherer"

echo "🎯 Made executable"
echo "✅ Track Gluer Togetherer (New).app created successfully!"
echo "🚀 You can now double-click the app to launch it"