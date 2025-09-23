#!/usr/bin/env python3
"""
Track Gluer Togetherer Setup Script
Creates a standalone Mac application bundle
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_and_install_dependencies():
    """Check and install required dependencies in virtual environment"""
    print("Checking dependencies...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print("✅ Python version OK")

    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ ffmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found. Install with: brew install ffmpeg")
        return False

    # Check if we're in a virtual environment or need to create one
    venv_path = Path('venv')
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', 'venv'])
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False

    # Use virtual environment python
    if os.name == 'nt':  # Windows
        venv_python = venv_path / 'Scripts' / 'python.exe'
        venv_pip = venv_path / 'Scripts' / 'pip.exe'
    else:  # macOS/Linux
        venv_python = venv_path / 'bin' / 'python'
        venv_pip = venv_path / 'bin' / 'pip'

    # Install pip packages in virtual environment
    required_packages = ['flask>=3.0.0', 'eyed3>=0.9.0', 'musicbrainzngs>=0.7.0']

    print("📦 Installing packages in virtual environment...")
    try:
        subprocess.check_call([str(venv_pip), 'install'] + required_packages)
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages in virtual environment")
        return False

    print(f"\n💡 To run the application:")
    print(f"   source venv/bin/activate  # (or venv\\Scripts\\activate on Windows)")
    print(f"   python trackgluer.py")
    print(f"\n   Or simply run: python3 run.py")

    return True

def create_app_bundle():
    """Create a Mac .app bundle"""
    print("\nCreating Mac application bundle...")
    
    app_name = "Track Gluer Togetherer"
    app_dir = Path(f"{app_name}.app")
    
    # Remove existing bundle
    if app_dir.exists():
        shutil.rmtree(app_dir)
    
    # Create bundle structure
    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    
    for dir_path in [contents_dir, macos_dir, resources_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create Info.plist
    info_plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>track-gluer-togetherer</string>
    <key>CFBundleIdentifier</key>
    <string>com.trackgluer.togetherer</string>
    <key>CFBundleName</key>
    <string>{app_name}</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.14</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>'''
    
    with open(contents_dir / "Info.plist", 'w') as f:
        f.write(info_plist)
    
    # Create launcher script
    launcher_script = '''#!/bin/bash
# Track Gluer Togetherer Launcher

# Get the directory containing this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES_DIR="$DIR/../Resources"

# Change to resources directory
cd "$RESOURCES_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install flask eyed3 musicbrainzngs
else
    source venv/bin/activate
fi

# Launch the application
python3 trackgluer.py

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo "Press any key to close..."
    read -n 1
fi
'''
    
    launcher_path = macos_dir / "track-gluer-togetherer"
    with open(launcher_path, 'w') as f:
        f.write(launcher_script)
    
    # Make launcher executable
    os.chmod(launcher_path, 0o755)
    
    # Copy application files
    app_files = ['trackgluer.py', 'merge_albums.py']
    for file_name in app_files:
        if Path(file_name).exists():
            shutil.copy2(file_name, resources_dir)
            print(f"✅ Copied {file_name}")
        else:
            print(f"❌ {file_name} not found")
            return False
    
    # Copy requirements.txt if it exists
    if Path('requirements.txt').exists():
        shutil.copy2('requirements.txt', resources_dir)
    else:
        # Create requirements.txt
        with open(resources_dir / 'requirements.txt', 'w') as f:
            f.write('flask>=2.0.0\neyed3>=0.9.0\nmusicbrainzngs>=0.7.0\n')
    
    print(f"✅ Created {app_name}.app")
    print(f"📁 You can now drag {app_name}.app to your Applications folder")
    print("🚀 Double-click the app to launch Track Gluer Togetherer")
    
    return True

def create_installer_script():
    """Create a simple installer script"""
    print("\nCreating installer script...")
    
    installer_script = '''#!/bin/bash
# Track Gluer Togetherer Installer

echo "🎵 Track Gluer Togetherer Installer"
echo "=================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
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
'''
    
    with open('install.sh', 'w') as f:
        f.write(installer_script)
    
    os.chmod('install.sh', 0o755)
    print("✅ Created install.sh")
    
    return True

def main():
    """Main setup function"""
    print("🎵 Track Gluer Togetherer Setup")
    print("==============================")
    
    if not check_and_install_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return False
    
    print("\n✅ All dependencies found!")
    
    # Create app bundle
    if not create_app_bundle():
        print("\n❌ Failed to create app bundle")
        return False
    
    # Create installer script
    if not create_installer_script():
        print("\n❌ Failed to create installer script")
        return False
    
    print("\n🎉 Setup complete!")
    print("\nDistribution options:")
    print("1. Share the .app bundle - users can drag it to Applications")
    print("2. Share install.sh - users run it to install dependencies")
    print("3. Share the Python files - users run 'python3 trackgluer.py'")
    
    return True

if __name__ == "__main__":
    main()
