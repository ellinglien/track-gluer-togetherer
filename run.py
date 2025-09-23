#!/usr/bin/env python3
"""
Quick start script for Track Gluer Togetherer
Sets up virtual environment and starts the application
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_and_run():
    """Setup virtual environment and run the application"""
    print("🎵 Track Gluer Togetherer - Quick Start")
    print("=" * 40)
    
    venv_path = Path('venv')
    
    # Check if virtual environment exists
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', 'venv'])
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    # Determine virtual environment paths
    if os.name == 'nt':  # Windows
        venv_python = venv_path / 'Scripts' / 'python.exe'
        venv_pip = venv_path / 'Scripts' / 'pip.exe'
    else:  # macOS/Linux
        venv_python = venv_path / 'bin' / 'python'
        venv_pip = venv_path / 'bin' / 'pip'
    
    # Install required packages
    required_packages = ['flask>=3.0.0', 'eyed3>=0.9.0', 'musicbrainzngs>=0.7.0']
    
    print("📦 Installing/updating packages...")
    try:
        subprocess.check_call([str(venv_pip), 'install'] + required_packages, 
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ Dependencies ready")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False
    
    # Run the application
    print("🚀 Starting Track Gluer Togetherer...")
    try:
        subprocess.check_call([str(venv_python), 'trackgluer.py'])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Application error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    setup_and_run()
