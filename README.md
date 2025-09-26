# track gluer togetherer

minimal macos tool to merge mp3 audio files into single-track files

## macos guide 

### option 1: double-click launcher (easiest)
1. Download or clone this repository
2. Double-click `start_app.command` in Finder
3. The launcher will automatically install dependencies and start the web interface

### option 2: manual setup
```bash
# install dependencies
./install.sh

# run the app
python3 trackgluer.py
```
open: http://localhost:9876

## requirements

- macOS (for `.command` launcher)
- python 3.8+
- ffmpeg (auto-installed via Homebrew by launcher)
- flask, eyed3, musicbrainzngs (auto-installed by launcher)
- MP3 files only - currently supports MP3 format exclusively

## what it does

- groups mp3s by embedded album metadata
- merges tracks into single files
- preserves album art and metadata
- detects albums vs compilations 
- "glue em all" feature for quick compilation creation
- option to check online database musicbrainz
- drag-and-drop track reordering
- web interface 

## files

- `start_app.command` - double-clickable launcher (main entry point)
- `trackgluer.py` - web interface and core application
- `merge_albums.py` - audio merging logic
- `install.sh` - dependency installation script

---

## development & disclaimers

**Created with Claude Code:** This application was developed using Anthropic's Claude Code AI assistant.

**Experimental Software:** Provided as-is for educational and personal use. I made this for myself so I don't have time to offer support.

**Audio Format Support:** Currently supports MP3 files exclusively. Other audio formats (FLAC, WAV, M4A, etc.) are not supported.

**Backup Your Files:** Always keep backups of your original music files. This tool modifies and merges audio files - test with copies first.

**System Requirements:** Designed and tested on macOS 15.6.1. 

## 🔒 privacy & data usage

**Local Processing:** Audio processing happens locally on your device. Your music files never leave your computer.

**MusicBrainz Integration:** When using MusicBrainz features, the app queries the public [MusicBrainz database](https://musicbrainz.org/) to enhance track metadata and ordering. This involves:
- Sending track metadata (artist, album, track names) to MusicBrainz servers
- Only text metadata is transmitted
- MusicBrainz has their own [privacy policy](https://metabrainz.org/privacy) and data practices
- You can use the app without MusicBrainz features

**No Analytics:** This application does not collect, store, or transmit any user data or usage analytics.

**Dependencies:** This app installs and uses third-party packages (Flask, eyed3, musicbrainzngs, ffmpeg). Each has their own privacy policies and data practices.

## 📜 license & usage

This project is open source. Use at your own risk for personal, educational, and non-commercial purposes. Respect copyright laws when working with music files - only merge tracks you legally own or have permission to modify.
