# track gluer togetherer

minimal tool to merge mp3 collections with drag-and-drop track reordering and custom album creation

> **Created for intentional music listening** - Originally built to create full-album files for the Light Phone 3's basic music player, enabling a more focused, album-oriented listening experience.

## quick start (mac)

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
- **MP3 files only** - currently supports MP3 format exclusively

## what it does

- groups mp3s by album metadata
- merges tracks into single files
- preserves album art and metadata
- detects albums vs compilations automatically
- custom artist support with intelligent "Various Artists" detection
- "glue em all" feature for quick compilation creation
- uses musicbrainz for accurate track ordering
- drag-and-drop track reordering
- web interface for easy use

## features

✅ **smart grouping** - automatically detects single-artist albums vs compilations
✅ **custom albums** - create custom compilations from mixed tracks
✅ **musicbrainz integration** - enhanced metadata and accurate track ordering
✅ **glue em all** - one-click merging of all tracks into a single compilation
✅ **drag-and-drop** - intuitive track reordering interface
✅ **metadata preservation** - keeps album art, artist info, and track data

## files

- `start_app.command` - double-clickable launcher (main entry point)
- `trackgluer.py` - web interface and core application
- `merge_albums.py` - audio merging logic
- `install.sh` - dependency installation script

---

## 🤖 development & disclaimers

**Created with Claude Code:** This application was developed using Anthropic's Claude Code AI assistant. While thoroughly tested, functionality may vary across different systems and use cases.

**Experimental Software:** This tool is provided as-is for educational and personal use. It may not work perfectly for everyone or in all scenarios.

**Audio Format Support:** Currently supports MP3 files exclusively. Other audio formats (FLAC, WAV, M4A, etc.) are not supported.

**Backup Your Files:** Always keep backups of your original music files. This tool modifies and merges audio files - test with copies first.

**System Requirements:** Designed and tested on macOS. Performance and compatibility on other systems not guaranteed.

## 🔒 privacy & data usage

**Local Processing:** All audio processing happens locally on your device. Your music files never leave your computer.

**MusicBrainz Integration:** When using MusicBrainz features, the app queries the public [MusicBrainz database](https://musicbrainz.org/) to enhance track metadata and ordering. This involves:
- Sending track metadata (artist, album, track names) to MusicBrainz servers
- No audio files are transmitted - only text metadata
- MusicBrainz has their own [privacy policy](https://metabrainz.org/privacy) and data practices
- You can use the app without MusicBrainz features to avoid external queries

**No Analytics:** This application does not collect, store, or transmit any user data or usage analytics.

**Dependencies:** This app installs and uses third-party packages (Flask, eyed3, musicbrainzngs, ffmpeg). Each has their own privacy policies and data practices.

## 📜 license & usage

This project is open source. Use at your own risk for personal, educational, and non-commercial purposes. Respect copyright laws when working with music files - only merge tracks you legally own or have permission to modify.