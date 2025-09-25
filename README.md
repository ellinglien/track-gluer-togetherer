# track gluer togetherer

minimal tool to merge mp3 collections with drag-and-drop track reordering and custom album creation

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