# track gluer togetherer

minimal tool to merge mp3 collections with drag-and-drop track reordering and custom album creation

## quick start (mac)

### option 1: run the app bundle
1. Download `Track Gluer Togetherer.app`
2. Drag to Applications folder
3. Double-click to run

### option 2: install from source
```bash
# install dependencies
./install.sh

# run the app
python3 trackgluer.py
```
open: http://localhost:9876

### option 3: create app bundle
```bash
python3 setup.py
```

## requirements

- python 3.8+
- ffmpeg (install with: `brew install ffmpeg`)
- flask, eyed3, musicbrainzngs (auto-installed)

## what it does

- groups mp3s by album metadata
- merges tracks into single files
- preserves album art
- detects albums vs mixes
- uses musicbrainz for accurate track ordering
- web interface for easy use

## files

- `electron/` - desktop app
- `trackgluer.py` - web interface
- `merge_albums.py` - core merging logic
- `requirements.md` - detailed setup instructions