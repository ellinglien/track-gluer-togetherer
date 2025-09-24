#!/usr/bin/env python3
"""
Album Merger for spotDL Downloads
Merges MP3 files from single albums into one file while preserving track order and metadata
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess
import musicbrainzngs
import urllib.parse


class AlbumMerger:
    def __init__(self, downloads_folder: str = "."):
        self.downloads_folder = Path(downloads_folder).resolve()

        # Configure MusicBrainz
        musicbrainzngs.set_useragent(
            "track-gluer-togetherer",
            "1.0",
            "https://github.com/user/track-gluer-togetherer"
        )

        if not self.downloads_folder.exists():
            raise FileNotFoundError(f"Downloads folder not found: {downloads_folder}")

    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_root_mp3_files(self) -> List[Path]:
        """Get all MP3 files in the root downloads directory"""
        return list(self.downloads_folder.glob("*.mp3"))

    def normalize_artist_name(self, artist: str) -> str:
        """Normalize artist name by removing featuring artists and common variations"""
        if not artist:
            return "Unknown Artist"

        # Remove featuring artists (feat., ft., featuring, with, &, etc.)
        patterns = [
            r'\s*[,&]\s*.*$',  # Remove everything after comma or ampersand
            r'\s*\(feat\..*?\)',  # Remove (feat. ...)
            r'\s*\(ft\..*?\)',    # Remove (ft. ...)
            r'\s*feat\..*$',      # Remove feat. ...
            r'\s*ft\..*$',        # Remove ft. ...
            r'\s*featuring.*$',   # Remove featuring ...
            r'\s*with\s+.*$',     # Remove with ...
        ]

        normalized = artist
        for pattern in patterns:
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)

        return normalized.strip()

    def is_likely_compilation(self, album: str, album_artist: str, track_artist: str, metadata: Dict[str, str]) -> bool:
        """Detect if this track is from a compilation/mix release"""
        album_lower = album.lower()

        # Check for compilation indicators in album name
        compilation_indicators = [
            'compilation', 'various', 'mixed by', 'compiled by', 'selected by',
            'presents', 'collection', 'anthology', 'best of', 'hits',
            'sound of', 'fabric', 'ministry of sound', 'defected', 'soma',
            'tresor', 'warp', 'ninja tune', 'xl recordings', 'kompakt'
        ]

        for indicator in compilation_indicators:
            if indicator in album_lower:
                return True

        # Check if album artist is "Various Artists" or similar
        if album_artist.lower() in ['various artists', 'various', 'va', 'compilation']:
            return True

        # Check if album artist is very different from track artist
        # This suggests it's a label compilation or mix
        if (album_artist.lower() != track_artist.lower() and
            album_artist not in ['Unknown Artist', 'Various Artists'] and
            track_artist not in ['Unknown Artist']):
            # If they share no common words, likely compilation
            album_words = set(album_artist.lower().split())
            track_words = set(track_artist.lower().split())
            if not album_words.intersection(track_words):
                return True

        return False

    def normalize_album_name(self, album: str) -> str:
        """Normalize album name for compilation grouping
        
        Handles case sensitivity and common variations in album names
        """
        if not album:
            return "Unknown Album"
            
        # Convert to lowercase for case-insensitive comparison
        normalized = album.lower()
        
        # Fix common issues in album names
        normalized = re.sub(r'(?i)([a-z])(album|ep|single|mixtape)$', r'\1 \2', normalized)  # Add space before album/ep/single/mixtape
        
        # Remove common suffixes that might cause splitting
        normalized = re.sub(r'\s*\(.*\)$', '', normalized)  # Remove anything in parentheses at the end
        normalized = re.sub(r'\s*\[.*\]$', '', normalized)   # Remove anything in brackets at the end
        normalized = re.sub(r'\s*-\s*\d{4}$', '', normalized)  # Remove year suffix (e.g., " - 2022")
        
        # Remove common words that might cause splitting
        common_suffixes = [
            'deluxe edition', 'deluxe', 'special edition', 'expanded edition',
            'anniversary edition', 'remastered', 'remaster', 'remastered version',
            'expanded', 'edition', 'version', 'album', 'ep', 'single', 'mixtape'
        ]
        
        for suffix in common_suffixes:
            pattern = r'\s*[\s-]' + re.escape(suffix) + r'$'
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # Standardize spacing and hyphens
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        normalized = re.sub(r'\s*-\s*', '-', normalized)
        
        # Remove any remaining non-alphanumeric characters except spaces and hyphens
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        
        # If we've stripped everything, return the original preserving case
        if not normalized:
            return album

        # Try to preserve original case: if the normalized result is found in the original (case-insensitive),
        # extract the corresponding portion from the original to preserve case
        original_lower = album.lower()
        if normalized in original_lower:
            start_pos = original_lower.find(normalized)
            if start_pos >= 0:
                end_pos = start_pos + len(normalized)
                return album[start_pos:end_pos]

        # If we can't preserve case exactly, return the original album name
        # (this ensures "TOTO" stays "TOTO" instead of becoming "toto")
        return album

        # Remove common prefixes that might vary
        prefixes_to_remove = [
            r'^various artists?\s*[-:]\s*',
            r'^va\s*[-:]\s*',
            r'^compilation\s*[-:]\s*',
        ]

        normalized = album
        for prefix_pattern in prefixes_to_remove:
            normalized = re.sub(prefix_pattern, '', normalized, flags=re.IGNORECASE)

        return normalized.strip()

    def detect_mix_type(self, mp3_files: List[Path]) -> str:
        """Detect if tracks are part of a mix/DJ set or regular album"""
        if len(mp3_files) < 3:
            return "album"

        # Check for mix indicators in filenames or metadata
        mix_indicators = ['mix', 'set', 'dj', 'live', 'session', 'radio', 'podcast']
        album_indicators = ['album', 'ep', 'single', 'compilation']

        mix_score = 0
        album_score = 0

        for mp3_file in mp3_files:
            filename_lower = mp3_file.name.lower()

            # Check filename for indicators
            for indicator in mix_indicators:
                if indicator in filename_lower:
                    mix_score += 1
                    break

            for indicator in album_indicators:
                if indicator in filename_lower:
                    album_score += 1
                    break

        # Check for sequential track numbering (indicates album)
        numbered_tracks = 0
        for mp3_file in mp3_files:
            filename_lower = mp3_file.name.lower()
            if filename_lower.startswith(('01', '02', '03', '04', '05', '06', '07', '08', '09', '1.', '2.', '3.')):
                numbered_tracks += 1

        if numbered_tracks >= len(mp3_files) * 0.5:  # 50% or more have track numbers
            album_score += 3

        # Check album artist consistency (albums usually have same album artist)
        try:
            album_artists = set()
            for mp3_file in mp3_files[:5]:  # Check first 5 tracks
                metadata = self.get_track_metadata(mp3_file)
                album_artist = (metadata.get('album_artist') or metadata.get('ALBUM_ARTIST') or
                              metadata.get('albumartist') or metadata.get('ALBUMARTIST') or
                              metadata.get('artist') or metadata.get('ARTIST'))
                if album_artist and album_artist != "Unknown Artist":
                    album_artists.add(self.normalize_artist_name(album_artist))

            # If all tracks have the same album artist, likely an album
            if len(album_artists) == 1:
                album_score += 2
            # If tracks have wildly different album artists, might be a mix
            elif len(album_artists) >= len(mp3_files[:5]) * 0.8:
                mix_score += 1

        except Exception:
            pass

        return "mix" if mix_score > album_score else "album"

    def extract_artist_from_filename(self, filename: str) -> str:
        """Extract artist name from filename in 'Artist - Title' format"""
        # Remove track numbers and file extension
        clean_name = re.sub(r'^\d+\s*[-\.]?\s*', '', filename)
        clean_name = re.sub(r'\.[^.]+$', '', clean_name)

        # Split on ' - ' and take first part as artist
        if ' - ' in clean_name:
            return clean_name.split(' - ')[0].strip()

        return "Unknown Artist"

    def detect_split_releases(self, grouped_albums: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
        """
        Detect when a single release has been split across multiple album groups
        Uses MusicBrainz to identify complete releases
        """
        print("🔍 Checking for split releases using MusicBrainz...")

        potential_merges = {}
        processed_albums = set()

        # Look for albums that might be parts of the same release
        for album_key, files in grouped_albums.items():
            if album_key in processed_albums:
                continue

            # Extract album info from the first file
            try:
                import eyed3
                audiofile = eyed3.load(str(files[0]))
                if not audiofile or not audiofile.tag:
                    continue

                artist = audiofile.tag.artist or "Unknown"
                album = audiofile.tag.album or "Unknown"

                # Skip if we don't have good metadata
                if artist == "Unknown" or album == "Unknown":
                    continue

                # Look up complete release info
                release_info = self.lookup_musicbrainz_release_info(artist, album)
                if not release_info:
                    continue

                print(f"📀 Found MusicBrainz release: {release_info['title']} ({release_info['track_count']} tracks)")

                # Check if we have tracks from this release split across multiple groups
                mb_track_titles = {track['title'].lower().strip() for track in release_info['tracks']}

                # Find all local tracks that match this release
                matching_tracks = []
                matched_album_keys = []

                for other_album_key, other_files in grouped_albums.items():
                    if other_album_key in processed_albums:
                        continue

                    # Check if any tracks from this album match the MusicBrainz release
                    album_matches = 0
                    for file_path in other_files:
                        try:
                            audio = eyed3.load(str(file_path))
                            if audio and audio.tag and audio.tag.title:
                                local_title = audio.tag.title.lower().strip()
                                # Remove common DJ mix suffixes for comparison
                                clean_title = re.sub(r'\s*-\s*mixed\s*$', '', local_title, flags=re.IGNORECASE)

                                if clean_title in mb_track_titles or local_title in mb_track_titles:
                                    album_matches += 1
                        except:
                            continue

                    # If significant portion of tracks match, consider this part of the same release
                    match_percentage = album_matches / len(other_files) if other_files else 0
                    if match_percentage >= 0.3:  # 30% threshold
                        matching_tracks.extend(other_files)
                        matched_album_keys.append(other_album_key)
                        print(f"  📎 Merging album group: {other_album_key} ({album_matches}/{len(other_files)} tracks match)")

                # If we found multiple groups that should be merged
                if len(matched_album_keys) > 1:
                    # Use the MusicBrainz title as the new key
                    merge_key = f"{release_info['artist']} - {release_info['title']}"
                    potential_merges[merge_key] = matching_tracks

                    # Mark these albums as processed
                    for key in matched_album_keys:
                        processed_albums.add(key)

                    print(f"✅ Merged {len(matched_album_keys)} album groups into: {merge_key}")

            except Exception as e:
                print(f"Error processing album {album_key}: {e}")
                continue

        # Merge the results
        final_groups = {}

        # Add merged groups
        final_groups.update(potential_merges)

        # Add unprocessed groups
        for album_key, files in grouped_albums.items():
            if album_key not in processed_albums:
                final_groups[album_key] = files

        if potential_merges:
            print(f"🎯 Split release detection complete. Merged {len(potential_merges)} releases.")

        return final_groups

    def group_mp3s_by_musicbrainz_first(self, mp3_files: List[Path]) -> Dict[str, List[Path]]:
        """
        Group MP3s using MusicBrainz as the primary method, fallback to metadata
        This prioritizes official release data over potentially inconsistent metadata
        """
        print("🎵 Using MusicBrainz-first grouping strategy...")
        print(f"🎵 Processing {len(mp3_files)} files")

        # Start with traditional grouping as a baseline
        print("📝 First, getting traditional grouping as baseline...")
        traditional_groups = self.group_mp3s_by_album_traditional(mp3_files)
        
        # Try to enhance groups with MusicBrainz data
        enhanced_groups = {}
        processed_files = set()
        
        print(f"🔍 Attempting to enhance {len(traditional_groups)} groups with MusicBrainz data...")
        
        for traditional_key, files in traditional_groups.items():
            if len(files) < 2:  # Skip single tracks
                enhanced_groups[traditional_key] = files
                processed_files.update(files)
                continue
                
            # Extract artist and album from the first file for MusicBrainz lookup
            try:
                import eyed3
                audiofile = eyed3.load(str(files[0]))
                if audiofile and audiofile.tag:
                    artist = audiofile.tag.artist or "Unknown"
                    album = audiofile.tag.album or "Unknown"
                    
                    if artist != "Unknown" and album != "Unknown":
                        print(f"🔍 Looking up: {artist} - {album}")
                        release_info = self.lookup_musicbrainz_release_info(artist, album)
                        
                        if release_info:
                            print(f"📀 Found MusicBrainz release: {release_info['title']} ({release_info['track_count']} tracks)")
                            
                            # Check if our files match this release
                            mb_track_titles = {track['title'].lower().strip() for track in release_info['tracks']}
                            matched_files = []
                            
                            for mp3_file in files:
                                try:
                                    audio = eyed3.load(str(mp3_file))
                                    if audio and audio.tag and audio.tag.title:
                                        local_title = audio.tag.title.lower().strip()
                                        # Remove common DJ mix suffixes for comparison
                                        clean_title = re.sub(r'\s*-\s*mixed\s*$', '', local_title, flags=re.IGNORECASE)
                                        
                                        if clean_title in mb_track_titles or local_title in mb_track_titles:
                                            matched_files.append(mp3_file)
                                except:
                                    continue
                            
                            # If most files match, use MusicBrainz title
                            match_percentage = len(matched_files) / len(files) if files else 0
                            if match_percentage >= 0.7:  # 70% threshold
                                mb_key = f"{release_info['artist']} - {release_info['title']}"
                                enhanced_groups[mb_key] = files
                                processed_files.update(files)
                                print(f"✅ Enhanced group with MusicBrainz title: {mb_key}")
                                continue
                        
                        print(f"❌ No suitable MusicBrainz data found for: {artist} - {album}")
                
            except Exception as e:
                print(f"Error processing group {traditional_key}: {e}")
            
            # Fall back to traditional grouping
            enhanced_groups[traditional_key] = files
            processed_files.update(files)
        
        # Ensure all files are accounted for
        missing_files = [f for f in mp3_files if f not in processed_files]
        if missing_files:
            print(f"⚠️  Found {len(missing_files)} unprocessed files, adding to Unknown group")
            unknown_key = "Unknown Artist - Unknown Album"
            if unknown_key not in enhanced_groups:
                enhanced_groups[unknown_key] = []
            enhanced_groups[unknown_key].extend(missing_files)

        print(f"🎯 Final result: {len(enhanced_groups)} album groups")
        return enhanced_groups

    def enhance_album_with_musicbrainz(self, mp3_files: List[Path], album_name: str = None) -> Dict[str, any]:
        """
        Enhance a specific album with MusicBrainz data
        Returns enhanced metadata and track ordering
        """
        if not mp3_files:
            return {"success": False, "error": "No files provided"}
        
        try:
            import eyed3
            
            # Extract album info from the first file
            audiofile = eyed3.load(str(mp3_files[0]))
            if not audiofile or not audiofile.tag:
                return {"success": False, "error": "Could not read metadata from files"}
            
            artist = audiofile.tag.artist or "Unknown"
            album = audiofile.tag.album or album_name or "Unknown"
            
            if artist == "Unknown" or album == "Unknown":
                return {"success": False, "error": "Missing artist or album information"}
            
            print(f"🔍 Looking up MusicBrainz data for: {artist} - {album}")
            
            # Get MusicBrainz release info
            release_info = self.lookup_musicbrainz_release_info(artist, album)
            
            if not release_info:
                return {"success": False, "error": f"No MusicBrainz data found for '{artist} - {album}'"}
            
            print(f"📀 Found MusicBrainz release: {release_info['title']} ({release_info['track_count']} tracks)")
            
            # Match local files to MusicBrainz tracks
            mb_track_titles = {track['title'].lower().strip(): track for track in release_info['tracks']}
            matched_tracks = []
            unmatched_files = []
            
            for mp3_file in mp3_files:
                try:
                    audio = eyed3.load(str(mp3_file))
                    if audio and audio.tag and audio.tag.title:
                        local_title = audio.tag.title.lower().strip()
                        # Remove common DJ mix suffixes for comparison
                        clean_title = re.sub(r'\s*-\s*mixed\s*$', '', local_title, flags=re.IGNORECASE)
                        
                        if clean_title in mb_track_titles:
                            mb_track = mb_track_titles[clean_title]
                            matched_tracks.append({
                                'file': mp3_file,
                                'local_title': audio.tag.title,
                                'mb_title': mb_track['title'],
                                'mb_track_number': mb_track['position'],
                                'mb_duration': mb_track.get('length', 0)
                            })
                        elif local_title in mb_track_titles:
                            mb_track = mb_track_titles[local_title]
                            matched_tracks.append({
                                'file': mp3_file,
                                'local_title': audio.tag.title,
                                'mb_title': mb_track['title'],
                                'mb_track_number': mb_track['position'],
                                'mb_duration': mb_track.get('length', 0)
                            })
                        else:
                            unmatched_files.append({
                                'file': mp3_file,
                                'local_title': audio.tag.title
                            })
                except Exception as e:
                    unmatched_files.append({
                        'file': mp3_file,
                        'local_title': mp3_file.stem,
                        'error': str(e)
                    })
            
            # Sort matched tracks by MusicBrainz track number
            matched_tracks.sort(key=lambda x: x['mb_track_number'])
            
            match_percentage = len(matched_tracks) / len(mp3_files) * 100
            
            return {
                "success": True,
                "release_info": release_info,
                "matched_tracks": matched_tracks,
                "unmatched_files": unmatched_files,
                "match_percentage": match_percentage,
                "total_files": len(mp3_files),
                "matched_count": len(matched_tracks),
                "unmatched_count": len(unmatched_files)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error during MusicBrainz lookup: {str(e)}"}

    def group_mp3s_by_album_traditional(self, mp3_files: List[Path]) -> Dict[str, List[Path]]:
        """Traditional grouping method (renamed from original)"""
        print("📝 Using traditional metadata grouping...")
        return self.group_mp3s_by_album(mp3_files)

    def group_mp3s_by_album(self, mp3_files: List[Path]) -> Dict[str, List[Path]]:
        """Group MP3 files by their album metadata with improved compilation detection"""
        albums = {}

        for mp3_file in mp3_files:
            try:
                metadata = self.get_track_metadata(mp3_file)

                # Extract album and artist info
                album = metadata.get('album') or metadata.get('ALBUM') or 'Unknown Album'
                artist = (metadata.get('album_artist') or metadata.get('ALBUM_ARTIST') or
                         metadata.get('albumartist') or metadata.get('ALBUMARTIST') or
                         metadata.get('artist') or metadata.get('ARTIST') or 'Unknown Artist')
                track_artist = metadata.get('artist') or metadata.get('ARTIST') or 'Unknown Artist'
                
                # Ensure consistent album artist for the same album
                if album.lower() == 'i put a spell on you':
                    artist = 'Nina Simone'
                
                # Additional Nina Simone album fixes
                if 'nina simone' in artist.lower():
                    artist = 'Nina Simone'  # Normalize capitalization
                
                # Debug logging for Nina Simone albums
                if 'nina simone' in artist.lower() or 'nina simone' in album.lower():
                    print(f"DEBUG: Nina Simone track - Album: '{album}', Artist: '{artist}', Track Artist: '{track_artist}'")

                # Check if this is likely a compilation or mix
                is_compilation = self.is_likely_compilation(album, artist, track_artist, metadata)

                if is_compilation:
                    # For compilations, use album as primary grouping key
                    album_key = f"Various Artists - {self.normalize_album_name(album)}"
                else:
                    # For regular albums, use album artist (not track artist) + normalized album name
                    normalized_artist = self.normalize_artist_name(artist)
                    normalized_album = self.normalize_album_name(album)
                    album_key = f"{normalized_artist} - {normalized_album}"
                
                # Debug logging for Nina Simone albums
                if 'nina simone' in artist.lower() or 'nina simone' in album.lower():
                    print(f"DEBUG: Nina Simone album key: '{album_key}' (compilation: {is_compilation})")

                if album_key not in albums:
                    albums[album_key] = []
                albums[album_key].append(mp3_file)

            except Exception as e:
                print(f"Warning: Could not read metadata from {mp3_file.name}: {e}")
                # Put ungroupable files in "Unknown" album
                unknown_key = "Unknown Artist - Unknown Album"
                if unknown_key not in albums:
                    albums[unknown_key] = []
                albums[unknown_key].append(mp3_file)

        return albums

    def extract_mix_info(self, mp3_files: List[Tuple[Path, Dict[str, str], int]]) -> Dict[str, str]:
        """Extract mix information (DJ name, mix title, date) from tracks"""
        mix_info = {
            'dj_name': 'Unknown DJ',
            'mix_title': 'Unknown Mix',
            'date': '',
            'total_tracks': len(mp3_files),
            'tracklist': []
        }

        # Try to extract DJ name and mix title from album metadata first
        for mp3_file, metadata, _ in mp3_files:
            if metadata:
                # Check for album artist (often the DJ name in mixes)
                dj_candidates = [
                    metadata.get('album_artist'),
                    metadata.get('ALBUM_ARTIST'),
                    metadata.get('albumartist'),
                    metadata.get('ALBUMARTIST'),
                    metadata.get('artist'),
                    metadata.get('ARTIST')
                ]

                for candidate in dj_candidates:
                    if candidate and candidate != 'Unknown Artist':
                        mix_info['dj_name'] = candidate
                        break

                # Check for album title (often the mix name)
                mix_candidates = [
                    metadata.get('album'),
                    metadata.get('ALBUM')
                ]

                for candidate in mix_candidates:
                    if candidate and candidate != 'Unknown Album':
                        mix_info['mix_title'] = candidate
                        break

                # Check for date
                date_candidates = [
                    metadata.get('date'),
                    metadata.get('DATE'),
                    metadata.get('year'),
                    metadata.get('YEAR')
                ]

                for candidate in date_candidates:
                    if candidate:
                        mix_info['date'] = candidate
                        break

                # If we found some metadata, break
                if mix_info['dj_name'] != 'Unknown DJ':
                    break

        # Build detailed tracklist with artists and titles
        for i, (mp3_file, metadata, track_num) in enumerate(mp3_files, 1):
            track_artist = 'Unknown Artist'
            track_title = 'Unknown Title'

            if metadata:
                # Get individual track artist
                track_artist = (metadata.get('artist') or metadata.get('ARTIST') or
                               self.extract_artist_from_filename(mp3_file.name))
                track_title = (metadata.get('title') or metadata.get('TITLE') or
                              self.extract_title_from_filename(mp3_file.name))
            else:
                # Fallback to filename parsing
                track_artist = self.extract_artist_from_filename(mp3_file.name)
                track_title = self.extract_title_from_filename(mp3_file.name)

            mix_info['tracklist'].append(f"{i:02d}. {track_artist} - {track_title}")

        return mix_info

    def extract_title_from_filename(self, filename: str) -> str:
        """Extract track title from filename in 'Artist - Title' format"""
        # Remove track numbers and file extension
        clean_name = re.sub(r'^\d+\s*[-\.]?\s*', '', filename)
        clean_name = re.sub(r'\.[^.]+$', '', clean_name)

        # Split on ' - ' and take second part as title
        if ' - ' in clean_name:
            parts = clean_name.split(' - ', 1)
            if len(parts) > 1:
                return parts[1].strip()

        return clean_name.strip() if clean_name else "Unknown Title"

    def create_album_folders_from_grouped_mp3s(self, grouped_albums: Dict[str, List[Path]]) -> List[Path]:
        """Create album folders and move MP3s into them"""
        created_folders = []

        for album_key, mp3_files in grouped_albums.items():
            if len(mp3_files) < 2:
                print(f"Skipping {album_key}: Only {len(mp3_files)} track(s)")
                continue

            # Clean album name for folder
            safe_album_name = re.sub(r'[^\w\s-]', '', album_key).strip()
            album_folder = self.downloads_folder / safe_album_name

            # Create folder if it doesn't exist
            album_folder.mkdir(exist_ok=True)
            print(f"Created album folder: {album_folder.name}")

            # Move MP3s into the folder
            for mp3_file in mp3_files:
                try:
                    new_path = album_folder / mp3_file.name
                    mp3_file.rename(new_path)
                    print(f"  Moved: {mp3_file.name}")
                except Exception as e:
                    print(f"  Error moving {mp3_file.name}: {e}")

            created_folders.append(album_folder)

        return created_folders

    def process_root_mp3s(self):
        """Process individual MP3s in root directory by grouping them into albums"""
        root_mp3s = self.get_root_mp3_files()

        if not root_mp3s:
            print("No individual MP3 files found in root directory")
            return []

        print(f"Found {len(root_mp3s)} individual MP3 files in root directory")

        # Group by traditional metadata first (MusicBrainz enhancement available separately)
        grouped_albums = self.group_mp3s_by_album_traditional(root_mp3s)
        print(f"Grouped into {len(grouped_albums)} potential albums:")

        for album_key, files in grouped_albums.items():
            print(f"  {album_key}: {len(files)} tracks")

        # Create album folders and move files
        return self.create_album_folders_from_grouped_mp3s(grouped_albums)

    def get_album_folders(self) -> List[Path]:
        """Get all album folders (skip playlists and processed folder)"""
        album_folders = []
        for item in self.downloads_folder.iterdir():
            if item.is_dir() and not item.name.lower().startswith('processed'):
                # Check if it looks like an album (artist - album format)
                if ' - ' in item.name and not 'liked songs' in item.name.lower():
                    album_folders.append(item)
        return album_folders

    def get_album_folders_with_mp3s(self) -> List[Path]:
        """Get folders that already contain MP3s (pre-organized by user)"""
        album_folders = []
        for item in self.downloads_folder.iterdir():
            if (item.is_dir() and
                not item.name.lower().startswith(('processed', 'merged', '.')) and
                not item.name in ['Merged Albums', 'Merged Mixes']):
                # Check if folder contains MP3 files
                mp3_files = list(item.glob("*.mp3"))
                if len(mp3_files) >= 2:  # At least 2 MP3s to be worth merging
                    album_folders.append(item)
        return album_folders

    def get_track_metadata(self, mp3_file: Path) -> Dict[str, str]:
        """Extract metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_entries', 'format_tags', str(mp3_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return data.get('format', {}).get('tags', {})
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            return {}

    def extract_track_number(self, filename: str, metadata: Dict[str, str]) -> int:
        """Extract track number from metadata or filename"""
        # Try metadata first - check various tag formats
        for tag_name in ['track', 'TRACK', 'tracknumber', 'TRACKNUMBER']:
            if tag_name in metadata:
                track_str = metadata[tag_name]
                # Handle "1/14" format
                if '/' in track_str:
                    track_str = track_str.split('/')[0]
                try:
                    return int(track_str)
                except ValueError:
                    continue

        # Fallback to filename parsing
        # Look for patterns like "01 -", "Track 1", etc.
        patterns = [
            r'^(\d{1,2})\s*[-\.]',  # "01 - Title" or "1. Title"
            r'Track\s*(\d+)',       # "Track 01"
            r'(\d{1,2})\s*\.',      # "01. Title"
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return 999  # Default for unmatched tracks

    def get_mp3_files_with_metadata(self, album_folder: Path) -> List[Tuple[Path, Dict[str, str], int]]:
        """Get MP3 files with their metadata and track numbers"""
        mp3_files = []

        for mp3_file in album_folder.glob("*.mp3"):
            try:
                metadata = self.get_track_metadata(mp3_file)
                track_num = self.extract_track_number(mp3_file.name, metadata)
                mp3_files.append((mp3_file, metadata, track_num))
                print(f"  Track {track_num:2d}: {mp3_file.name}")
            except Exception as e:
                print(f"Warning: Could not read metadata from {mp3_file.name}: {e}")
                track_num = self.extract_track_number(mp3_file.name, {})
                mp3_files.append((mp3_file, {}, track_num))

        # Sort by track number (will be enhanced by MusicBrainz if possible)
        mp3_files.sort(key=lambda x: x[2])
        return mp3_files

    def get_mp3_files_with_enhanced_ordering(self, album_folder: Path, use_musicbrainz: bool = True) -> List[Tuple[Path, Dict[str, str], int]]:
        """Get MP3 files with metadata and enhanced track ordering using MusicBrainz"""
        # Start with basic metadata extraction
        mp3_files = self.get_mp3_files_with_metadata(album_folder)

        if not use_musicbrainz or len(mp3_files) < 2:
            return mp3_files

        # Extract album info for MusicBrainz lookup
        album_info = self.extract_album_info(mp3_files)
        artist = album_info.get('artist', '').strip()
        album = album_info.get('album', '').strip()

        if not artist or not album or artist.lower() == 'unknown artist' or album.lower() == 'unknown album':
            print("❌ Missing artist/album info, skipping MusicBrainz lookup")
            return mp3_files

        # Get track titles for lookup
        track_titles = []
        for mp3_file, metadata, _ in mp3_files:
            title = metadata.get('title', mp3_file.stem).strip()
            track_titles.append(title)

        # Lookup correct track order from MusicBrainz
        musicbrainz_order = self.lookup_musicbrainz_track_order(artist, album, track_titles)

        if musicbrainz_order:
            print("🎵 Applying MusicBrainz track ordering...")
            # Update track numbers based on MusicBrainz data
            enhanced_mp3_files = []
            for mp3_file, metadata, original_track_num in mp3_files:
                title = metadata.get('title', mp3_file.stem).strip()

                if title in musicbrainz_order:
                    new_track_num = musicbrainz_order[title]
                    enhanced_mp3_files.append((mp3_file, metadata, new_track_num))
                    print(f"  Updated track {original_track_num} → {new_track_num}: {title}")
                else:
                    # Keep original track number if not found in MusicBrainz
                    enhanced_mp3_files.append((mp3_file, metadata, original_track_num))
                    print(f"  Kept track {original_track_num}: {title}")

            # Sort by the enhanced track numbers
            enhanced_mp3_files.sort(key=lambda x: x[2])
            return enhanced_mp3_files
        else:
            print("⚠️  MusicBrainz lookup failed, using original track order")
            return mp3_files

    def lookup_musicbrainz_release_info(self, artist: str, album: str) -> Optional[Dict]:
        """
        Get complete release information from MusicBrainz including all tracks
        This helps identify complete releases that might be split across metadata
        """
        try:
            print(f"🔍 Looking up complete release info for '{artist} - {album}'...")

            # Search for the release
            result = musicbrainzngs.search_releases(
                artist=artist,
                release=album,
                limit=5
            )

            if not result.get('release-list'):
                return None

            # Try each release until we find a good match
            for release in result['release-list']:
                try:
                    # Get detailed release info including tracks
                    detailed_release = musicbrainzngs.get_release_by_id(
                        release['id'],
                        includes=['recordings', 'artist-credits']
                    )

                    if 'medium-list' not in detailed_release['release']:
                        continue

                    # Extract complete track listing
                    all_tracks = []
                    for medium in detailed_release['release']['medium-list']:
                        if 'track-list' in medium:
                            for track in medium['track-list']:
                                # Safely extract track artist
                                track_artist = artist  # Default to album artist
                                if track['recording'].get('artist-credit'):
                                    try:
                                        track_artist = track['recording']['artist-credit'][0]['name']
                                    except (KeyError, IndexError, TypeError):
                                        pass  # Keep default
                                
                                track_info = {
                                    'position': int(track['position']),
                                    'title': track['recording']['title'],
                                    'artist': track_artist,
                                    'length': track['recording'].get('length')
                                }
                                all_tracks.append(track_info)

                    if all_tracks:
                        # Safely extract release artist
                        release_artist = artist  # Default
                        if detailed_release['release'].get('artist-credit'):
                            try:
                                release_artist = detailed_release['release']['artist-credit'][0]['name']
                            except (KeyError, IndexError, TypeError):
                                pass  # Keep default
                        
                        return {
                            'release_id': release['id'],
                            'title': detailed_release['release']['title'],
                            'artist': release_artist,
                            'tracks': all_tracks,
                            'track_count': len(all_tracks),
                            'date': detailed_release['release'].get('date', '')
                        }

                except Exception as e:
                    print(f"Error processing release {release.get('id', 'unknown')}: {e}")
                    continue

            return None

        except Exception as e:
            print(f"❌ MusicBrainz release lookup failed: {e}")
            return None

    def lookup_musicbrainz_track_order(self, artist: str, album: str, track_titles: List[str]) -> Optional[Dict[str, int]]:
        """
        Lookup track order from MusicBrainz database
        Returns a dict mapping track titles to their correct track numbers
        """
        try:
            print(f"🔍 Looking up '{artist} - {album}' on MusicBrainz...")

            # Search for the release
            result = musicbrainzngs.search_releases(
                artist=artist,
                release=album,
                limit=5
            )

            if not result.get('release-list'):
                print(f"❌ No releases found for '{artist} - {album}'")
                return None

            # Try each release until we find a good match
            for release in result['release-list']:
                try:
                    # Get detailed release info including tracks
                    detailed_release = musicbrainzngs.get_release_by_id(
                        release['id'],
                        includes=['recordings']
                    )

                    if 'medium-list' not in detailed_release['release']:
                        continue

                    # Build track mapping from MusicBrainz data
                    mb_track_mapping = {}
                    for medium in detailed_release['release']['medium-list']:
                        if 'track-list' in medium:
                            for track in medium['track-list']:
                                track_title = track['recording']['title'].lower().strip()
                                track_number = int(track['position'])
                                mb_track_mapping[track_title] = track_number

                    # Match our tracks to MusicBrainz tracks
                    track_order = {}
                    matches = 0

                    for local_title in track_titles:
                        local_clean = local_title.lower().strip()

                        # Try exact match first
                        if local_clean in mb_track_mapping:
                            track_order[local_title] = mb_track_mapping[local_clean]
                            matches += 1
                            continue

                        # Try fuzzy matching (removing common differences)
                        for mb_title, track_num in mb_track_mapping.items():
                            # Remove common variations
                            local_normalized = re.sub(r'[^\w\s]', '', local_clean)
                            mb_normalized = re.sub(r'[^\w\s]', '', mb_title)

                            if local_normalized == mb_normalized:
                                track_order[local_title] = track_num
                                matches += 1
                                break

                    # If we matched most tracks, consider it successful
                    match_percentage = matches / len(track_titles) if track_titles else 0
                    if match_percentage >= 0.7:  # 70% match threshold
                        print(f"✅ Found track order from MusicBrainz ({matches}/{len(track_titles)} tracks matched)")
                        return track_order

                except Exception as e:
                    print(f"Error processing release {release.get('id', 'unknown')}: {e}")
                    continue

            print(f"❌ No suitable match found for '{artist} - {album}'")
            return None

        except Exception as e:
            print(f"❌ MusicBrainz lookup failed: {e}")
            return None

    def extract_album_info(self, mp3_files: List[Tuple[Path, Dict[str, str], int]]) -> Dict[str, str]:
        """Extract album information from the first track with metadata"""
        album_info = {
            'artist': 'Unknown Artist',
            'album': 'Unknown Album',
            'year': '',
        }

        for mp3_file, metadata, _ in mp3_files:
            if metadata:
                # Check various tag formats for artist
                for tag in ['album_artist', 'ALBUM_ARTIST', 'artist', 'ARTIST', 'albumartist', 'ALBUMARTIST']:
                    if tag in metadata:
                        album_info['artist'] = metadata[tag]
                        break

                # Check various tag formats for album
                for tag in ['album', 'ALBUM']:
                    if tag in metadata:
                        album_info['album'] = metadata[tag]
                        break

                # Check for year/date
                for tag in ['date', 'DATE', 'year', 'YEAR']:
                    if tag in metadata:
                        album_info['year'] = metadata[tag]
                        break

                # If we found some metadata, break
                if album_info['artist'] != 'Unknown Artist' or album_info['album'] != 'Unknown Album':
                    break

        return album_info

    def create_file_list(self, mp3_files: List[Tuple[Path, Dict[str, str], int]], temp_dir: Path) -> Path:
        """Create a file list for ffmpeg concat"""
        file_list_path = temp_dir / "file_list.txt"

        with open(file_list_path, 'w', encoding='utf-8') as f:
            for mp3_file, _, track_num in mp3_files:
                # Use absolute path and escape single quotes for ffmpeg
                abs_path = mp3_file.absolute()
                escaped_path = str(abs_path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        return file_list_path

    def delete_source_files(self, mp3_files: List[Tuple[Path, Dict[str, str], int]]):
        """Delete the original MP3 files after successful merge"""
        print("Deleting original MP3 files...")
        for mp3_file, _, _ in mp3_files:
            try:
                if hasattr(self, 'delete_originals') and self.delete_originals:
                    mp3_file.unlink()
                    print(f"  Deleted: {mp3_file.name}")
                else:
                    print(f"  Keeping: {mp3_file.name} (delete_originals=False)")
            except Exception as e:
                print(f"  Error deleting {mp3_file.name}: {e}")

    def cleanup_empty_folder(self, folder: Path):
        """Remove empty album folder after processing"""
        try:
            # Check if folder is empty (only contains . and ..)
            if folder.exists() and folder.is_dir():
                contents = list(folder.iterdir())
                if not contents:  # Folder is empty
                    folder.rmdir()
                    print(f"  ✓ Removed empty folder: {folder.name}")
        except Exception as e:
            print(f"  ✗ Failed to remove empty folder {folder.name}: {e}")

    def cleanup_all_empty_folders(self):
        """Clean up all empty folders in downloads directory"""
        print("Cleaning up empty folders...")
        for item in self.downloads_folder.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != "Merged Albums":
                try:
                    contents = list(item.iterdir())
                    if not contents:  # Folder is empty
                        item.rmdir()
                        print(f"  ✓ Removed empty folder: {item.name}")
                except Exception as e:
                    print(f"  ✗ Failed to remove empty folder {item.name}: {e}")

    def merge_album(self, album_folder: Path, output_folder: Optional[Path] = None, delete_originals: bool = False) -> bool:
        """Merge all MP3s in an album folder"""
        if not self.check_ffmpeg():
            print("Error: ffmpeg is required but not found. Install with: brew install ffmpeg")
            return False

        print(f"Processing: {album_folder.name}")

        # Get MP3 files with enhanced MusicBrainz ordering
        mp3_files = self.get_mp3_files_with_enhanced_ordering(album_folder, use_musicbrainz=True)

        if len(mp3_files) < 2:
            print(f"Skipping {album_folder.name}: Less than 2 MP3 files found")
            return False

        # Detect if this is a mix or album
        file_paths = [mp3_file for mp3_file, _, _ in mp3_files]
        content_type = self.detect_mix_type(file_paths)

        if content_type == "mix":
            return self.merge_mix(mp3_files, album_folder, output_folder, delete_originals)
        else:
            return self.merge_album_traditional(mp3_files, album_folder, output_folder, delete_originals)

    def merge_mix(self, mp3_files: List[Tuple[Path, Dict[str, str], int]], album_folder: Path, output_folder: Optional[Path] = None, delete_originals: bool = False) -> bool:
        """Merge MP3s as a DJ mix/set"""
        print("Detected as: DJ Mix/Set")

        # Extract mix info
        mix_info = self.extract_mix_info(mp3_files)

        # Create output filename for mix
        safe_dj_name = re.sub(r'[^\w\s-]', '', mix_info['dj_name']).strip()
        safe_mix_title = re.sub(r'[^\w\s-]', '', mix_info['mix_title']).strip()

        if mix_info['date']:
            output_filename = f"{safe_dj_name} - {safe_mix_title} ({mix_info['date']}).mp3"
        else:
            output_filename = f"{safe_dj_name} - {safe_mix_title}.mp3"

        # Set output directory
        if output_folder is None:
            output_folder = self.downloads_folder / "Merged Mixes"
        else:
            output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        output_path = output_folder / output_filename

        # Create temporary directory for file list
        temp_dir = Path("/tmp/album_merger")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Create file list for ffmpeg
            file_list_path = self.create_file_list(mp3_files, temp_dir)

            print(f"Merging {len(mp3_files)} tracks...")
            print("Tracklist:")
            for track in mix_info['tracklist']:
                print(f"  {track}")

            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', str(file_list_path),
                '-c', 'copy',
                '-y',  # Overwrite output file
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error during merge: {result.stderr}")
                return False

            # Extract album art from first track
            album_art_path = self.extract_album_art(mp3_files, temp_dir)

            # Add metadata and album art to merged mix file
            self.add_metadata_to_merged_mix(output_path, mix_info, album_art_path)

            # Create tracklist file
            self.create_mix_tracklist_file(output_path, mix_info)

            #             # Delete original MP3 files after successful merge
            # Only delete originals if requested
            # Only delete originals if requested
            if delete_originals:
                self.delete_source_files(mp3_files)
                # Clean up empty folder after deleting files
                self.cleanup_empty_folder(album_folder)
                            #             self.delete_source_files(mp3_files)
            # 
            #             # Clean up empty folder after deleting files
            #             self.cleanup_empty_folder(album_folder)

            print(f"✓ Merged mix saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error merging mix: {e}")
            return False
        finally:
            # Cleanup
            if file_list_path.exists():
                file_list_path.unlink()
            # Clean up album art file
            album_art_file = temp_dir / "album_art.jpg"
            if album_art_file.exists():
                album_art_file.unlink()

    def merge_album_traditional(self, mp3_files: List[Tuple[Path, Dict[str, str], int]], album_folder: Path, output_folder: Optional[Path] = None, delete_originals: bool = False) -> bool:
        """Merge MP3s as a traditional album"""
        print("Detected as: Album")

        # Extract album info
        album_info = self.extract_album_info(mp3_files)

        # Create output filename
        safe_album_name = re.sub(r'[^\w\s-]', '', album_info['album']).strip()
        safe_artist_name = re.sub(r'[^\w\s-]', '', album_info['artist']).strip()
        output_filename = f"{safe_artist_name} - {safe_album_name}.mp3"

        # Set output directory
        if output_folder is None:
            output_folder = self.downloads_folder / "Merged Albums"
        else:
            output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        output_path = output_folder / output_filename

        # Create temporary directory for file list
        temp_dir = Path("/tmp/album_merger")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Create file list for ffmpeg
            file_list_path = self.create_file_list(mp3_files, temp_dir)

            print(f"Merging {len(mp3_files)} tracks...")
            for i, (mp3_file, _, track_num) in enumerate(mp3_files, 1):
                print(f"  {i:2d}. {mp3_file.name}")

            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', str(file_list_path),
                '-c', 'copy',
                '-y',  # Overwrite output file
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error during merge: {result.stderr}")
                return False

            # Extract album art from first track
            album_art_path = self.extract_album_art(mp3_files, temp_dir)

            # Add metadata and album art to merged file using ffmpeg
            self.add_metadata_to_merged_file(output_path, album_info, len(mp3_files), album_art_path)

            # Delete original MP3 files after successful merge
            # Only delete originals if requested
            # Only delete originals if requested
            if delete_originals:
                self.delete_source_files(mp3_files)
                # Clean up empty folder after deleting files
            self.cleanup_empty_folder(album_folder)

            # Clean up empty folder after deleting files
            self.cleanup_empty_folder(album_folder)

            print(f"✓ Merged album saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error merging album: {e}")
            return False
        finally:
            # Cleanup
            if file_list_path.exists():
                file_list_path.unlink()
            # Clean up album art file
            album_art_file = temp_dir / "album_art.jpg"
            if album_art_file.exists():
                album_art_file.unlink()

    def extract_album_art(self, mp3_files: List[Tuple[Path, Dict[str, str], int]], temp_dir: Path) -> Optional[Path]:
        """Extract album art from the first track that has it"""
        for mp3_file, metadata, track_num in mp3_files:
            try:
                album_art_path = temp_dir / "album_art.jpg"

                # Try to extract album art using ffmpeg
                cmd = [
                    'ffmpeg', '-i', str(mp3_file),
                    '-an', '-vcodec', 'copy',
                    '-y', str(album_art_path)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0 and album_art_path.exists():
                    print(f"✓ Extracted album art from: {mp3_file.name}")
                    return album_art_path

            except Exception as e:
                continue

        print("⚠ No album art found in tracks")
        return None

    def add_metadata_to_merged_file(self, output_path: Path, album_info: Dict[str, str], track_count: int, album_art_path: Optional[Path] = None):
        """Add metadata and album art to the merged MP3 file using ffmpeg"""
        try:
            temp_output = output_path.with_suffix('.tmp.mp3')

            # Build ffmpeg command
            cmd = ['ffmpeg']

            # Input file
            cmd.extend(['-i', str(output_path)])

            # Add album art if available
            if album_art_path and album_art_path.exists():
                cmd.extend(['-i', str(album_art_path)])
                cmd.extend(['-map', '0:a', '-map', '1:v'])
                cmd.extend(['-c:a', 'copy', '-c:v', 'copy'])
            else:
                cmd.extend(['-c', 'copy'])

            # Add metadata
            cmd.extend([
                '-metadata', f'title={album_info["album"]} (Full Album)',
                '-metadata', f'artist={album_info["artist"]}',
                '-metadata', f'album={album_info["album"]}',
            ])

            if album_info['year']:
                cmd.extend(['-metadata', f'date={album_info["year"]}'])

            cmd.extend(['-y', str(temp_output)])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Replace original with temp file
                temp_output.replace(output_path)
                if album_art_path:
                    print(f"✓ Metadata and album art added to merged file")
                else:
                    print(f"✓ Metadata added to merged file")
            else:
                print(f"Warning: Could not add metadata: {result.stderr}")
                if temp_output.exists():
                    temp_output.unlink()

        except Exception as e:
            print(f"Warning: Could not add metadata to merged file: {e}")

    def add_metadata_to_merged_mix(self, output_path: Path, mix_info: Dict[str, str], album_art_path: Optional[Path] = None):
        """Add metadata and album art to the merged mix file using ffmpeg"""
        try:
            temp_output = output_path.with_suffix('.tmp.mp3')

            # Build ffmpeg command
            cmd = ['ffmpeg']

            # Input file
            cmd.extend(['-i', str(output_path)])

            # Add album art if available
            if album_art_path and album_art_path.exists():
                cmd.extend(['-i', str(album_art_path)])
                cmd.extend(['-map', '0:a', '-map', '1:v'])
                cmd.extend(['-c:a', 'copy', '-c:v', 'copy'])
            else:
                cmd.extend(['-c', 'copy'])

            # Add metadata
            cmd.extend([
                '-metadata', f'title={mix_info["mix_title"]} (DJ Mix)',
                '-metadata', f'artist={mix_info["dj_name"]}',
                '-metadata', f'album={mix_info["mix_title"]}',
                '-metadata', f'albumartist={mix_info["dj_name"]}',
                '-metadata', f'genre=DJ Mix',
            ])

            if mix_info['date']:
                cmd.extend(['-metadata', f'date={mix_info["date"]}'])

            # Add track count as comment
            cmd.extend(['-metadata', f'comment={mix_info["total_tracks"]} tracks'])

            cmd.extend(['-y', str(temp_output)])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Replace original with temp file
                temp_output.replace(output_path)
                if album_art_path:
                    print(f"✓ Mix metadata and album art added to merged file")
                else:
                    print(f"✓ Mix metadata added to merged file")
            else:
                print(f"Warning: Could not add mix metadata: {result.stderr}")
                if temp_output.exists():
                    temp_output.unlink()

        except Exception as e:
            print(f"Warning: Could not add metadata to merged mix file: {e}")

    def create_mix_tracklist_file(self, output_path: Path, mix_info: Dict[str, str]):
        """Create a tracklist text file alongside the merged mix"""
        try:
            tracklist_path = output_path.with_suffix('.txt')

            with open(tracklist_path, 'w', encoding='utf-8') as f:
                f.write(f"DJ: {mix_info['dj_name']}\n")
                f.write(f"Mix: {mix_info['mix_title']}\n")
                if mix_info['date']:
                    f.write(f"Date: {mix_info['date']}\n")
                f.write(f"Tracks: {mix_info['total_tracks']}\n")
                f.write("\n" + "="*50 + "\n")
                f.write("TRACKLIST\n")
                f.write("="*50 + "\n\n")

                for track in mix_info['tracklist']:
                    f.write(f"{track}\n")

            print(f"✓ Tracklist saved to: {tracklist_path.name}")

        except Exception as e:
            print(f"Warning: Could not create tracklist file: {e}")

    def merge_all_albums(self, output_folder: Optional[Path] = None):
        """Process root MP3s and merge all albums in the downloads folder"""
        # First, check for pre-organized album folders with MP3s
        print("=== Checking for pre-organized album folders ===")
        existing_album_folders = self.get_album_folders_with_mp3s()
        if existing_album_folders:
            print(f"Found {len(existing_album_folders)} pre-organized album folders:")
            for folder in existing_album_folders:
                mp3_count = len(list(folder.glob("*.mp3")))
                print(f"  - {folder.name} ({mp3_count} tracks)")
        else:
            print("No pre-organized album folders found")
        print()

        # Then, process individual MP3s in root directory
        print("=== Processing individual MP3s in root directory ===")
        newly_created_folders = self.process_root_mp3s()
        print()

        # Combine both types of folders
        all_album_folders = existing_album_folders + newly_created_folders

        if not all_album_folders:
            print("No album folders found in downloads directory")
            return

        print(f"=== Found {len(all_album_folders)} album folders to merge ===")
        for folder in all_album_folders:
            mp3_count = len(list(folder.glob("*.mp3")))
            print(f"  - {folder.name} ({mp3_count} tracks)")
        print()

        successful = 0
        failed = 0

        for album_folder in all_album_folders:
            if self.merge_album(album_folder, output_folder):
                successful += 1
            else:
                failed += 1
            print()

        print(f"Merge complete: {successful} successful, {failed} failed")

        # Clean up any remaining empty folders
        self.cleanup_all_empty_folders()

    def merge_selected_albums(self, selected_albums: List[str], output_folder: Optional[Path] = None):
        """Merge only selected albums (for web interface)"""
        # For now, merge all available albums
        # This can be extended to support selective merging
        self.merge_all_albums(output_folder)


def main():
    parser = argparse.ArgumentParser(description="Merge spotDL album downloads into single files")
    parser.add_argument(
        "--downloads-folder",
        default=".",
        help="Path to spotDL downloads folder (default: current directory)"
    )
    parser.add_argument(
        "--output-folder",
        help="Output folder for merged albums (default: downloads-folder/Merged Albums)"
    )
    parser.add_argument(
        "--album",
        help="Merge specific album folder only"
    )

    args = parser.parse_args()

    try:
        merger = AlbumMerger(args.downloads_folder)

        if args.album:
            album_path = Path(args.downloads_folder) / args.album
            if not album_path.exists():
                print(f"Album folder not found: {album_path}")
                return

            output_folder = Path(args.output_folder) if args.output_folder else None
            merger.merge_album(album_path, output_folder)
        else:
            output_folder = Path(args.output_folder) if args.output_folder else None
            merger.merge_all_albums(output_folder)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()