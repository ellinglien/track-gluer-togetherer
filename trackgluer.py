#!/usr/bin/env python3
"""
Album Merger Web Interface - Production Ready Version
"""

import os
import json
import webbrowser
import threading
import time
import sys
import logging
import secrets
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
from threading import Lock, Timer
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import uuid

# Configuration Management
@dataclass
class AppConfig:
    """Application configuration with environment variable support"""
    SECRET_KEY: str = field(default_factory=lambda: os.environ.get('SECRET_KEY', secrets.token_hex(32)))
    PORT: int = field(default_factory=lambda: int(os.environ.get('PORT', '9876')))
    DEBUG: bool = field(default_factory=lambda: os.environ.get('DEBUG', 'False').lower() == 'true')
    LOG_LEVEL: str = field(default_factory=lambda: os.environ.get('LOG_LEVEL', 'INFO'))
    LOG_FILE: str = field(default_factory=lambda: os.environ.get('LOG_FILE', 'trackgluer.log'))
    SESSION_TTL_MINUTES: int = field(default_factory=lambda: int(os.environ.get('SESSION_TTL_MINUTES', '30')))
    MAX_SESSIONS: int = field(default_factory=lambda: int(os.environ.get('MAX_SESSIONS', '10')))
    CACHE_SIZE: int = field(default_factory=lambda: int(os.environ.get('CACHE_SIZE', '128')))
    FFMPEG_TIMEOUT: int = field(default_factory=lambda: int(os.environ.get('FFMPEG_TIMEOUT', '300')))
    MAX_FILE_SIZE_MB: int = field(default_factory=lambda: int(os.environ.get('MAX_FILE_SIZE_MB', '500')))
    WORKER_THREADS: int = field(default_factory=lambda: int(os.environ.get('WORKER_THREADS', '4')))
    BATCH_SIZE: int = field(default_factory=lambda: int(os.environ.get('BATCH_SIZE', '50')))

# Initialize configuration
config = AppConfig()

# Configure logging with config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Application starting with config: PORT={config.PORT}, DEBUG={config.DEBUG}")

# Import our existing AlbumMerger class
from merge_albums import AlbumMerger

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Enhanced Thread-safe session management
class SessionManager:
    def __init__(self, ttl_minutes: int = None, max_sessions: int = None):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.progress: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        self.ttl = timedelta(minutes=ttl_minutes or config.SESSION_TTL_MINUTES)
        self.max_sessions = max_sessions or config.MAX_SESSIONS
        self._start_cleanup_timer()
    
    def _start_cleanup_timer(self):
        """Start periodic cleanup of expired sessions"""
        def cleanup():
            self.cleanup_expired()
            self._start_cleanup_timer()
        
        timer = Timer(300.0, cleanup)  # Cleanup every 5 minutes
        timer.daemon = True
        timer.start()
    
    def cleanup_expired(self):
        """Remove expired sessions"""
        with self.lock:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session_data in self.sessions.items():
                if 'created_at' in session_data:
                    if current_time - session_data['created_at'] > self.ttl:
                        expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                logger.info(f"Cleaning up expired session: {session_id}")
                self.sessions.pop(session_id, None)
                self.progress.pop(session_id, None)
    
    def create_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Create a new session with timestamp and session limits"""
        with self.lock:
            # Check session limit
            if len(self.sessions) >= self.max_sessions:
                logger.warning(f"Session limit reached ({self.max_sessions}), cleaning up oldest sessions")
                self._cleanup_oldest_sessions()
            
            data['created_at'] = datetime.now()
            data['last_accessed'] = datetime.now()
            self.sessions[session_id] = data
            logger.info(f"Created session: {session_id} (total: {len(self.sessions)})")
            return True
    
    def _cleanup_oldest_sessions(self):
        """Remove oldest sessions to make room for new ones"""
        if not self.sessions:
            return
            
        # Sort by creation time and remove oldest
        sorted_sessions = sorted(
            self.sessions.items(), 
            key=lambda x: x[1].get('created_at', datetime.min)
        )
        
        # Remove oldest 25% of sessions
        remove_count = max(1, len(sorted_sessions) // 4)
        for i in range(remove_count):
            session_id = sorted_sessions[i][0]
            logger.info(f"Removing oldest session: {session_id}")
            self.sessions.pop(session_id, None)
            self.progress.pop(session_id, None)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        with self.lock:
            return self.sessions.get(session_id)
    
    def update_progress(self, session_id: str, progress_data: Dict[str, Any]) -> None:
        """Update progress for a session"""
        with self.lock:
            self.progress[session_id] = progress_data
    
    def get_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get progress for a session"""
        with self.lock:
            return self.progress.get(session_id)

# Rate Limiting
class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self.lock = Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        with self.lock:
            now = time.time()
            
            # Clean old entries
            if client_id in self.requests:
                self.requests[client_id] = [
                    req_time for req_time in self.requests[client_id]
                    if now - req_time < self.window_seconds
                ]
            else:
                self.requests[client_id] = []
            
            # Check rate limit
            if len(self.requests[client_id]) >= self.max_requests:
                logger.warning(f"Rate limit exceeded for client {client_id}")
                return False
            
            # Add current request
            self.requests[client_id].append(now)
            return True

# Background Task Queue
class BackgroundTaskQueue:
    """Thread pool for CPU/IO intensive operations"""
    
    def __init__(self, max_workers: int = None):
        self.executor = ThreadPoolExecutor(max_workers=max_workers or config.WORKER_THREADS)
        self.tasks = {}
        self.lock = Lock()
    
    def submit_task(self, task_id: str, func, *args, **kwargs):
        """Submit a task to the background queue"""
        with self.lock:
            future = self.executor.submit(func, *args, **kwargs)
            self.tasks[task_id] = {
                'future': future,
                'submitted_at': datetime.now(),
                'status': 'running'
            }
            logger.info(f"Submitted background task: {task_id}")
            return future
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a background task"""
        with self.lock:
            if task_id not in self.tasks:
                return {'status': 'not_found'}
            
            task = self.tasks[task_id]
            future = task['future']
            
            if future.done():
                try:
                    result = future.result()
                    task['status'] = 'completed'
                    task['result'] = result
                    return {'status': 'completed', 'result': result}
                except Exception as e:
                    task['status'] = 'failed'
                    task['error'] = str(e)
                    return {'status': 'failed', 'error': str(e)}
            else:
                return {'status': 'running', 'submitted_at': task['submitted_at']}
    
    def cleanup_completed_tasks(self):
        """Remove completed tasks to free memory"""
        with self.lock:
            completed_tasks = [
                task_id for task_id, task in self.tasks.items()
                if task['future'].done()
            ]
            for task_id in completed_tasks:
                self.tasks.pop(task_id, None)
            
            if completed_tasks:
                logger.debug(f"Cleaned up {len(completed_tasks)} completed tasks")

# Initialize components
task_queue = BackgroundTaskQueue()
rate_limiter = RateLimiter()
session_manager = SessionManager()

# Legacy compatibility - will be removed
active_sessions = session_manager.sessions
merge_progress = session_manager.progress

class WebAlbumMerger(AlbumMerger):
    """Extended AlbumMerger with web interface support"""

    def __init__(self, downloads_folder: str = "."):
        super().__init__(downloads_folder)
        self.session_id = None

    def preview_groupings(self):
        """Get a preview of how files will be grouped without processing"""
        print("🔍 DEBUG: preview_groupings called")
        # Get loose MP3 files
        root_mp3s = self.get_root_mp3_files()
        print(f"🔍 DEBUG: Found {len(root_mp3s)} root mp3 files")

        # Get pre-organized folders
        existing_folders = self.get_album_folders_with_mp3s()

        preview = {
            'loose_files': {
                'count': len(root_mp3s),
                'groupings': {}
            },
            'existing_folders': [],
            'total_albums': 0
        }

        # Preview loose file groupings using traditional approach
        if root_mp3s:
            grouped_albums = self.group_mp3s_by_album_traditional(root_mp3s)
            individual_tracks = []
            
            print(f"🔍 Traditional grouping found {len(grouped_albums)} album groups:")
            for album_key, files in grouped_albums.items():
                print(f"  - {album_key}: {len(files)} tracks ({[f.name for f in files]})")
                
                if len(files) >= 2:  # Multi-track albums
                    preview['loose_files']['groupings'][album_key] = {
                        'files': [f.name for f in files],
                        'count': len(files),
                        'type': self.detect_mix_type(files)
                    }
                else:  # Single tracks - collect for custom album
                    individual_tracks.extend(files)
            
            # Add individual tracks as a custom album collection
            if individual_tracks:
                print(f"🔍 Collected {len(individual_tracks)} individual tracks for custom album")
                preview['loose_files']['groupings']['🎵 Individual Tracks'] = {
                    'files': [f.name for f in individual_tracks],
                    'count': len(individual_tracks),
                    'type': 'custom'
                }

        # Preview existing folders
        for folder in existing_folders:
            mp3_files = list(folder.glob("*.mp3"))
            file_paths = [mp3_file for mp3_file in mp3_files]
            content_type = self.detect_mix_type(file_paths) if len(mp3_files) >= 3 else "album"

            preview['existing_folders'].append({
                'name': folder.name,
                'files': [f.name for f in mp3_files],
                'count': len(mp3_files),
                'type': content_type
            })

        # Count actual displayable albums (not the original grouping count)
        displayable_albums = len(preview['loose_files']['groupings']) + len(preview['existing_folders'])
        preview['total_albums'] = displayable_albums
        
        print(f"🔍 FINAL PREVIEW SUMMARY:")
        print(f"  - Total albums: {preview['total_albums']}")
        print(f"  - Loose file groupings: {len(preview['loose_files']['groupings'])}")
        print(f"  - Existing folders: {len(preview['existing_folders'])}")
        print(f"  - Grouping keys: {list(preview['loose_files']['groupings'].keys())}")
        
        return preview

# Memory-Efficient File Operations
class FileSystemOperations:
    """Memory-efficient file system operations with generators"""
    
    @staticmethod
    def scan_mp3_files_generator(folder_path: str):
        """Generator for MP3 files to avoid loading all into memory"""
        try:
            path = Path(folder_path)
            for mp3_file in path.glob("*.mp3"):
                if validate_file_size(mp3_file):
                    yield mp3_file
                else:
                    logger.warning(f"Skipping oversized file: {mp3_file}")
        except Exception as e:
            logger.error(f"Error scanning {folder_path}: {e}")
    
    @staticmethod
    def batch_process_files(files, batch_size: int = None):
        """Process files in batches to manage memory usage"""
        batch_size = batch_size or config.BATCH_SIZE
        batch = []
        for file_path in files:
            batch.append(file_path)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # Don't forget the last batch
            yield batch
    
    @staticmethod
    def parallel_metadata_extraction(file_paths: list, max_workers: int = None) -> Generator[Dict[str, Any], None, None]:
        """Extract metadata from multiple files in parallel"""
        max_workers = max_workers or config.WORKER_THREADS
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(FileSystemCache.get_file_metadata, str(file_path)): file_path
                for file_path in file_paths
            }
            
            # Yield results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    metadata = future.result()
                    metadata['file_path'] = str(file_path)
                    yield metadata
                except Exception as e:
                    logger.error(f"Error extracting metadata from {file_path}: {e}")
                    yield {
                        'file_path': str(file_path),
                        'artist': "Unknown",
                        'album': "Unknown",
                        'title': file_path.stem,
                        'error': str(e)
                    }

# File System Caching
class FileSystemCache:
    """LRU cache for expensive file system operations"""
    
    @staticmethod
    @lru_cache(maxsize=config.CACHE_SIZE)
    def get_mp3_files(folder_path: str) -> tuple:
        """Cached MP3 file listing - use for small directories"""
        try:
            path = Path(folder_path)
            mp3_files = []
            for mp3_file in FileSystemOperations.scan_mp3_files_generator(folder_path):
                mp3_files.append(mp3_file)
                # Limit cached results to prevent memory issues
                if len(mp3_files) > 1000:
                    logger.warning(f"Large directory {folder_path}, truncating cache")
                    break
            
            result = tuple(mp3_files)
            logger.debug(f"Cached MP3 scan for {folder_path}: {len(result)} files")
            return result
        except Exception as e:
            logger.error(f"Error scanning {folder_path}: {e}")
            return tuple()
    
    @staticmethod
    @lru_cache(maxsize=config.CACHE_SIZE)
    def get_file_metadata(file_path: str) -> Dict[str, Any]:
        """Cached metadata extraction"""
        try:
            import eyed3
            audiofile = eyed3.load(file_path)
            if audiofile and audiofile.tag:
                metadata = {
                    'artist': audiofile.tag.artist or "Unknown",
                    'album': audiofile.tag.album or "Unknown", 
                    'title': audiofile.tag.title or Path(file_path).stem,
                    'track_num': audiofile.tag.track_num[0] if audiofile.tag.track_num else None,
                    'genre': audiofile.tag.genre.name if audiofile.tag.genre else None,
                    'date': str(audiofile.tag.recording_date) if audiofile.tag.recording_date else None
                }
                logger.debug(f"Cached metadata for {file_path}")
                return metadata
        except Exception as e:
            logger.debug(f"Could not extract metadata from {file_path}: {e}")
        
        return {
            'artist': "Unknown",
            'album': "Unknown", 
            'title': Path(file_path).stem,
            'track_num': None,
            'genre': None,
            'date': None
        }
    
    @staticmethod
    def clear_cache():
        """Clear all caches"""
        FileSystemCache.get_mp3_files.cache_clear()
        FileSystemCache.get_file_metadata.cache_clear()
        logger.info("File system cache cleared")

def validate_path(path_str: str) -> Path:
    """Validate and normalize path input"""
    try:
        path = Path(path_str).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid path: {path_str} - {e}")
        raise ValueError(f"Invalid path format: {path_str}")

def validate_file_size(file_path: Path) -> bool:
    """Validate file size against limits"""
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > config.MAX_FILE_SIZE_MB:
            logger.warning(f"File {file_path} exceeds size limit: {size_mb:.1f}MB > {config.MAX_FILE_SIZE_MB}MB")
            return False
        return True
    except Exception as e:
        logger.error(f"Could not check file size for {file_path}: {e}")
        return False

# Request validation decorator
def validate_request(require_json: bool = False, rate_limit: bool = True):
    """Decorator for request validation and rate limiting"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Rate limiting
            if rate_limit:
                client_id = request.remote_addr or 'unknown'
                if not rate_limiter.is_allowed(client_id):
                    return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # JSON validation
            if require_json:
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400
                try:
                    request.get_json(force=True)
                except Exception as e:
                    logger.warning(f"Invalid JSON in request: {e}")
                    return jsonify({'error': 'Invalid JSON format'}), 400
            
            return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

def merge_grouped_files(files, album_name, output_folder, session_id, progress_base, delete_originals=False, custom_track_order=None):
    """Merge a group of files that have been grouped together"""
    global merge_progress
    
    try:
        from pathlib import Path
        import tempfile
        import shutil
        import subprocess
        import re
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_folder) if output_folder else Path.cwd() / "Merged Albums"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply custom track order if provided
        ordered_files = files
        if custom_track_order:
            print(f"🎵 Applying custom track order: {custom_track_order}")
            # Create a mapping of filename to Path object
            file_map = {f.name: f for f in files}
            # Reorder files based on custom order
            ordered_files = []
            for filename in custom_track_order:
                if filename in file_map:
                    ordered_files.append(file_map[filename])
            # Add any files not in the custom order at the end
            for f in files:
                if f not in ordered_files:
                    ordered_files.append(f)
            print(f"🎵 Final track order: {[f.name for f in ordered_files]}")
        
        # Create a safe filename
        safe_album_name = re.sub(r'[^\w\s-]', '', album_name).strip()
        output_path = output_dir / f"{safe_album_name}.mp3"
        
        # Create a temporary file list for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for file_path in ordered_files:
                # Escape single quotes and wrap in quotes for the file list
                file_path_str = str(file_path).replace("'", "'\\''")
                f.write(f"file '{file_path_str}'\n")
            file_list_path = f.name
        
        # Update progress
        merge_progress[session_id].update({
            'status': 'processing',
            'progress': progress_base + 10,
            'message': f'Merging {len(files)} tracks into {album_name}...'
        })
        
        # Extract metadata from the first file for the merged album
        import eyed3
        print(f"📝 Extracting metadata from first file: {files[0].name}")
        
        metadata = {}
        album_art_data = None
        
        try:
            audiofile = eyed3.load(str(files[0]))
            if audiofile and audiofile.tag:
                metadata = {
                    'artist': audiofile.tag.artist or "Unknown Artist",
                    'album': audiofile.tag.album or album_name,
                    'genre': audiofile.tag.genre.name if audiofile.tag.genre else None,
                    'date': str(audiofile.tag.getBestDate()) if audiofile.tag.getBestDate() else None,
                    'album_artist': audiofile.tag.album_artist or audiofile.tag.artist
                }
                
                # Extract album art
                if audiofile.tag.images:
                    for image in audiofile.tag.images:
                        if image.picture_type in [3, 0]:  # Front cover or other
                            album_art_data = image.image_data
                            print(f"🎨 Found album art: {len(album_art_data)} bytes")
                            break
                
                print(f"📝 Metadata extracted: {metadata}")
        except Exception as e:
            print(f"Warning: Could not extract metadata: {e}")
            metadata = {'artist': "Unknown Artist", 'album': album_name}

        # Run ffmpeg to merge the files
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list_path,
            '-c', 'copy',
            '-y',  # Overwrite output file if it exists
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up the temporary file list
        try:
            os.unlink(file_list_path)
        except OSError as e:
            logger.warning(f"Could not delete temporary file {file_list_path}: {e}")
        
        if result.returncode != 0:
            error_msg = f"Error merging {album_name}: {result.stderr}"
            print(error_msg)
            merge_progress[session_id].update({
                'status': 'error',
                'message': error_msg
            })
            return False
        
        # Add metadata to the merged file
        print(f"📝 Adding metadata to merged file: {output_path.name}")
        try:
            merged_audiofile = eyed3.load(str(output_path))
            if merged_audiofile:
                if not merged_audiofile.tag:
                    merged_audiofile.initTag()
                
                # Use custom album name (from user input) or fallback to safe_album_name
                final_album_name = album_name if album_name != safe_album_name else metadata.get('album', album_name)
                
                # Set metadata for custom compilation
                merged_audiofile.tag.artist = "Various Artists"  # Always use Various Artists for custom albums
                merged_audiofile.tag.album = final_album_name  # Use the custom title from user
                merged_audiofile.tag.album_artist = "Various Artists"  # Set album artist to Various Artists
                merged_audiofile.tag.title = f"{final_album_name} (Full Album)"
                merged_audiofile.tag.track_num = (1, 1)  # Track 1 of 1
                
                # Keep original genre and date if available
                if metadata.get('genre'):
                    merged_audiofile.tag.genre = metadata['genre']
                if metadata.get('date'):
                    try:
                        year = int(metadata['date'][:4]) if len(metadata['date']) >= 4 else None
                        if year:
                            merged_audiofile.tag.recording_date = year
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.warning(f"Could not parse date metadata: {e}")
                        pass
                
                # Add album art from first track if we have it
                if album_art_data:
                    print(f"🎨 Adding album art from first track to merged file")
                    merged_audiofile.tag.images.set(3, album_art_data, "image/jpeg", "Front Cover")
                
                # Save the metadata
                merged_audiofile.tag.save()
                print(f"✅ Custom album metadata added: Various Artists - {final_album_name}")
                
        except Exception as e:
            print(f"Warning: Could not add metadata to merged file: {e}")
            # Don't fail the merge if metadata fails
        
        # Delete original files if requested
        if delete_originals:
            for file_path in files:
                try:
                    file_path.unlink()
                    print(f"  Deleted: {file_path.name}")
                except Exception as e:
                    print(f"  Error deleting {file_path.name}: {e}")
        
        # Update progress
        merge_progress[session_id].update({
            'progress': progress_base + 30,
            'message': f'Successfully created: {output_path.name}'
        })
        
        return True
        
    except Exception as e:
        error_msg = f"Error in merge_grouped_files: {str(e)}"
        print(error_msg)
        merge_progress[session_id].update({
            'status': 'error',
            'message': error_msg
        })
        return False


def merge_with_progress(session_id, output_folder=None, delete_originals=False, selected_albums=None, custom_track_orders=None, custom_album_titles=None, custom_album_files=None):
    """Merge albums with progress tracking"""
    global merge_progress, active_sessions

    try:
        print(f"🔧 merge_with_progress called for session: {session_id}")
        merge_progress[session_id] = {'status': 'starting', 'progress': 0, 'message': 'Initializing...'}
        print(f"📊 Progress initialized for session: {session_id}")

        # Get session data
        if session_id not in active_sessions:
            print(f"❌ Session {session_id} not found in active_sessions")
            merge_progress[session_id] = {'status': 'error', 'progress': 0, 'message': 'Session not found'}
            return

        print(f"✅ Found session data for session: {session_id}")

        session_data = active_sessions[session_id]
        merger = session_data['merger']
        grouped_albums = session_data['grouped_albums']
        preview = session_data['preview']

        print(f"🎵 Using pre-scanned albums from session (no re-scanning)")
        print(f"🎵 Available albums: {list(grouped_albums.keys())}")

        # Create virtual album data structures for grouped albums
        from pathlib import Path
        import tempfile

        # Create a simple class to hold virtual album data
        class VirtualAlbum:
            def __init__(self, name, files):
                self.album_name = name
                self.album_files = files
                self.path = Path(tempfile.mkdtemp(prefix=f"virtual_album_"))
            
            def __str__(self):
                return f"VirtualAlbum({self.album_name})"

        # Convert grouped albums to virtual albums
        existing_folders = []
        individual_tracks = []
        
        for album_name, files in grouped_albums.items():
            if len(files) >= 2:  # Multi-track albums
                virtual_album = VirtualAlbum(album_name, files)
                existing_folders.append(virtual_album)
                print(f"🎵 Virtual album: {album_name} with {len(files)} files")
            else:  # Single tracks - collect for custom album
                individual_tracks.extend(files)
        
        # Add individual tracks as a custom virtual album
        if individual_tracks:
            individual_album = VirtualAlbum("Individual Tracks", individual_tracks)
            existing_folders.append(individual_album)
            print(f"🎵 Individual tracks collection: {len(individual_tracks)} files")

        # Also add any real folders from the preview
        if preview.get('existing_folders'):
            for folder_info in preview['existing_folders']:
                folder_path = Path(session_data['folder_path']) / folder_info['name']
                if folder_path.exists():
                    existing_folders.append(folder_path)
                    print(f"🎵 Real folder: {folder_info['name']} with {folder_info['count']} files")

        if not existing_folders:
            merge_progress[session_id] = {'status': 'complete', 'progress': 100, 'message': 'No albums found to merge'}
            return

        merge_progress[session_id] = {'status': 'processing', 'progress': 10, 'message': f'Found {len(existing_folders)} albums to merge'}
        time.sleep(1)

        # Set delete_originals flag on the merger instance
        merger.delete_originals = delete_originals
        
        # Track all albums being processed
        all_albums = []
        
        # First, prepare the list of all albums to be processed
        for i, album_folder in enumerate(existing_folders):
            if hasattr(album_folder, 'album_files'):
                # Virtual album from grouped MP3s
                album_name = album_folder.album_name
                track_count = len(album_folder.album_files)
                all_albums.append({
                    'name': album_name,
                    'type': 'virtual',
                    'track_count': track_count,
                    'folder': album_folder
                })
            else:
                # Real folder
                mp3_files = list(album_folder.glob('*.mp3'))
                all_albums.append({
                    'name': album_folder.name,
                    'type': 'real',
                    'track_count': len(mp3_files),
                    'folder': album_folder
                })
        
        # Filter albums based on selected_albums if provided
        if selected_albums:
            print(f"🔍 Filtering albums based on selection: {selected_albums}")
            print(f"🔍 Available albums to match:")
            for album in all_albums:
                print(f"   - '{album['name']}' (type: {album['type']})")
            
            filtered_albums = []
            for album in all_albums:
                # Create the expected checkbox ID for this album (matching frontend logic)
                import re
                if album['type'] == 'virtual':
                    # For virtual albums, use the album name to create ID
                    # For virtual albums, use the album name to create ID
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', album['name'])
                    album_id = f"album_{clean_name}"
                else:
                    # For real folders, use the folder name
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', album['name'])
                    album_id = f"folder_{clean_name}"
                
                print(f"🔍 Generated ID for '{album['name']}': '{album_id}'")
                
                # Check if this album is selected
                if album_id in selected_albums:
                    filtered_albums.append(album)
                    print(f"✅ Selected album: {album['name']} (ID: {album_id})")
                else:
                    print(f"⏭️  Skipping unselected album: {album['name']} (ID: {album_id})")
                    # Also check for partial matches to help debug
                    for selected_id in selected_albums:
                        if album['name'].lower() in selected_id.lower() or selected_id.lower() in album['name'].lower():
                            print(f"   🔍 Potential match found: '{selected_id}' vs '{album_id}'")
            
            all_albums = filtered_albums
            print(f"📊 Processing {len(filtered_albums)} selected albums out of {len(all_albums) + len(filtered_albums)} total")
        
        if not all_albums:
            merge_progress[session_id] = {'status': 'complete', 'progress': 100, 'message': 'No albums selected for merging'}
            return

        # Now process each selected album
        processed = 0
        for i, album in enumerate(all_albums):
            progress = 30 + int((i / len(all_albums)) * 60)
            album_name = album['name']
            album_folder = album['folder']
            track_count = album['track_count']
            
            merge_progress[session_id] = {
                'status': 'processing',
                'progress': progress,
                'message': f'Merging: {album_name} ({track_count} tracks)',
                'current_album': album_name,
                'album_count': len(all_albums),
                'current_album_index': i + 1
            }
            
            print(f"🎵 Merging album {i+1}/{len(all_albums)}: {album_name} ({track_count} tracks)")
            
            if album['type'] == 'virtual':
                # Get custom track order if available
                import re
                album_id = f"album_{re.sub(r'[^a-zA-Z0-9]', '_', album_name)}"
                custom_order = None
                if custom_track_orders and album_id in custom_track_orders:
                    custom_order = custom_track_orders[album_id]
                    print(f"🎵 Using custom track order for {album_name}: {custom_order}")
                
                # Check if this is a custom album with custom title and selected files
                final_album_name = album_name
                final_files = album_folder.album_files
                
                if custom_album_titles and album_id in custom_album_titles:
                    final_album_name = custom_album_titles[album_id]
                    print(f"🎵 Using custom album title: {final_album_name}")
                
                if custom_album_files and album_id in custom_album_files:
                    # Filter files to only include selected ones
                    selected_filenames = custom_album_files[album_id]
                    file_map = {f.name: f for f in album_folder.album_files}
                    final_files = [file_map[fname] for fname in selected_filenames if fname in file_map]
                    print(f"🎵 Using custom file selection: {len(final_files)} of {len(album_folder.album_files)} files")
                
                # Use simplified merge process for grouped files
                merge_grouped_files(
                    final_files, 
                    final_album_name, 
                    output_folder, 
                    session_id, 
                    progress,
                    delete_originals=delete_originals,
                    custom_track_order=custom_order
                )
            else:
                # Real folder - use existing merge method with delete_originals flag
                merger.merge_album(album_folder, output_folder, delete_originals)

            processed += 1
            time.sleep(0.5)  # Small delay for UI feedback

        merge_progress[session_id] = {
            'status': 'complete',
            'progress': 100,
            'message': f'Successfully merged {processed} albums!'
        }

    except Exception as e:
        merge_progress[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

def merge_all_with_progress(session_id, output_folder=None, delete_originals=False):
    """Merge all MP3s in folder into single file with progress tracking"""
    global merge_progress, active_sessions
    from pathlib import Path

    try:
        merge_progress[session_id] = {'status': 'starting', 'progress': 0, 'message': 'Initializing merge all...'}

        # Get merger instance from session
        if session_id not in active_sessions:
            merge_progress[session_id] = {'status': 'error', 'progress': 0, 'message': 'Session not found'}
            return

        merger = active_sessions[session_id]

        # Get all MP3 files
        all_mp3s = list(merger.downloads_folder.glob("*.mp3"))

        if len(all_mp3s) < 2:
            merge_progress[session_id] = {'status': 'complete', 'progress': 100, 'message': 'Need at least 2 MP3 files to merge'}
            return

        merge_progress[session_id] = {'status': 'processing', 'progress': 10, 'message': f'Found {len(all_mp3s)} MP3 files to merge'}
        time.sleep(1)

        # Sort files by track number if available, otherwise by filename
        def get_sort_key(mp3_file):
            try:
                import eyed3
                audiofile = eyed3.load(str(mp3_file))
                if audiofile and audiofile.tag and audiofile.tag.track_num:
                    return audiofile.tag.track_num[0]
            except (OSError, AttributeError, TypeError) as e:
                logger.debug(f"Could not extract track number from {mp3_file}: {e}")
                pass
            return mp3_file.name.lower()

        all_mp3s.sort(key=get_sort_key)

        merge_progress[session_id] = {'status': 'processing', 'progress': 30, 'message': 'Sorting tracks by order...'}
        time.sleep(0.5)

        # Use folder name as album title
        folder_name = merger.downloads_folder.name
        output_filename = f"{folder_name}.mp3"

        merge_progress[session_id] = {'status': 'processing', 'progress': 50, 'message': f'Merging into {output_filename}...'}

        # Create temporary concat file
        temp_concat = merger.downloads_folder / "temp_concat.txt"
        try:
            with open(temp_concat, 'w', encoding='utf-8') as f:
                for mp3_file in all_mp3s:
                    safe_path = str(mp3_file).replace("'", "\\'")
                    f.write(f"file '{safe_path}'\n")

            merge_progress[session_id] = {'status': 'processing', 'progress': 70, 'message': 'Running ffmpeg merge...'}

            # Run ffmpeg to merge
            if output_folder:
                output_dir = Path(output_folder)
            else:
                output_dir = merger.downloads_folder
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
            output_path = output_dir / output_filename
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(temp_concat),
                '-c', 'copy', '-metadata', f'album={folder_name}',
                '-metadata', f'title={folder_name}', '-y', str(output_path)
            ]

            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Delete original files if requested
                if delete_originals:
                    merge_progress[session_id] = {'status': 'processing', 'progress': 90, 'message': 'Deleting original files...'}
                    for mp3_file in all_mp3s:
                        try:
                            mp3_file.unlink()
                        except Exception as e:
                            print(f"Warning: Could not delete {mp3_file}: {e}")

                merge_progress[session_id] = {
                    'status': 'complete',
                    'progress': 100,
                    'message': f'Successfully merged {len(all_mp3s)} tracks into {output_filename}!'
                }
            else:
                merge_progress[session_id] = {
                    'status': 'error',
                    'progress': 0,
                    'message': f'ffmpeg error: {result.stderr}'
                }

        finally:
            # Clean up temp file
            if temp_concat.exists():
                temp_concat.unlink()

    except Exception as e:
        merge_progress[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

# HTML Template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>track gluer togetherer</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Roboto Mono', monospace;
            margin: 0;
            padding: 20px;
            background: #ffffff;
            color: #000000;
            font-size: 11px;
            line-height: 1.6;
            letter-spacing: 0.5px;
            font-feature-settings: "kern" 1;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            background: #ffffff;
            border: 3px solid #000000;
            overflow: hidden;
        }

        .header {
            background: #000000;
            color: #ffffff;
            padding: 15px;
            text-align: center;
            border-bottom: 3px solid #000000;
        }

        .header h1 {
            margin: 0;
            font-size: 11px;
            font-weight: 400;
            letter-spacing: 2px;
            line-height: 1.8;
            text-transform: uppercase;
        }

        .main-content { padding: 20px; }

        .step {
            margin-bottom: 20px;
            padding: 15px;
            border: 3px solid #000000;
            background: #ffffff;
        }

        .step.active { 
            border: 3px solid #000000;
            background: #ffffff;
        }

        .step h2 {
            color: #000000;
            margin-bottom: 18px;
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            line-height: 1.7;
        }
        
        /* File Explorer Modal Styles */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            overflow: auto;
        }
        
        .modal-content {
            background-color: #fefefe;
            margin: 10% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .modal-header h3 {
            margin: 0;
            color: #333;
        }
        
        .close {
            color: #aaa;
            font-size: 11px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: #000;
        }
        
        .file-explorer-nav {
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        
        .file-explorer-list {
            border: 1px solid #ddd;
            min-height: 200px;
            max-height: 50vh;
            overflow-y: auto;
            padding: 10px;
            background-color: #fff;
            border-radius: 4px;
        }
        
        .file-item {
            padding: 8px 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            border-radius: 4px;
            margin-bottom: 4px;
        }
        
        .file-item:hover {
            background-color: #f5f5f5;
        }
        
        .file-icon {
            margin-right: 10px;
            font-size: 11px;
        }
        
        .file-name {
            flex-grow: 1;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .empty-folder {
            text-align: center;
            padding: 20px;
            color: #999;
            font-style: italic;
        }
        
        .error {
            color: #d32f2f;
            padding: 10px;
            background-color: #ffebee;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .modal-footer {
            margin-top: 15px;
            text-align: right;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }

        .folder-input {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .folder-input input {
            flex: 1;
            padding: 6px 10px;
            border: 3px solid #000000;
            font-family: 'Roboto Mono', monospace;
            font-size: 11px;
            background: #ffffff;
            color: #000000;
            height: 32px;
            box-sizing: border-box;
        }

        .btn {
            background: #000000;
            color: #ffffff;
            border: 3px solid #000000;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 11px;
            font-family: 'Roboto Mono', monospace;
            text-transform: uppercase;
            min-width: 80px;
            height: 32px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            letter-spacing: 1px;
            line-height: 1.2;
        }

        .btn:hover { background: #ffffff; color: #000000; }
        .btn:disabled { background: #ffffff; color: #000000; cursor: not-allowed; }
        
        .btn-small {
            padding: 6px 10px;
            font-size: 11px;
            margin-left: 10px;
            min-width: 60px;
            height: 28px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .album-group {
            border: 3px solid #000000;
            margin-bottom: 10px;
            padding: 12px;
            background: #ffffff;
            font-size: 11px;
        }

        .album-header {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }

        .album-title { font-weight: 400; color: #000000; }

        .album-type {
            background: #000000;
            color: #ffffff;
            padding: 6px 10px;
            font-size: 11px;
            border: 3px solid #000000;
            text-transform: uppercase;
            height: 28px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 50px;
        }

        .file-list {
            list-style: none;
            margin-left: 20px;
        }

        .file-list li {
            padding: 5px 0;
            color: #000000;
            border-bottom: 1px solid #000000;
        }

        .file-list li:last-child { border-bottom: none; }
        
        .mb-info {
            background: #f0f0f0;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 10px;
            font-size: 11px;
        }
        
        .mb-info h4 {
            margin: 0 0 5px 0;
            color: #000;
        }
        
        .mb-info .mb-match {
            color: #006600;
        }
        
        .mb-info .mb-mismatch {
            color: #cc0000;
        }
        
        .mb-info .mb-stats {
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        /* Drag and drop styles */
        .file-list {
            min-height: 20px;
        }
        
        .file-list li {
            cursor: move;
            user-select: none;
            position: relative;
            transition: background-color 0.2s;
            padding: 6px 0;
            font-size: 11px;
            line-height: 1.5;
            letter-spacing: 0.3px;
        }
        
        .file-list li:hover {
            background-color: #000000;
            color: #ffffff;
        }
        
        .file-list li.dragging {
            opacity: 0.5;
            background-color: #000000;
            color: #ffffff;
        }
        
        .file-list li.drag-over {
            border-top: 3px solid #000000;
        }
        
        .drag-handle {
            display: inline-block;
            margin-right: 6px;
            color: #000000;
            font-size: 11px;
            width: 12px;
        }
        
        .track-number {
            display: inline-block;
            min-width: 20px;
            font-weight: 500;
            color: #000000;
            font-size: 11px;
        }
        
        /* Custom album styles */
        .custom-album {
            border: 3px dashed #000000;
            background-color: #ffffff;
        }
        
        .custom-title-input {
            font-family: 'Roboto Mono', monospace;
            font-size: 11px;
            font-weight: 500;
            border: 3px solid #000000;
            padding: 6px 10px;
            margin: 0 10px;
            min-width: 200px;
            height: 28px;
            background-color: #ffffff;
            color: #000000;
            box-sizing: border-box;
        }
        
        .custom-title-input:focus {
            outline: none;
            border-color: #000000;
            background-color: #000000;
            color: #ffffff;
        }
        
        .album-type.custom {
            background-color: #000000;
            color: #ffffff;
        }
        
        .custom-track {
            padding-left: 5px;
        }
        
        .custom-track input[type="checkbox"] {
            margin-right: 8px;
        }

        .progress-bar {
            background: #ffffff;
            border: 3px solid #000000;
            overflow: hidden;
            height: 20px;
            margin: 20px 0;
        }

        .progress-fill {
            background: #000000;
            height: 100%;
            width: 0%;
            transition: width 0.3s ease;
        }

        .hidden { display: none; }

        .alert {
            border: 3px solid #000000;
            background: #ffffff;
            color: #000000;
            padding: 10px;
            margin: 10px 0;
            text-transform: uppercase;
        }

        .alert.success {
            background: #000000;
            color: #ffffff;
        }

        .alert.error {
            background: #ffffff;
            color: #000000;
            border: 3px solid #000000;
        }

        /* File Explorer Styles */
        .file-explorer {
            border: 3px solid #000000;
            margin: 20px 0;
            background: #ffffff;
        }

        .current-path {
            background: #000000;
            color: #ffffff;
            padding: 10px;
            font-family: 'Roboto Mono', monospace;
        }

        .folder-contents {
            max-height: 300px;
            overflow-y: auto;
            border: 3px solid #000000;
        }

        .folder-item {
            display: flex;
            align-items: center;
            padding: 8px;
            cursor: pointer;
            border-bottom: 1px solid #000000;
        }

        .folder-item:hover {
            background: #000000;
            color: #ffffff;
        }

        .folder-item:last-child {
            border-bottom: none;
        }

        .folder-icon {
            margin-right: 10px;
            font-size: 11px;
        }

        .folder-name {
            flex: 1;
            font-weight: 500;
        }

        .mp3-count {
            color: #ffffff;
            font-size: 11px;
            background: #000000;
            padding: 6px 10px;
            border: 3px solid #000000;
            height: 28px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 30px;
        }

        .album-checkbox {
            margin-right: 10px;
            width: 16px;
            height: 16px;
            border: 2px solid #000000;
            background: #ffffff;
        }
        
        .album-checkbox:checked {
            background: #000000;
        }

        .album-header {
            display: flex;
            align-items: center;
        }

        /* Brutalist Loading Animation */
        .loader {
            display: block;
            margin: 30px auto;
            position: relative;
        }

        .brutalist-loader {
            width: 200px;
            height: 120px;
            margin: 0 auto;
            position: relative;
            border: 4px solid #000;
            background: #fff;
            overflow: hidden;
        }

        .brutalist-bars {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 100%;
            display: flex;
            align-items: flex-end;
            justify-content: space-around;
            padding: 8px;
        }

        .bar {
            width: 12px;
            background: #000;
            animation: brutalist-pulse 2.4s infinite;
            transform-origin: bottom;
        }

        .bar:nth-child(1) { animation-delay: 0s; }
        .bar:nth-child(2) { animation-delay: 0.2s; }
        .bar:nth-child(3) { animation-delay: 0.4s; }
        .bar:nth-child(4) { animation-delay: 0.6s; }
        .bar:nth-child(5) { animation-delay: 0.8s; }
        .bar:nth-child(6) { animation-delay: 1.0s; }
        .bar:nth-child(7) { animation-delay: 1.2s; }
        .bar:nth-child(8) { animation-delay: 1.4s; }
        .bar:nth-child(9) { animation-delay: 1.6s; }
        .bar:nth-child(10) { animation-delay: 1.8s; }

        .brutalist-glitch {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: #000;
            animation: glitch-slide 3s infinite linear;
        }

        .brutalist-text {
            position: absolute;
            top: -40px;
            left: 0;
            right: 0;
            text-align: center;
            font-family: 'Roboto Mono', monospace;
            font-weight: 500;
            font-size: 14px;
            letter-spacing: 2px;
            animation: text-flicker 1.8s infinite;
        }

        @keyframes brutalist-pulse {
            0%, 100% { height: 20px; }
            50% { height: 80px; }
        }

        @keyframes glitch-slide {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(300%); }
        }

        @keyframes text-flicker {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        /* MusicBrainz specific loader */
        .musicbrainz-loader {
            border: 6px solid #000;
            background: #333;
        }

        .musicbrainz-loader .bar {
            background: #fff;
        }

        .musicbrainz-loader .brutalist-glitch {
            background: #fff;
        }

        .musicbrainz-loader .brutalist-text {
            color: #fff;
        }

    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>track gluer togetherer</h1>
        </div>

        <div class="main-content">
            <div class="step active" id="step1">
                <h2>select folder</h2>
                <div class="folder-input">
                    <input type="text" id="folderPath" placeholder="folder path" value=".">
                    <button class="btn" onclick="browseFolder()">browse</button>
                </div>

                <div class="scan-buttons" style="margin-top: 15px; display: flex; gap: 10px; justify-content: center;">
                    <button class="btn" onclick="scanFolder()">regular scan</button>
                    <button class="btn" onclick="musicbrainzResort()" style="background: #333; color: white;">musicbrainz scan</button>
                    <button class="btn" onclick="glueEmAll()" style="background: #000; color: white; border: 2px solid #000;">glue em all</button>
                </div>

            </div>

            <div class="step" id="step2">
                <h2>preview</h2>
                <div id="previewContent"></div>

                <div class="hidden" id="outputSection">
                    <h3>output folder</h3>
                    <div class="folder-input">
                        <input type="text" id="outputPath" placeholder="output folder path" value=".">
                        <button class="btn" onclick="browseOutputFolder()">browse</button>
                    </div>

                    <div style="margin-top: 15px;">
                        <label>
                            <input type="checkbox" id="deleteOriginals" class="album-checkbox">
                            delete original files after merge
                        </label>
                    </div>
                </div>


                <button class="btn hidden" onclick="startMerge('albums')" id="startMergeBtn">glue em</button>
            </div>

            <div class="step" id="step3">
                <h2>progress</h2>
                <div class="hidden" id="progressSection">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <p id="progressText"></p>
                </div>
            </div>
        </div>

        <div style="padding: 20px; border-top: 1px solid #000000; text-align: center;">
            <button class="btn" onclick="shutdownApp()">quit</button>
        </div>
    </div>

    <!-- File Explorer Modal -->
    <div id="fileExplorerModal" class="modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.7); z-index: 1000; overflow: auto;">
        <div class="modal-content" style="background-color: #fefefe; margin: 10% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 800px; max-height: 80vh; overflow-y: auto;">
            <div class="modal-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee;">
                <h3 style="margin: 0;">Select Folder</h3>
                <span class="close" onclick="closeFileExplorer()" style="color: #aaa; font-size: 11px; font-weight: bold; cursor: pointer;">&times;</span>
            </div>
            <div class="modal-body">
                <div class="file-explorer-nav" style="margin-bottom: 15px; display: flex; align-items: center;">
                    <button id="goUpButton" class="btn" onclick="navigateUp()" style="margin-right: 10px;" disabled>↑ Up</button>
                    <span id="currentPath" style="font-family: monospace; word-break: break-all;">/</span>
                </div>
                <div class="file-explorer-list" id="fileExplorerList" style="border: 1px solid #ddd; min-height: 200px; max-height: 50vh; overflow-y: auto; padding: 10px;">
                    <div class="loading">Loading...</div>
                </div>
            </div>
            <div class="modal-footer" style="margin-top: 15px; text-align: right; padding-top: 15px; border-top: 1px solid #eee;">
                <button class="btn" onclick="closeFileExplorer()" style="margin-right: 10px;">Cancel</button>
                <button class="btn primary" onclick="selectCurrentFolder()">Select This Folder</button>
            </div>
        </div>
    </div>

    <script>
        let currentSession = null;
        let browseMode = 'input'; // 'input' or 'output'

        async function scanFolder() {
            const folderPath = document.getElementById('folderPath').value;
            console.log('Scanning folder:', folderPath);

            // Show loading animation
            document.getElementById('step1').classList.remove('active');
            document.getElementById('step2').classList.add('active');
            document.getElementById('previewContent').innerHTML = `
                <div style="text-align: center; padding: 40px;">
                    <div class="loader">
                        <div class="brutalist-text">SCANNING FOLDER</div>
                        <div class="brutalist-loader">
                            <div class="brutalist-glitch"></div>
                            <div class="brutalist-bars">
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            try {
                const response = await fetch('/api/scan_folder', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder_path: folderPath })
                });

                const data = await response.json();

                if (response.ok) {
                    currentSession = data.session_id;
                    displayPreview(data.preview);
                    document.getElementById('step1').classList.remove('active');
                    document.getElementById('step2').classList.add('active');
                } else {
                    showAlert(data.error, 'error');
                }
            } catch (error) {
                showAlert('Failed to scan folder: ' + error.message, 'error');
            }
        }

        async function musicbrainzResort() {
            const folderPath = document.getElementById('folderPath').value;
            console.log('MusicBrainz resort for folder:', folderPath);

            // Show loading animation
            document.getElementById('step1').classList.remove('active');
            document.getElementById('step2').classList.add('active');
            document.getElementById('previewContent').innerHTML = `
                <div style="text-align: center; padding: 40px;">
                    <div class="loader">
                        <div class="brutalist-text">MUSICBRAINZ GROUPING</div>
                        <div class="brutalist-loader musicbrainz-loader">
                            <div class="brutalist-glitch"></div>
                            <div class="brutalist-bars">
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            try {
                const response = await fetch('/api/musicbrainz_resort', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder_path: folderPath })
                });

                const data = await response.json();

                if (response.ok) {
                    console.log('=== MUSICBRAINZ RESPONSE DEBUG ===');
                    console.log('Full response:', data);
                    console.log('Preview object:', data.preview);
                    console.log('Loose files:', data.preview?.loose_files);
                    console.log('Groupings:', data.preview?.loose_files?.groupings);
                    displayPreview(data.preview);
                    document.getElementById('step2').classList.add('active');
                } else {
                    showAlert(data.error, 'error');
                }
            } catch (error) {
                showAlert('Failed to resort with MusicBrainz: ' + error.message, 'error');
            }
        }

        async function glueEmAll() {
            const folderPath = document.getElementById('folderPath').value;
            console.log('Glue Em All for folder:', folderPath);

            // Show loading animation
            document.getElementById('step1').classList.remove('active');
            document.getElementById('step2').classList.add('active');
            document.getElementById('previewContent').innerHTML = `
                <div style="text-align: center; padding: 40px;">
                    <div class="loader">
                        <div class="brutalist-text">LOADING ALL FILES</div>
                        <div class="brutalist-loader">
                            <div class="brutalist-glitch"></div>
                            <div class="brutalist-bars">
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                                <div class="bar"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            try {
                const response = await fetch('/api/glue_em_all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder_path: folderPath })
                });

                const data = await response.json();

                if (response.ok) {
                    console.log('=== GLUE EM ALL RESPONSE ===');
                    console.log('Full response:', data);
                    displayPreview(data.preview);
                    document.getElementById('step2').classList.add('active');
                } else {
                    showAlert(data.error, 'error');
                }
            } catch (error) {
                showAlert('Failed to load all files: ' + error.message, 'error');
            }
        }

        function displayPreview(preview) {
            console.log('=== PREVIEW DEBUG ===');
            console.log('Total albums:', preview.total_albums);
            console.log('Loose files count:', preview.loose_files?.count);
            console.log('Groupings found:', Object.keys(preview.loose_files?.groupings || {}));
            
            // Log each grouping
            if (preview.loose_files?.groupings) {
                Object.entries(preview.loose_files.groupings).forEach(([albumName, group]) => {
                    console.log(`- ${albumName}: ${group.count} files (${group.type})`);
                });
            }
            
            const content = document.getElementById('previewContent');
            let html = '';

            if (preview.total_albums === 0) {
                html += `<div class="alert">
                    found ${preview.total_albums} albums - try "merge all" for dj mixes or playlists
                </div>`;
            } else {
                html += `<div class="alert success">
                    found ${preview.total_albums} albums
                </div>`;
            }

            if (preview.loose_files && preview.loose_files.groupings && Object.keys(preview.loose_files.groupings || {}).length > 0) {
                html += '<h3>auto-detected:</h3>';
                for (const [albumName, group] of Object.entries(preview.loose_files.groupings)) {
                    const albumId = albumName.replace(/[^a-zA-Z0-9]/g, '_');
                    const isIndividual = albumName === 'Individual Tracks';

                    // Special handling for Individual Tracks - create custom album
                    if (isIndividual) {
                        // Fix the albumId to match backend generation
                        const fixedAlbumId = albumName.replace(/[^a-zA-Z0-9]/g, '_');
                        html += `
                            <div class="album-group custom-album">
                                <div class="album-header">
                                    <input type="checkbox" id="album_${fixedAlbumId}" checked class="album-checkbox">
                                    <input type="text" id="custom-title-${fixedAlbumId}" class="custom-title-input" placeholder="Enter custom album title..." value="Custom Album">
                                    <span class="album-type custom">custom</span>
                                    <button class="btn btn-small" onclick="selectAllTracks('${fixedAlbumId}')" title="Select all tracks">All</button>
                                    <button class="btn btn-small" onclick="selectNoTracks('${fixedAlbumId}')" title="Deselect all tracks">None</button>
                                </div>
                                <p><strong>${group.count} loose tracks - select which to include:</strong></p>
                                <ul class="file-list" id="filelist-${fixedAlbumId}">
                                    ${(group.files || []).map((file, index) => `
                                        <li draggable="true" data-filename="${file}" data-album-id="${fixedAlbumId}" class="custom-track">
                                            <input type="checkbox" id="track_${fixedAlbumId}_${index}" class="track-checkbox" data-album="${fixedAlbumId}" data-file="${file}" checked>
                                            <span class="drag-handle">≡</span>
                                            <span class="track-number">${index + 1}.</span>
                                            <span class="track-name">${file}</span>
                                        </li>
                                    `).join('')}
                                </ul>
                                <div id="mb-info-${fixedAlbumId}" class="mb-info" style="display: none;"></div>
                            </div>
                        `;
                    } else {
                        // Regular album display
                        html += `
                            <div class="album-group">
                                <div class="album-header">
                                    <input type="checkbox" id="album_${albumId}" checked class="album-checkbox">
                                    <span class="album-title">${albumName}</span>
                                    <span class="album-type ${group.type || 'album'}">${group.type || 'album'}</span>
                                    <!-- Enhance button temporarily disabled to fix JS error -->
                                    <!-- <button class="btn btn-small" onclick="enhanceWithMusicBrainz('${albumName}', ${JSON.stringify(group.files || []).replace(/"/g, '&quot;')})" title="Enhance with MusicBrainz data">MusicBrainz</button> -->
                                </div>
                                <p><strong>${group.count || 0} tracks</strong></p>
                                <ul class="file-list" id="filelist-${albumId}">
                                    ${(group.files || []).map((file, index) => `
                                        <li draggable="true" data-filename="${file}" data-album-id="${albumId}">
                                            <span class="drag-handle">≡</span>
                                            <span class="track-number">${index + 1}.</span>
                                            <span class="track-name">${file}</span>
                                        </li>
                                    `).join('')}
                                </ul>
                                <div id="mb-info-${albumId}" class="mb-info" style="display: none;"></div>
                            </div>
                        `;
                    }
                }
            }

            if (preview.existing_folders.length > 0) {
                html += '<h3>folders:</h3>';
                preview.existing_folders.forEach(folder => {
                    const folderId = folder.name.replace(/[^a-zA-Z0-9]/g, '_');
                    html += `
                        <div class="album-group">
                            <div class="album-header">
                                <input type="checkbox" id="folder_${folderId}" checked class="album-checkbox">
                                <span class="album-title">${folder.name}</span>
                                <span class="album-type ${folder.type || 'album'}">${folder.type || 'album'}</span>
                                <!-- Enhance button temporarily disabled to fix JS error -->
                                <!-- <button class="btn btn-small" onclick="enhanceWithMusicBrainz('${folder.name}', ${JSON.stringify(folder.files || []).replace(/"/g, '&quot;')})" title="Enhance with MusicBrainz data">MusicBrainz</button> -->
                            </div>
                            <p><strong>${folder.count || 0} tracks</strong></p>
                            <ul class="file-list" id="filelist-${folderId}">
                                ${folder.files.map((file, index) => `
                                    <li draggable="true" data-filename="${file}" data-album-id="${folderId}">
                                        <span class="drag-handle">≡</span>
                                        <span class="track-number">${index + 1}.</span>
                                        <span class="track-name">${file}</span>
                                    </li>
                                `).join('')}
                            </ul>
                            <div id="mb-info-${folderId}" class="mb-info" style="display: none;"></div>
                        </div>
                    `;
                });
            }

            content.innerHTML = html;

            // Initialize drag and drop for all file lists
            initializeDragAndDrop();

            const outputSection = document.getElementById('outputSection');
            const startMergeBtn = document.getElementById('startMergeBtn');

            if (outputSection) {
                outputSection.classList.remove('hidden');
            }
            if (startMergeBtn) {
                startMergeBtn.classList.remove('hidden');
            }
        }

        async function startMerge(mergeType = 'albums') {
            if (!currentSession) {
                showAlert('No active session. Please scan a folder first.', 'error');
                return;
            }

            document.getElementById('step2').classList.remove('active');
            document.getElementById('step3').classList.add('active');
            document.getElementById('progressSection').classList.remove('hidden');

            const outputPath = document.getElementById('outputPath').value || '.';
            const deleteOriginals = document.getElementById('deleteOriginals').checked;

            // Get selected albums and their custom track orders
            const selectedAlbums = [];
            const customTrackOrders = {};
            const customAlbumTitles = {};
            const customAlbumFiles = {};
            
            document.querySelectorAll('.album-checkbox:checked').forEach(checkbox => {
                if (checkbox.id.startsWith('album_') || checkbox.id.startsWith('folder_')) {
                    selectedAlbums.push(checkbox.id);
                    console.log(`Selected checkbox ID: ${checkbox.id}`);
                    
                    // Get custom track order if it exists
                    const albumId = checkbox.id.replace(/^(album_|folder_)/, '');
                    const trackOrder = getTrackOrder(albumId);
                    if (trackOrder) {
                        customTrackOrders[checkbox.id] = trackOrder;
                        console.log(`Track order for ${checkbox.id}:`, trackOrder);
                    }
                    
                    // Check if this is a custom album
                    const customTitleInput = document.getElementById(`custom-title-${albumId}`);
                    if (customTitleInput) {
                        const customData = getCustomAlbumData(albumId);
                        customAlbumTitles[checkbox.id] = customData.title;
                        customAlbumFiles[checkbox.id] = customData.selectedFiles;
                        console.log(`Custom album: ${customData.title} with ${customData.selectedFiles.length} tracks`);
                        console.log(`Custom album ID: ${checkbox.id}, Album ID: ${albumId}`);
                    }
                }
            });
            
            console.log('Final selected albums:', selectedAlbums);
            console.log('Final custom titles:', customAlbumTitles);
            console.log('Final custom files:', customAlbumFiles);
            
            // Debug custom files in detail
            Object.entries(customAlbumFiles).forEach(([key, files]) => {
                console.log(`Custom files for ${key}:`, files);
            });

            try {
                const response = await fetch('/api/start_merge', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: currentSession,
                        merge_type: mergeType,
                        output_path: outputPath,
                        delete_originals: deleteOriginals,
                        selected_albums: selectedAlbums,
                        custom_track_orders: customTrackOrders,
                        custom_album_titles: customAlbumTitles,
                        custom_album_files: customAlbumFiles
                    })
                });

                if (response.ok) {
                    monitorProgress();
                } else {
                    const data = await response.json();
                    showAlert(data.error, 'error');
                }
            } catch (error) {
                showAlert('Failed to start merge: ' + error.message, 'error');
            }
        }

        function monitorProgress() {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/progress/${currentSession}`);
                    const data = await response.json();

                    if (response.ok) {
                        document.getElementById('progressFill').style.width = data.progress + '%';
                        document.getElementById('progressText').textContent = data.message;

                        if (data.status === 'complete') {
                            clearInterval(interval);
                            showAlert('Merge completed successfully!', 'success');
                        } else if (data.status === 'error') {
                            clearInterval(interval);
                            showAlert(data.message, 'error');
                        }
                    }
                } catch (error) {
                    console.error('Progress check failed:', error);
                }
            }, 1000);
        }

        function showAlert(message, type) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert ${type}`;
            alertDiv.textContent = message;

            const mainContent = document.querySelector('.main-content');
            mainContent.insertBefore(alertDiv, mainContent.firstChild);

            setTimeout(() => alertDiv.remove(), 5000);
        }

        // Load saved paths on page load
        window.addEventListener('load', function() {
            const savedInputPath = localStorage.getItem('lastInputFolder');
            const savedOutputPath = localStorage.getItem('lastOutputFolder');

            if (savedInputPath) {
                document.getElementById('folderPath').value = savedInputPath;
            }
            if (savedOutputPath) {
                document.getElementById('outputPath').value = savedOutputPath;
            }
        });

        // File Explorer Functions
        let currentFolderPath = '';
        
        async function browseFolder() {
            browseMode = 'input';
            currentFolderPath = document.getElementById('folderPath').value || '.';
            await openFileExplorer(currentFolderPath);
        }

        async function browseOutputFolder() {
            browseMode = 'output';
            currentFolderPath = document.getElementById('outputPath').value || '.';
            await openFileExplorer(currentFolderPath);
        }
        
        async function openFileExplorer(path) {
            const modal = document.getElementById('fileExplorerModal');
            const fileList = document.getElementById('fileExplorerList');
            const currentPathElement = document.getElementById('currentPath');
            const goUpButton = document.getElementById('goUpButton');
            
            // Show loading
            fileList.innerHTML = '<div class="loading">Loading...</div>';
            modal.style.display = 'block';
            
            try {
                console.log('Making request to /api/browse_folder with path:', path);
                const response = await fetch('/api/browse_folder', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({ folder_path: path })
                });
                
                console.log('Response status:', response.status);
                
                if (!response.ok) {
                    console.error('Error response:', response);
                    let errorText;
                    try {
                        const errorData = await response.text();
                        console.error('Error response text:', errorText);
                        try {
                            const errorJson = JSON.parse(errorData);
                            errorText = errorJson.error || 'Failed to load directory';
                        } catch (e) {
                            errorText = errorData || 'Failed to parse error response';
                        }
                    } catch (e) {
                        errorText = 'Failed to read error response';
                    }
                    throw new Error(`Server error (${response.status}): ${errorText}`);
                }
                
                let data;
                try {
                    data = await response.json();
                    currentFolderPath = data.current_path;
                    currentPathElement.textContent = currentFolderPath;
                } catch (e) {
                    console.error('Error parsing JSON response:', e);
                    throw new Error('Failed to parse server response');
                }
                
                // Update UI
                goUpButton.disabled = !data.parent_path || data.parent_path === data.current_path;
                
                // Clear and populate file list
                fileList.innerHTML = '';
                
                // Add parent directory entry if available
                if (data.parent_path && data.parent_path !== data.current_path) {
                    const parentItem = document.createElement('div');
                    parentItem.className = 'file-item folder';
                    parentItem.innerHTML = `
                        <span class="file-icon">📁</span>
                        <span class="file-name">.. (Parent Directory)</span>
                    `;
                    parentItem.onclick = () => openFileExplorer(data.parent_path);
                    fileList.appendChild(parentItem);
                }
                
                // Add directories and files
                data.items.forEach(item => {
                    const itemElement = document.createElement('div');
                    itemElement.className = 'file-item' + (item.is_dir ? ' folder' : '');
                    
                    const icon = item.is_dir ? '📁' : '📄';
                    const mp3Count = item.is_dir && item.mp3_count !== undefined ? ` (${item.mp3_count} MP3s)` : '';
                    
                    itemElement.innerHTML = `
                        <span class="file-icon">${icon}</span>
                        <span class="file-name">${item.name}${mp3Count}</span>
                    `;
                    
                    if (item.is_dir) {
                        itemElement.onclick = (e) => {
                            e.stopPropagation();
                            openFileExplorer(item.path);
                        };
                    }
                    
                    fileList.appendChild(itemElement);
                });
                
                if (data.items.length === 0) {
                    fileList.innerHTML = '<div class="empty-folder">Empty folder</div>';
                }
                
            } catch (error) {
                fileList.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                console.error('File explorer error:', error);
            }
        }
        
        function closeFileExplorer() {
            const modal = document.getElementById('fileExplorerModal');
            modal.style.display = 'none';
        }
        
        function selectCurrentFolder() {
            if (browseMode === 'input') {
                document.getElementById('folderPath').value = currentFolderPath;
            } else {
                document.getElementById('outputPath').value = currentFolderPath;
            }
            closeFileExplorer();
        }
        
        function navigateUp() {
            const currentPath = document.getElementById('currentPath').textContent;
            const parentPath = currentPath.split('/').slice(0, -1).join('/') || '/';
            if (parentPath && parentPath !== currentPath) {
                openFileExplorer(parentPath);
            }
        }
        
        // Close modal when clicking outside of it
        window.onclick = function(event) {
            const modal = document.getElementById('fileExplorerModal');
            if (event.target === modal) {
                closeFileExplorer();
            }
        };

        // Drag and Drop functionality
        function initializeDragAndDrop() {
            const fileListItems = document.querySelectorAll('.file-list li[draggable="true"]');
            
            fileListItems.forEach(item => {
                item.addEventListener('dragstart', handleDragStart);
                item.addEventListener('dragover', handleDragOver);
                item.addEventListener('dragenter', handleDragEnter);
                item.addEventListener('dragleave', handleDragLeave);
                item.addEventListener('drop', handleDrop);
                item.addEventListener('dragend', handleDragEnd);
            });
        }

        let draggedElement = null;

        function handleDragStart(e) {
            draggedElement = this;
            this.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/html', this.outerHTML);
        }

        function handleDragOver(e) {
            if (e.preventDefault) {
                e.preventDefault();
            }
            e.dataTransfer.dropEffect = 'move';
            return false;
        }

        function handleDragEnter(e) {
            if (this !== draggedElement && this.dataset.albumId === draggedElement.dataset.albumId) {
                this.classList.add('drag-over');
            }
        }

        function handleDragLeave(e) {
            this.classList.remove('drag-over');
        }

        function handleDrop(e) {
            if (e.stopPropagation) {
                e.stopPropagation();
            }

            if (draggedElement !== this && this.dataset.albumId === draggedElement.dataset.albumId) {
                // Determine if we should insert before or after
                const rect = this.getBoundingClientRect();
                const midpoint = rect.top + rect.height / 2;
                const insertAfter = e.clientY > midpoint;

                if (insertAfter) {
                    this.parentNode.insertBefore(draggedElement, this.nextSibling);
                } else {
                    this.parentNode.insertBefore(draggedElement, this);
                }

                // Update track numbers
                updateTrackNumbers(this.dataset.albumId);
                
                // Save the new order
                saveTrackOrder(this.dataset.albumId);
            }

            this.classList.remove('drag-over');
            return false;
        }

        function handleDragEnd(e) {
            this.classList.remove('dragging');
            
            // Remove drag-over class from all items
            document.querySelectorAll('.file-list li').forEach(item => {
                item.classList.remove('drag-over');
            });
            
            draggedElement = null;
        }

        function updateTrackNumbers(albumId) {
            const fileList = document.getElementById(`filelist-${albumId}`);
            if (fileList) {
                const items = fileList.querySelectorAll('li');
                items.forEach((item, index) => {
                    const trackNumber = item.querySelector('.track-number');
                    if (trackNumber) {
                        trackNumber.textContent = `${index + 1}.`;
                    }
                });
            }
        }

        // Store track orders for each album
        const trackOrders = {};

        function saveTrackOrder(albumId) {
            const fileList = document.getElementById(`filelist-${albumId}`);
            if (fileList) {
                const items = fileList.querySelectorAll('li');
                const order = Array.from(items).map(item => item.dataset.filename);
                trackOrders[albumId] = order;
                console.log(`Saved track order for ${albumId}:`, order);
            }
        }

        function getTrackOrder(albumId) {
            return trackOrders[albumId] || null;
        }

        // Functions for custom album track selection
        function selectAllTracks(albumId) {
            const checkboxes = document.querySelectorAll(`#filelist-${albumId} .track-checkbox`);
            checkboxes.forEach(checkbox => {
                checkbox.checked = true;
            });
        }

        function selectNoTracks(albumId) {
            const checkboxes = document.querySelectorAll(`#filelist-${albumId} .track-checkbox`);
            checkboxes.forEach(checkbox => {
                checkbox.checked = false;
            });
        }

        function getCustomAlbumData(albumId) {
            const titleInput = document.getElementById(`custom-title-${albumId}`);
            const checkboxes = document.querySelectorAll(`#filelist-${albumId} .track-checkbox:checked`);
            
            return {
                title: titleInput ? titleInput.value.trim() || 'Custom Album' : 'Custom Album',
                selectedFiles: Array.from(checkboxes).map(cb => cb.dataset.file)
            };
        }

        async function enhanceWithMusicBrainz(albumName, albumFiles) {
            try {
                console.log('Enhancing album with MusicBrainz:', albumName, albumFiles);
                
                const albumId = albumName.replace(/[^a-zA-Z0-9]/g, '_');
                const infoDiv = document.getElementById(`mb-info-${albumId}`);
                
                if (!infoDiv) {
                    console.error('Info div not found for album:', albumId);
                    return;
                }
                
                // Show loading
                infoDiv.style.display = 'block';
                infoDiv.innerHTML = '<div class="mb-stats">🔍 Looking up MusicBrainz data...</div>';
                
                const response = await fetch('/api/enhance_with_musicbrainz', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        folder_path: document.getElementById('folderPath').value,
                        album_files: albumFiles,
                        album_name: albumName
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Display MusicBrainz information
                    let html = `
                        <h4>📀 MusicBrainz: ${result.release_info.artist} - ${result.release_info.title}</h4>
                        <div class="mb-stats">
                            Match: ${result.matched_count}/${result.total_files} tracks (${result.match_percentage.toFixed(1)}%)
                        </div>
                    `;
                    
                    if (result.matched_tracks.length > 0) {
                        html += '<div><strong>Matched tracks:</strong></div>';
                        result.matched_tracks.forEach(track => {
                            html += `<div class="mb-match">✓ ${track.track_number}. ${track.mb_title}</div>`;
                        });
                    }
                    
                    if (result.unmatched_files.length > 0) {
                        html += '<div style="margin-top: 10px;"><strong>Unmatched files:</strong></div>';
                        result.unmatched_files.forEach(file => {
                            html += `<div class="mb-mismatch">✗ ${file.local_title}</div>`;
                        });
                    }
                    
                    if (result.release_info.date) {
                        html += `<div style="margin-top: 10px;"><strong>Release Date:</strong> ${result.release_info.date}</div>`;
                    }
                    
                    infoDiv.innerHTML = html;
                } else {
                    infoDiv.innerHTML = `<div class="mb-mismatch">❌ ${result.error}</div>`;
                }
                
            } catch (error) {
                console.error('MusicBrainz enhancement error:', error);
                const albumId = albumName.replace(/[^a-zA-Z0-9]/g, '_');
                const infoDiv = document.getElementById(`mb-info-${albumId}`);
                if (infoDiv) {
                    infoDiv.style.display = 'block';
                    infoDiv.innerHTML = `<div class="mb-mismatch">❌ Error: ${error.message}</div>`;
                }
            }
        }

        async function shutdownApp() {
            try {
                fetch('/api/shutdown', { method: 'POST' });
                document.body.innerHTML = `
                    <div style="
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        background: #ffffff;
                        color: #000000;
                        font-family: 'Roboto', sans-serif;
                        text-align: center;
                    ">
                        <div>
                            <h1>track gluer togetherer</h1>
                            <p>closed</p>
                        </div>
                    </div>
                `;
            } catch (error) {
                document.body.innerHTML = `
                    <div style="
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        background: #ffffff;
                        color: #000000;
                        font-family: 'Roboto', sans-serif;
                        text-align: center;
                    ">
                        <div>
                            <h1>track gluer togetherer</h1>
                            <p>closed</p>
                        </div>
                    </div>
                `;
            }
        }
    </script>
</body>
</html>'''

@app.route('/')
def index():
    """Main interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/scan_folder', methods=['POST'])
@validate_request(require_json=True, rate_limit=True)
def scan_folder():
    """Scan a folder and return preview with traditional grouping (MusicBrainz enhancement available per album)"""
    sys.stdout.flush()  # Force output
    print("🎵 STARTING TRADITIONAL SCAN", flush=True)

    try:
        data = request.get_json()
        folder_path = data.get('folder_path', '.')
        print(f"🎵 Scanning folder: {folder_path}", flush=True)

        if not Path(folder_path).exists():
            return jsonify({'error': 'Folder does not exist'}), 400

        # Get all MP3 files directly
        mp3_files = list(Path(folder_path).glob("*.mp3"))
        print(f"🎵 Found {len(mp3_files)} MP3 files", flush=True)

        if not mp3_files:
            return jsonify({
                'session_id': str(uuid.uuid4()),
                'folder_path': folder_path,
                'preview': {'loose_files': {'count': 0, 'groupings': {}}, 'existing_folders': [], 'total_albums': 0}
            })

        # TRADITIONAL GROUPING (MusicBrainz enhancement available per album)
        from merge_albums import AlbumMerger
        merger = AlbumMerger(folder_path)
        albums = merger.group_mp3s_by_album_traditional(mp3_files)
        print(f"🎵 Traditional grouping resulted in {len(albums)} album(s)", flush=True)

        # Build preview response
        preview = {
            'loose_files': {
                'count': len(mp3_files),
                'groupings': {}
            },
            'existing_folders': [],
            'total_albums': 0  # Will be set correctly below
        }

        # Collect individual tracks
        individual_tracks = []
        
        print(f"🔍 Processing {len(albums)} album groups:", flush=True)
        for album_name, files in albums.items():
            print(f"  - {album_name}: {len(files)} tracks", flush=True)
            
            if len(files) >= 2:  # Multi-track albums
                # Ensure we're only including the filenames, not full Path objects
                file_names = [f.name if isinstance(f, Path) else f for f in files]
                
                # Create a clean album name for display
                display_name = album_name
                if album_name.startswith("🎵 "):
                    display_name = album_name[2:]  # Remove the emoji for the display name
                
                preview['loose_files']['groupings'][display_name] = {
                    'files': file_names,
                    'count': len(files),
                    'type': 'mix' if 'mix' in album_name.lower() or 'kicks' in album_name.lower() else 'album'
                }
                print(f"🎵 Album: {display_name} - {len(files)} tracks", flush=True)
                print(f"🎵 Files: {file_names}", flush=True)
            else:  # Single tracks - collect for custom album
                individual_tracks.extend(files)
        
        # Add individual tracks as a custom album collection
        if individual_tracks:
            file_names = [f.name if isinstance(f, Path) else f for f in individual_tracks]
            preview['loose_files']['groupings']['Individual Tracks'] = {
                'files': file_names,
                'count': len(individual_tracks),
                'type': 'custom'
            }
            print(f"🎵 Individual Tracks Collection: {len(individual_tracks)} tracks", flush=True)
            print(f"🎵 Files: {file_names}", flush=True)
        
        # Set correct total albums count
        preview['total_albums'] = len(preview['loose_files']['groupings'])

        # Create session with AlbumMerger instance
        session_id = str(uuid.uuid4())

        # Import AlbumMerger and create instance
        from merge_albums import AlbumMerger
        merger = AlbumMerger(folder_path)

        # Store in session manager with scan results
        session_manager.create_session(session_id, {
            'merger': merger,
            'folder_path': folder_path,
            'grouped_albums': albums,  # Store the grouped albums from scan
            'preview': preview
        })

        return jsonify({
            'session_id': session_id,
            'folder_path': folder_path,
            'preview': preview
        })

    except Exception as e:
        print(f"🎵 ERROR: {str(e)}", flush=True)
        return jsonify({'error': str(e)}), 500


def musicbrainz_group_files(mp3_files):
    """Pure MusicBrainz-first grouping function"""
    print("🔍 Starting MusicBrainz lookup for all files...", flush=True)

    import eyed3
    import musicbrainzngs

    # Configure MusicBrainz
    musicbrainzngs.set_useragent("track-gluer-togetherer", "1.0", "https://github.com/user/track-gluer-togetherer")

    # Extract unique artist/album combinations
    album_candidates = {}
    for mp3_file in mp3_files:
        try:
            audiofile = eyed3.load(str(mp3_file))
            if audiofile and audiofile.tag:
                artist = audiofile.tag.artist or "Unknown"
                album = audiofile.tag.album or "Unknown"

                if artist != "Unknown" and album != "Unknown":
                    key = f"{artist}|||{album}"
                    if key not in album_candidates:
                        album_candidates[key] = []
                    album_candidates[key].append(mp3_file)
        except (OSError, ValueError, AttributeError) as e:
            logger.debug(f"Could not process MP3 file {mp3_file}: {e}")
            continue

    print(f"🔍 Found {len(album_candidates)} unique artist/album combinations", flush=True)

    # Try MusicBrainz for each combination
    final_groups = {}
    matched_files = set()

    for album_key, candidate_files in album_candidates.items():
        # Skip if we've already matched all files
        if len(matched_files) >= len(mp3_files):
            print(f"🎯 All files already matched, skipping remaining lookups", flush=True)
            break

        artist, album = album_key.split("|||")
        print(f"🔍 Looking up: {artist} - {album}", flush=True)

        try:
            # Search MusicBrainz with timeout
            musicbrainzngs.set_rate_limit(limit_or_interval=1.0, new_requests=1)
            result = musicbrainzngs.search_releases(artist=artist, release=album, limit=2)

            if result.get('release-list'):
                for release in result['release-list']:
                    try:
                        # Get track listing
                        detailed_release = musicbrainzngs.get_release_by_id(
                            release['id'],
                            includes=['recordings']
                        )

                        if 'medium-list' in detailed_release['release']:
                            # Extract track titles with position
                            mb_tracks = {}  # title -> position
                            for medium in detailed_release['release']['medium-list']:
                                if 'track-list' in medium:
                                    for track in medium['track-list']:
                                        title = track['recording']['title'].lower().strip()
                                        position = int(track.get('position', 0))
                                        mb_tracks[title] = position

                            # Match ONLY files from this candidate group to this release
                            release_files = []
                            for mp3_file in candidate_files:
                                if mp3_file in matched_files:
                                    continue

                                try:
                                    audio = eyed3.load(str(mp3_file))
                                    if audio and audio.tag and audio.tag.title:
                                        local_title = audio.tag.title.lower().strip()
                                        clean_title = local_title.replace(' - mixed', '').replace('-mixed', '')

                                        track_position = None
                                        if clean_title in mb_tracks:
                                            track_position = mb_tracks[clean_title]
                                        elif local_title in mb_tracks:
                                            track_position = mb_tracks[local_title]

                                        if track_position is not None:
                                            release_files.append((mp3_file, track_position))
                                            matched_files.add(mp3_file)
                                except (ValueError, KeyError, AttributeError) as e:
                                    logger.debug(f"Could not match track {mp3_file}: {e}")
                                    continue

                            if release_files:
                                # Sort by MusicBrainz track position
                                release_files.sort(key=lambda x: x[1])
                                sorted_files = [f[0] for f in release_files]  # Extract just the file paths

                                release_title = detailed_release['release']['title']
                                release_artist = detailed_release['release']['artist-credit'][0]['name'] if detailed_release['release'].get('artist-credit') else artist
                                mb_key = f"{release_artist} - {release_title}"
                                final_groups[mb_key] = sorted_files
                                print(f"✅ MusicBrainz match: {mb_key} ({len(sorted_files)} tracks, ordered by MusicBrainz)", flush=True)
                                break
                    except Exception as e:
                        print(f"❌ Error processing release: {e}", flush=True)
                        continue

        except Exception as e:
            print(f"❌ MusicBrainz lookup failed for {artist} - {album}: {e}", flush=True)

    # Add unmatched files using traditional grouping and individual tracks
    unmatched_files = [f for f in mp3_files if f not in matched_files]
    if unmatched_files:
        print(f"📝 {len(unmatched_files)} files unmatched, using fallback grouping", flush=True)

        # Group files by album first
        album_groups = {}
        individual_tracks = []

        for mp3_file in unmatched_files:
            try:
                audiofile = eyed3.load(str(mp3_file))
                if audiofile and audiofile.tag:
                    artist = audiofile.tag.artist or "Unknown"
                    album = audiofile.tag.album or "Unknown"

                    # If no album info or looks like a single, treat as individual
                    if album == "Unknown" or album.lower() in ["single", "singles", ""]:
                        individual_tracks.append(mp3_file)
                    else:
                        key = f"{artist} - {album}"
                        if key not in album_groups:
                            album_groups[key] = []
                        album_groups[key].append(mp3_file)
                else:
                    individual_tracks.append(mp3_file)
            except:
                individual_tracks.append(mp3_file)

        # Add album groups to final groups
        final_groups.update(album_groups)

        # Add individual tracks as a special group if any exist
        if individual_tracks:
            final_groups["🎵 Individual Tracks"] = individual_tracks
            print(f"📝 Added {len(individual_tracks)} individual tracks", flush=True)

    return final_groups


@app.route('/api/musicbrainz_resort', methods=['POST'])
@validate_request(require_json=True, rate_limit=True)
def musicbrainz_resort():
    """Scan folder and group files using MusicBrainz-first approach"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path', '.')
        print(f"🎵 MusicBrainz resort for folder: {folder_path}", flush=True)

        if not Path(folder_path).exists():
            return jsonify({'error': 'Folder does not exist'}), 400

        # Get all MP3 files
        mp3_files = list(Path(folder_path).glob("*.mp3"))
        print(f"🎵 Found {len(mp3_files)} MP3 files for MusicBrainz grouping", flush=True)

        if not mp3_files:
            return jsonify({
                'session_id': str(uuid.uuid4()),
                'folder_path': folder_path,
                'preview': {'loose_files': {'count': 0, 'groupings': {}}, 'existing_folders': [], 'total_albums': 0}
            })

        # MUSICBRAINZ-FIRST GROUPING
        grouped_albums = musicbrainz_group_files(mp3_files)
        print(f"🎵 MusicBrainz grouping resulted in {len(grouped_albums)} album(s)", flush=True)

        # Build preview response similar to scan_folder
        preview = {
            'loose_files': {
                'count': len(mp3_files),
                'groupings': {}
            },
            'existing_folders': [],
            'total_albums': 0  # Will be counted accurately below
        }

        # Convert MusicBrainz groups to preview format - filter incomplete albums
        filtered_albums = 0
        for album_name, files in grouped_albums.items():
            if len(files) >= 3:  # Only albums with 3+ tracks (filter incomplete albums)
                preview['loose_files']['groupings'][album_name] = {
                    'type': 'album',
                    'count': len(files),
                    'files': [{'name': f.name, 'path': str(f)} for f in files]
                }
                filtered_albums += 1
                print(f"  📀 {album_name}: {len(files)} tracks", flush=True)
            else:
                print(f"  ⏭️ Filtered out {album_name}: only {len(files)} tracks (incomplete)", flush=True)

        preview['total_albums'] = filtered_albums
        print(f"🔍 MusicBrainz scan: {len(grouped_albums)} raw albums → {filtered_albums} complete albums (3+ tracks)", flush=True)

        # Create session
        session_id = str(uuid.uuid4())
        merger = WebAlbumMerger(folder_path)
        session_manager.create_session(session_id, {'merger': merger, 'folder_path': folder_path})

        return jsonify({
            'session_id': session_id,
            'folder_path': folder_path,
            'preview': preview
        })

    except Exception as e:
        logger.error(f"Error in musicbrainz_resort: {e}", exc_info=True)
        return jsonify({'error': f'Failed to resort with MusicBrainz: {str(e)}'}), 500


@app.route('/api/glue_em_all', methods=['POST'])
@validate_request(require_json=True, rate_limit=True)
def glue_em_all():
    """Load all MP3 files in folder as one big compilation for merging"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path', '.')
        print(f"🎵 Glue Em All for folder: {folder_path}", flush=True)

        if not Path(folder_path).exists():
            return jsonify({'error': 'Folder does not exist'}), 400

        # Get all MP3 files
        mp3_files = list(Path(folder_path).glob("*.mp3"))
        print(f"🎵 Found {len(mp3_files)} MP3 files for merging", flush=True)

        if not mp3_files:
            return jsonify({
                'session_id': str(uuid.uuid4()),
                'folder_path': folder_path,
                'preview': {'loose_files': {'count': 0, 'groupings': {}}, 'existing_folders': [], 'total_albums': 0}
            })

        # Sort files alphabetically by name for consistent ordering
        mp3_files.sort(key=lambda f: f.name.lower())

        # Create one big "All Files" album
        album_name = f"All Files ({len(mp3_files)} tracks)"

        preview = {
            'loose_files': {
                'count': len(mp3_files),
                'groupings': {
                    album_name: {
                        'type': 'mix',
                        'count': len(mp3_files),
                        'files': [{'name': f.name, 'path': str(f)} for f in mp3_files]
                    }
                }
            },
            'existing_folders': [],
            'total_albums': 1
        }

        print(f"📀 Created compilation: {album_name}", flush=True)

        # Create session
        session_id = str(uuid.uuid4())
        merger = WebAlbumMerger(folder_path)
        session_manager.create_session(session_id, {'merger': merger, 'folder_path': folder_path})

        return jsonify({
            'session_id': session_id,
            'folder_path': folder_path,
            'preview': preview
        })

    except Exception as e:
        logger.error(f"Error in glue_em_all: {e}", exc_info=True)
        return jsonify({'error': f'Failed to load all files: {str(e)}'}), 500


@app.route('/api/start_merge', methods=['POST'])
@validate_request(require_json=True, rate_limit=True)
def start_merge():
    """Start the merging process"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        merge_type = data.get('merge_type', 'albums')
        output_path = data.get('output_path', '.')
        delete_originals = data.get('delete_originals', False)
        selected_albums = data.get('selected_albums', [])
        custom_track_orders = data.get('custom_track_orders', {})
        custom_album_titles = data.get('custom_album_titles', {})
        custom_album_files = data.get('custom_album_files', {})
        
        print(f"🔍 MERGE DEBUG - Selected albums: {selected_albums}")
        print(f"🔍 MERGE DEBUG - Custom titles: {custom_album_titles}")
        print(f"🔍 MERGE DEBUG - Custom files: {custom_album_files}")

        if session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400

        merger = active_sessions[session_id]

        # Start merging in background thread
        def merge_process():
            try:
                print(f"🚀 Starting merge process for session: {session_id}")
                if merge_type == 'all':
                    merge_with_progress(session_id, output_path, delete_originals)
                else:
                    merge_with_progress(session_id, output_path, delete_originals, selected_albums, custom_track_orders, custom_album_titles, custom_album_files)
                print(f"✅ Merge process completed for session: {session_id}")
            except Exception as e:
                print(f"❌ Merge process error for session {session_id}: {e}")
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=merge_process)
        thread.daemon = True
        thread.start()

        return jsonify({'status': 'started'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/browse_folder', methods=['POST'])
def browse_folder():
    """Browse folders in the file system"""
    try:
        print("\n=== /api/browse_folder endpoint called ===")
        print(f"Request data: {request.data}")
        print(f"Request headers: {request.headers}")
        
        if not request.is_json:
            print("Error: Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        print(f"Parsed JSON data: {data}")
        
        folder_path = data.get('folder_path', '.')
        print(f"Requested folder path: {folder_path}")
        
        # Convert to absolute path if relative
        if not os.path.isabs(folder_path):
            folder_path = os.path.abspath(folder_path)
        
        # Get parent directory
        parent_dir = os.path.dirname(folder_path)
        
        # Get directory contents
        items = []
        try:
            with os.scandir(folder_path) as dir_entries:
                for entry in dir_entries:
                    # Skip hidden files/folders
                    if entry.name.startswith('.'):
                        continue
                        
                    try:
                        item_info = {
                            'name': entry.name,
                            'path': entry.path,
                            'is_dir': entry.is_dir(),
                            'type': 'folder' if entry.is_dir() else 'file'
                        }
                        
                        # Get file count for directories
                        if entry.is_dir():
                            try:
                                mp3_count = len([f for f in os.listdir(entry.path) if f.lower().endswith('.mp3')])
                                item_info['mp3_count'] = mp3_count
                            except:
                                item_info['mp3_count'] = 0
                                
                        items.append(item_info)
                    except:
                        continue
        except PermissionError:
            return jsonify({'error': 'Permission denied'}), 403
        except FileNotFoundError:
            return jsonify({'error': 'Directory not found'}), 404
        
        # Sort: directories first, then files, both alphabetically
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
        
        return jsonify({
            'current_path': folder_path,
            'parent_path': parent_dir if parent_dir != folder_path else None,
            'items': items
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress/<session_id>')
def get_progress(session_id):
    """Get progress updates"""
    if session_id not in merge_progress:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    progress = merge_progress[session_id].copy()
    
    # Add any additional progress information here
    return jsonify(progress)

@app.route('/api/enhance_with_musicbrainz', methods=['POST'])
def enhance_with_musicbrainz():
    """Enhance selected album with MusicBrainz data"""
    try:
        print("\n=== /api/enhance_with_musicbrainz endpoint called ===")
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        folder_path = data.get('folder_path', '.')
        album_files = data.get('album_files', [])
        album_name = data.get('album_name', '')
        
        if not album_files:
            return jsonify({'error': 'No album files provided'}), 400
        
        print(f"Enhancing album: {album_name}")
        print(f"Files: {album_files}")
        print(f"Files type: {type(album_files)}")
        
        # Convert file names to Path objects
        from pathlib import Path
        folder = Path(folder_path)
        print(f"Folder path: {folder}")
        
        # Ensure album_files is a list
        if not isinstance(album_files, list):
            print(f"ERROR: album_files is not a list, it's {type(album_files)}")
            return jsonify({'error': f'album_files must be a list, got {type(album_files)}'}), 400
        
        mp3_files = []
        for filename in album_files:
            file_path = folder / filename
            print(f"Checking file: {file_path}")
            if file_path.exists():
                mp3_files.append(file_path)
                print(f"  ✅ Found: {file_path}")
            else:
                print(f"  ❌ Not found: {file_path}")
        
        if not mp3_files:
            return jsonify({'error': 'No valid MP3 files found'}), 400
        
        # Create AlbumMerger instance
        merger = AlbumMerger(folder_path)
        
        # Enhance with MusicBrainz
        result = merger.enhance_album_with_musicbrainz(mp3_files, album_name)
        
        if result['success']:
            # Format the response for the frontend
            response = {
                'success': True,
                'release_info': {
                    'title': result['release_info']['title'],
                    'artist': result['release_info']['artist'],
                    'track_count': result['release_info']['track_count'],
                    'date': result['release_info'].get('date', ''),
                    'mbid': result['release_info'].get('mbid', '')
                },
                'matched_tracks': [
                    {
                        'filename': track['file'].name,
                        'local_title': track['local_title'],
                        'mb_title': track['mb_title'],
                        'track_number': track['mb_track_number'],
                        'duration': track['mb_duration']
                    }
                    for track in result['matched_tracks']
                ],
                'unmatched_files': [
                    {
                        'filename': track['file'].name,
                        'local_title': track['local_title'],
                        'error': track.get('error', '')
                    }
                    for track in result['unmatched_files']
                ],
                'match_percentage': result['match_percentage'],
                'matched_count': result['matched_count'],
                'unmatched_count': result['unmatched_count'],
                'total_files': result['total_files']
            }
            
            return jsonify(response)
        else:
            return jsonify({'success': False, 'error': result['error']}), 400
            
    except Exception as e:
        print(f"Error in enhance_with_musicbrainz: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Enhanced health check endpoint with system stats"""
    try:
        # Check ffmpeg availability
        ffmpeg_available = False
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
            ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Get cache statistics
        mp3_cache_info = FileSystemCache.get_mp3_files.cache_info()
        metadata_cache_info = FileSystemCache.get_file_metadata.cache_info()
        
        return jsonify({
            'status': 'healthy',
            'version': '2.0.0',
            'system': {
                'ffmpeg_available': ffmpeg_available,
                'python_version': sys.version.split()[0],
                'platform': sys.platform
            },
            'sessions': {
                'active_count': len(session_manager.sessions),
                'max_sessions': session_manager.max_sessions,
                'ttl_minutes': session_manager.ttl.total_seconds() / 60
            },
            'cache': {
                'mp3_files': {
                    'hits': mp3_cache_info.hits,
                    'misses': mp3_cache_info.misses,
                    'current_size': mp3_cache_info.currsize,
                    'max_size': mp3_cache_info.maxsize
                },
                'metadata': {
                    'hits': metadata_cache_info.hits,
                    'misses': metadata_cache_info.misses,
                    'current_size': metadata_cache_info.currsize,
                    'max_size': metadata_cache_info.maxsize
                }
            },
            'config': {
                'debug': config.DEBUG,
                'log_level': config.LOG_LEVEL,
                'max_file_size_mb': config.MAX_FILE_SIZE_MB
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
@validate_request(rate_limit=True)
def clear_cache():
    """Clear file system cache"""
    try:
        FileSystemCache.clear_cache()
        logger.info("Cache cleared via API request")
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/stats')
def cache_stats():
    """Get cache statistics"""
    try:
        mp3_cache_info = FileSystemCache.get_mp3_files.cache_info()
        metadata_cache_info = FileSystemCache.get_file_metadata.cache_info()
        
        return jsonify({
            'mp3_files': {
                'hits': mp3_cache_info.hits,
                'misses': mp3_cache_info.misses,
                'hit_rate': mp3_cache_info.hits / (mp3_cache_info.hits + mp3_cache_info.misses) if (mp3_cache_info.hits + mp3_cache_info.misses) > 0 else 0,
                'current_size': mp3_cache_info.currsize,
                'max_size': mp3_cache_info.maxsize
            },
            'metadata': {
                'hits': metadata_cache_info.hits,
                'misses': metadata_cache_info.misses,
                'hit_rate': metadata_cache_info.hits / (metadata_cache_info.hits + metadata_cache_info.misses) if (metadata_cache_info.hits + metadata_cache_info.misses) > 0 else 0,
                'current_size': metadata_cache_info.currsize,
                'max_size': metadata_cache_info.maxsize
            }
        })
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<task_id>')
def get_task_status(task_id: str):
    """Get background task status"""
    try:
        status = task_queue.get_task_status(task_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Task status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/cleanup', methods=['POST'])
@validate_request(rate_limit=True)
def cleanup_tasks():
    """Clean up completed background tasks"""
    try:
        task_queue.cleanup_completed_tasks()
        return jsonify({'message': 'Completed tasks cleaned up'})
    except Exception as e:
        logger.error(f"Task cleanup error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shutdown', methods=['POST'])
@validate_request(rate_limit=False)  # No rate limit for shutdown
def shutdown():
    """Shutdown the server"""
    try:
        logger.info("Server shutdown requested")
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            # For newer versions of Werkzeug, use this method
            import os
            import signal
            os.kill(os.getpid(), signal.SIGINT)
        else:
            func()

        return jsonify({'message': 'Server shutting down...'}), 200
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
        return jsonify({'error': str(e)}), 500

# Use configuration for port
PORT = config.PORT

def open_browser():
    """Open browser after delay"""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{PORT}')

if __name__ == '__main__':
    print("🎵 Album Merger Web Interface")
    print("=" * 40)
    print(f"Starting server on http://localhost:{PORT}")
    print("Press Ctrl+C to stop")
    
    # Check if running in Electron
    import sys
    is_electron = '--electron' in sys.argv
    
    # Start the Flask server with debug mode if not in Electron
    debug_mode = not is_electron
    
    # Configure server options
    server_options = {
        'host': '0.0.0.0',
        'port': PORT,  # Use the global PORT variable
        'debug': debug_mode,
        'use_reloader': debug_mode,
        'threaded': True
    }
    
    # Always run in the main thread
    print(f"Server starting on http://127.0.0.1:{PORT}")
    
    # Open browser after a short delay if not in Electron mode
    if not is_electron:
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://127.0.0.1:{PORT}')
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the server
    try:
        app.run(**server_options)
    except Exception as e:
        print(f"Error starting server: {e}")
        raise