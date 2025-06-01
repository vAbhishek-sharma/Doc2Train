# utils/files.py
"""
Complete file utilities for Doc2Train v2.0 Enhanced
File operations, directory management, and file system utilities
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import mimetypes
from datetime import datetime

def get_supported_files(directory: str, recursive: bool = True, max_files: Optional[int] = None) -> List[str]:
    """
    Get all supported files from a directory

    Args:
        directory: Directory path to scan
        recursive: Whether to scan recursively
        max_files: Maximum number of files to return

    Returns:
        List of supported file paths
    """
    supported_files = []
    directory_path = Path(directory)

    if directory_path.is_file():
        # Single file
        if is_supported_format(str(directory_path)):
            return [str(directory_path)]
        else:
            return []

    if not directory_path.is_dir():
        return []

    try:
        from config.settings import SUPPORTED_FORMATS
        supported_extensions = set(SUPPORTED_FORMATS.keys())
    except ImportError:
        # Fallback to basic supported extensions
        supported_extensions = {'.pdf', '.txt', '.epub', '.png', '.jpg', '.jpeg', '.srt', '.vtt', '.bmp', '.tiff'}

    # Search for files
    if recursive:
        # Recursive search
        for ext in supported_extensions:
            pattern = f"**/*{ext}"
            files = list(directory_path.glob(pattern))
            supported_files.extend([str(f) for f in files])
    else:
        # Non-recursive search
        for ext in supported_extensions:
            pattern = f"*{ext}"
            files = list(directory_path.glob(pattern))
            supported_files.extend([str(f) for f in files])

    # Sort and limit results
    supported_files = sorted(supported_files)

    if max_files and len(supported_files) > max_files:
        supported_files = supported_files[:max_files]

    return supported_files

def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported"""
    try:
        from config.settings import SUPPORTED_FORMATS
        extension = Path(file_path).suffix.lower()
        return extension in SUPPORTED_FORMATS
    except ImportError:
        # Fallback to basic check
        supported_exts = {'.pdf', '.txt', '.epub', '.png', '.jpg', '.jpeg', '.srt', '.vtt', '.bmp', '.tiff'}
        return Path(file_path).suffix.lower() in supported_exts

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive file information

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file information
    """
    path = Path(file_path)

    try:
        stat = path.stat()

        # Basic file info
        info = {
            'name': path.name,
            'stem': path.stem,
            'extension': path.suffix.lower(),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'size_gb': stat.st_size / (1024 * 1024 * 1024),
            'modified': stat.st_mtime,
            'modified_datetime': datetime.fromtimestamp(stat.st_mtime),
            'created': stat.st_ctime,
            'created_datetime': datetime.fromtimestamp(stat.st_ctime),
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'exists': path.exists(),
            'absolute_path': str(path.absolute()),
            'relative_path': str(path),
            'parent_dir': str(path.parent)
        }

        # MIME type detection
        mime_type, encoding = mimetypes.guess_type(str(path))
        info['mime_type'] = mime_type
        info['encoding'] = encoding

        # File format support
        info['is_supported'] = is_supported_format(file_path)

        # File hash (for caching and deduplication)
        if path.is_file() and stat.st_size < 100 * 1024 * 1024:  # Only for files < 100MB
            info['md5_hash'] = calculate_file_hash(file_path)

        # Determine processor type
        if info['is_supported']:
            try:
                from config.settings import get_processor_for_file
                info['processor'] = get_processor_for_file(file_path)
            except ImportError:
                info['processor'] = 'unknown'

        return info

    except Exception as e:
        return {
            'name': path.name,
            'error': str(e),
            'exists': False,
            'is_supported': False
        }

def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate file hash for deduplication and caching

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        File hash as hex string
    """
    try:
        if algorithm == 'md5':
            hash_obj = hashlib.md5()
        elif algorithm == 'sha1':
            hash_obj = hashlib.sha1()
        elif algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    except Exception as e:
        return f"error_{str(e)}"

def find_duplicate_files(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    Find duplicate files based on content hash

    Args:
        file_paths: List of file paths to check

    Returns:
        Dictionary mapping hash to list of duplicate file paths
    """
    hash_to_files = {}

    for file_path in file_paths:
        try:
            file_hash = calculate_file_hash(file_path)
            if file_hash not in hash_to_files:
                hash_to_files[file_hash] = []
            hash_to_files[file_hash].append(file_path)
        except Exception:
            continue

    # Return only hashes with multiple files
    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}

    return duplicates

def organize_files_by_type(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    Organize files by their type/extension

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary mapping file type to list of files
    """
    organized = {}

    for file_path in file_paths:
        extension = Path(file_path).suffix.lower()
        if extension not in organized:
            organized[extension] = []
        organized[extension].append(file_path)

    return organized

def create_directory_structure(base_dir: str, structure: Dict[str, Any]) -> bool:
    """
    Create directory structure from dictionary

    Args:
        base_dir: Base directory path
        structure: Dictionary defining directory structure

    Returns:
        True if successful
    """
    try:
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        for name, content in structure.items():
            path = base_path / name

            if isinstance(content, dict):
                # Subdirectory
                create_directory_structure(str(path), content)
            else:
                # File or empty directory
                if content is None:
                    path.mkdir(exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(str(content))

        return True

    except Exception as e:
        print(f"❌ Error creating directory structure: {e}")
        return False

def safe_copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    Safely copy file with error handling

    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files

    Returns:
        True if successful
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            print(f"❌ Source file does not exist: {src}")
            return False

        if dst_path.exists() and not overwrite:
            print(f"⚠️ Destination exists, skipping: {dst}")
            return False

        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src, dst)

        return True

    except Exception as e:
        print(f"❌ Error copying file {src} to {dst}: {e}")
        return False

def safe_move_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    Safely move file with error handling

    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files

    Returns:
        True if successful
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            print(f"❌ Source file does not exist: {src}")
            return False

        if dst_path.exists() and not overwrite:
            print(f"⚠️ Destination exists, skipping: {dst}")
            return False

        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Move file
        shutil.move(src, dst)

        return True

    except Exception as e:
        print(f"❌ Error moving file {src} to {dst}: {e}")
        return False

def cleanup_empty_directories(directory: str) -> int:
    """
    Remove empty directories recursively

    Args:
        directory: Directory to clean up

    Returns:
        Number of directories removed
    """
    removed_count = 0
    directory_path = Path(directory)

    if not directory_path.is_dir():
        return 0

    try:
        # Walk bottom-up to remove empty subdirectories first
        for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
            dir_path = Path(dirpath)

            # Skip the root directory
            if dir_path == directory_path:
                continue

            try:
                # Try to remove if empty
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    removed_count += 1
            except OSError:
                # Directory not empty or permission error
                continue

        return removed_count

    except Exception as e:
        print(f"❌ Error cleaning up directories: {e}")
        return 0

def get_directory_size(directory: str) -> Tuple[int, int]:
    """
    Get total size and file count of directory

    Args:
        directory: Directory path

    Returns:
        Tuple of (total_size_bytes, file_count)
    """
    total_size = 0
    file_count = 0

    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except OSError:
                    continue

        return total_size, file_count

    except Exception:
        return 0, 0

def find_files_by_pattern(directory: str, pattern: str, recursive: bool = True) -> List[str]:
    """
    Find files matching a pattern

    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.pdf", "**/*.txt")
        recursive: Whether to search recursively

    Returns:
        List of matching file paths
    """
    try:
        directory_path = Path(directory)

        if recursive and not pattern.startswith("**/"):
            pattern = f"**/{pattern}"

        matches = list(directory_path.glob(pattern))
        return [str(f) for f in matches if f.is_file()]

    except Exception as e:
        print(f"❌ Error finding files with pattern {pattern}: {e}")
        return []

def backup_file(file_path: str, backup_dir: Optional[str] = None) -> Optional[str]:
    """
    Create a backup of a file

    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup (default: same directory)

    Returns:
        Path to backup file or None if failed
    """
    try:
        src_path = Path(file_path)

        if not src_path.exists():
            return None

        # Determine backup location
        if backup_dir:
            backup_path = Path(backup_dir) / src_path.name
        else:
            backup_path = src_path.parent / f"{src_path.stem}_backup{src_path.suffix}"

        # Create unique backup name if exists
        counter = 1
        original_backup_path = backup_path
        while backup_path.exists():
            backup_path = original_backup_path.parent / f"{original_backup_path.stem}_{counter}{original_backup_path.suffix}"
            counter += 1

        # Create backup directory if needed
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(file_path, backup_path)

        return str(backup_path)

    except Exception as e:
        print(f"❌ Error creating backup for {file_path}: {e}")
        return None

def validate_file_permissions(file_path: str) -> Dict[str, bool]:
    """
    Check file permissions

    Args:
        file_path: Path to file

    Returns:
        Dictionary with permission status
    """
    path = Path(file_path)

    permissions = {
        'exists': path.exists(),
        'readable': False,
        'writable': False,
        'executable': False
    }

    if permissions['exists']:
        permissions['readable'] = os.access(file_path, os.R_OK)
        permissions['writable'] = os.access(file_path, os.W_OK)
        permissions['executable'] = os.access(file_path, os.X_OK)

    return permissions

def ensure_directory_exists(directory: str, create_parents: bool = True) -> bool:
    """
    Ensure directory exists, create if necessary

    Args:
        directory: Directory path
        create_parents: Whether to create parent directories

    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory).mkdir(parents=create_parents, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ Error creating directory {directory}: {e}")
        return False

def get_file_encoding(file_path: str) -> Optional[str]:
    """
    Detect file encoding

    Args:
        file_path: Path to file

    Returns:
        Detected encoding or None
    """
    try:
        import chardet

        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result.get('encoding')

    except ImportError:
        # Fallback without chardet
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # Try to read some content
                return encoding
            except UnicodeDecodeError:
                continue

        return None
    except Exception:
        return None

def split_large_file(file_path: str, chunk_size_mb: int = 100) -> List[str]:
    """
    Split large file into smaller chunks

    Args:
        file_path: Path to file to split
        chunk_size_mb: Size of each chunk in MB

    Returns:
        List of chunk file paths
    """
    try:
        src_path = Path(file_path)
        chunk_size_bytes = chunk_size_mb * 1024 * 1024

        chunk_paths = []
        chunk_num = 1

        with open(file_path, 'rb') as src_file:
            while True:
                chunk_data = src_file.read(chunk_size_bytes)
                if not chunk_data:
                    break

                chunk_path = src_path.parent / f"{src_path.stem}_chunk_{chunk_num:03d}{src_path.suffix}"

                with open(chunk_path, 'wb') as chunk_file:
                    chunk_file.write(chunk_data)

                chunk_paths.append(str(chunk_path))
                chunk_num += 1

        return chunk_paths

    except Exception as e:
        print(f"❌ Error splitting file {file_path}: {e}")
        return []

def merge_files(chunk_paths: List[str], output_path: str) -> bool:
    """
    Merge multiple files into one

    Args:
        chunk_paths: List of chunk file paths
        output_path: Path for merged output file

    Returns:
        True if successful
    """
    try:
        with open(output_path, 'wb') as output_file:
            for chunk_path in chunk_paths:
                with open(chunk_path, 'rb') as chunk_file:
                    shutil.copyfileobj(chunk_file, output_file)

        return True

    except Exception as e:
        print(f"❌ Error merging files: {e}")
        return False

# File system monitoring utilities
class FileWatcher:
    """Simple file system watcher for monitoring changes"""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.initial_state = self._get_directory_state()

    def _get_directory_state(self) -> Dict[str, float]:
        """Get current state of directory"""
        state = {}

        try:
            for file_path in self.directory.rglob('*'):
                if file_path.is_file():
                    state[str(file_path)] = file_path.stat().st_mtime
        except Exception:
            pass

        return state

    def get_changes(self) -> Dict[str, List[str]]:
        """Get changes since last check"""
        current_state = self._get_directory_state()

        changes = {
            'added': [],
            'modified': [],
            'deleted': []
        }

        # Find added and modified files
        for file_path, mtime in current_state.items():
            if file_path not in self.initial_state:
                changes['added'].append(file_path)
            elif self.initial_state[file_path] != mtime:
                changes['modified'].append(file_path)

        # Find deleted files
        for file_path in self.initial_state:
            if file_path not in current_state:
                changes['deleted'].append(file_path)

        # Update initial state
        self.initial_state = current_state

        return changes

# Utility functions for common operations
def list_files_with_info(directory: str, extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    List files in directory with detailed information

    Args:
        directory: Directory to list
        extensions: Filter by extensions (e.g., ['.pdf', '.txt'])

    Returns:
        List of file information dictionaries
    """
    file_infos = []

    try:
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                if extensions:
                    if file_path.suffix.lower() not in extensions:
                        continue

                info = get_file_info(str(file_path))
                file_infos.append(info)

        # Sort by name
        file_infos.sort(key=lambda x: x.get('name', ''))

    except Exception as e:
        print(f"❌ Error listing files: {e}")

    return file_infos

def cleanup_temp_files(directory: str, patterns: List[str] = None) -> int:
    """
    Clean up temporary files

    Args:
        directory: Directory to clean
        patterns: File patterns to remove (default: common temp patterns)

    Returns:
        Number of files removed
    """
    if patterns is None:
        patterns = ['*.tmp', '*.temp', '*~', '.DS_Store', 'Thumbs.db', '*.bak']

    removed_count = 0

    try:
        for pattern in patterns:
            for file_path in Path(directory).rglob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        removed_count += 1
                except OSError:
                    continue

    except Exception as e:
        print(f"❌ Error cleaning temp files: {e}")

    return removed_count
