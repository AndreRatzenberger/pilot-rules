# src/pilot_rules/collector/utils.py
import os
import datetime
from pathlib import Path
from typing import Dict, Any

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from a file."""
    metadata = {
        "path": file_path,
        "size_bytes": 0,
        "line_count": 0,
        "last_modified": "Unknown",
        "created": "Unknown",
    }

    try:
        p = Path(file_path)
        stats = p.stat()
        metadata["size_bytes"] = stats.st_size
        metadata["last_modified"] = datetime.datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        # ctime is platform dependent (creation on Windows, metadata change on Unix)
        # Use mtime as a reliable fallback for "created" if ctime is older than mtime
        ctime = stats.st_ctime
        mtime = stats.st_mtime
        best_ctime = ctime if ctime <= mtime else mtime # Heuristic
        metadata["created"] = datetime.datetime.fromtimestamp(best_ctime).strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Attempt to read as text, fallback for binary or encoding issues
            with p.open('r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                metadata["line_count"] = len(content.splitlines())
        except (OSError, UnicodeDecodeError) as read_err:
            # Handle cases where reading might fail (binary file, permissions etc.)
            print(f"Warning: Could not read content/count lines for {file_path}: {read_err}")
            metadata["line_count"] = 0 # Indicate unreadable or binary

    except Exception as e:
        print(f"Warning: Could not get complete metadata for {file_path}: {e}")

    return metadata