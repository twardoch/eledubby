# this_file: utils/temp_manager.py
"""Temporary file management utilities."""

import os
import shutil
import tempfile
from contextlib import contextmanager

from loguru import logger


class TempFileManager:
    """Manages temporary files and directories."""

    def __init__(self, prefix: str = "adamdubpy_"):
        """Initialize temp file manager.

        Args:
            prefix: Prefix for temporary directories
        """
        self.prefix = prefix
        self.temp_dirs: list[str] = []

    @contextmanager
    def temp_directory(self, cleanup: bool = True):
        """Create and manage a temporary directory.

        Args:
            cleanup: Whether to cleanup on exit

        Yields:
            Path to temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix=self.prefix)
        self.temp_dirs.append(temp_dir)
        logger.debug(f"Created temp directory: {temp_dir}")

        try:
            yield temp_dir
        finally:
            if cleanup:
                self.cleanup_directory(temp_dir)

    def cleanup_directory(self, directory: str):
        """Clean up a temporary directory.

        Args:
            directory: Directory to clean up
        """
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                logger.debug(f"Cleaned up: {directory}")
                if directory in self.temp_dirs:
                    self.temp_dirs.remove(directory)
            except Exception as e:
                logger.warning(f"Failed to cleanup {directory}: {e}")

    def cleanup_all(self):
        """Clean up all tracked temporary directories."""
        for directory in self.temp_dirs[:]:
            self.cleanup_directory(directory)

    def get_temp_path(self, directory: str, filename: str) -> str:
        """Get a path for a temporary file.

        Args:
            directory: Temporary directory
            filename: Filename to use

        Returns:
            Full path to temporary file
        """
        return os.path.join(directory, filename)

    @staticmethod
    def estimate_space_needed(video_path: str) -> float:
        """Estimate temporary space needed for processing.

        Args:
            video_path: Path to input video

        Returns:
            Estimated space needed in GB
        """
        try:
            video_size = os.path.getsize(video_path) / (1024**3)  # GB
            # Estimate: original + extracted audio + segments + converted segments
            estimated = video_size * 2.5
            return estimated
        except Exception:
            return 2.0  # Default estimate

    @staticmethod
    def check_disk_space(path: str, required_gb: float) -> bool:
        """Check if enough disk space is available.

        Args:
            path: Path to check
            required_gb: Required space in GB

        Returns:
            True if enough space available
        """
        try:
            stat = os.statvfs(path)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            return available_gb >= required_gb
        except Exception:
            return True  # Assume it's fine if we can't check
