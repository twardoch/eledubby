# this_file: tests/eledubby/utils/test_temp_manager.py
"""Unit tests for TempFileManager."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestTempFileManager:
    """Tests for TempFileManager class."""

    def test_init_when_default_then_uses_adamdubpy_prefix(self) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        manager = TempFileManager()
        assert manager.prefix == "adamdubpy_"

    def test_init_when_custom_prefix_then_uses_it(self) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        manager = TempFileManager(prefix="test_")
        assert manager.prefix == "test_"

    def test_temp_directory_when_cleanup_true_then_removes_dir(self) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        manager = TempFileManager()

        with manager.temp_directory(cleanup=True) as temp_dir:
            assert os.path.exists(temp_dir)
            temp_path = temp_dir

        assert not os.path.exists(temp_path)

    def test_temp_directory_when_cleanup_false_then_keeps_dir(self) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        manager = TempFileManager()

        with manager.temp_directory(cleanup=False) as temp_dir:
            temp_path = temp_dir

        try:
            assert os.path.exists(temp_path)
        finally:
            # Manual cleanup
            os.rmdir(temp_path)

    def test_temp_directory_when_tracks_directories(self) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        manager = TempFileManager()

        with manager.temp_directory(cleanup=False) as temp_dir:
            assert temp_dir in manager.temp_dirs

        try:
            # Still tracked until cleanup_all
            pass
        finally:
            manager.cleanup_all()

    def test_cleanup_directory_when_exists_then_removes(self, tmp_path: Path) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        manager = TempFileManager()
        test_dir = tmp_path / "test_cleanup"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        manager.cleanup_directory(str(test_dir))

        assert not test_dir.exists()

    def test_cleanup_directory_when_not_exists_then_no_error(self) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        manager = TempFileManager()

        # Should not raise
        manager.cleanup_directory("/nonexistent/path")

    def test_cleanup_all_when_multiple_dirs_then_cleans_all(self) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        manager = TempFileManager()
        dirs = []

        for _ in range(3):
            with manager.temp_directory(cleanup=False) as temp_dir:
                dirs.append(temp_dir)

        assert len(manager.temp_dirs) == 3

        manager.cleanup_all()

        for d in dirs:
            assert not os.path.exists(d)
        assert len(manager.temp_dirs) == 0

    def test_get_temp_path_when_called_then_joins_correctly(self) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        manager = TempFileManager()

        path = manager.get_temp_path("/tmp/mydir", "test.wav")

        assert path == "/tmp/mydir/test.wav"


class TestTempFileManagerEstimateSpace:
    """Tests for TempFileManager.estimate_space_needed."""

    def test_estimate_space_needed_when_valid_file_then_returns_estimate(
        self, tmp_path: Path
    ) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        # Create a 1MB test file
        test_file = tmp_path / "video.mp4"
        test_file.write_bytes(b"x" * (1024 * 1024))

        estimate = TempFileManager.estimate_space_needed(str(test_file))

        # Should be about 2.5x the file size
        assert estimate == pytest.approx(0.001 * 2.5, rel=0.1)

    def test_estimate_space_needed_when_file_not_exists_then_returns_default(
        self,
    ) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        estimate = TempFileManager.estimate_space_needed("/nonexistent/video.mp4")

        assert estimate == 2.0  # Default estimate


class TestTempFileManagerCheckDiskSpace:
    """Tests for TempFileManager.check_disk_space."""

    def test_check_disk_space_when_enough_space_then_returns_true(self, tmp_path: Path) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        # 0.001 GB is a tiny amount that should always be available
        result = TempFileManager.check_disk_space(str(tmp_path), 0.001)

        assert result is True

    def test_check_disk_space_when_too_much_requested_then_returns_false(
        self, tmp_path: Path
    ) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        # 1 petabyte should not be available
        result = TempFileManager.check_disk_space(str(tmp_path), 1_000_000)

        assert result is False

    def test_check_disk_space_when_error_then_returns_true(self) -> None:
        from eledubby.utils.temp_manager import TempFileManager

        # Invalid path should cause error, but return True (assume OK)
        result = TempFileManager.check_disk_space("/nonexistent/path", 1.0)

        assert result is True
