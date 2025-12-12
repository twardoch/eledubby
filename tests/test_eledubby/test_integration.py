# this_file: tests/test_eledubby/test_integration.py
"""Integration tests for eledubby dub command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eledubby.eledubby import (
    EleDubby,
    _expand_input_paths,
    dub,
)


class TestExpandInputPaths:
    """Tests for _expand_input_paths helper function."""

    def test_single_existing_file(self, tmp_path: Path) -> None:
        """Single file path returns list with that file."""
        test_file = tmp_path / "video.mp4"
        test_file.write_text("test")

        result = _expand_input_paths(str(test_file))

        assert result == [test_file]

    def test_single_nonexistent_file(self, tmp_path: Path) -> None:
        """Nonexistent file path returns empty list."""
        result = _expand_input_paths(str(tmp_path / "nonexistent.mp4"))

        assert result == []

    def test_glob_pattern_matches_files(self, tmp_path: Path) -> None:
        """Glob pattern expands to matching files."""
        (tmp_path / "a.mp4").write_text("a")
        (tmp_path / "b.mp4").write_text("b")
        (tmp_path / "c.wav").write_text("c")

        result = _expand_input_paths(str(tmp_path / "*.mp4"))

        assert len(result) == 2
        assert all(p.suffix == ".mp4" for p in result)

    def test_glob_pattern_no_matches(self, tmp_path: Path) -> None:
        """Glob pattern with no matches returns empty list."""
        result = _expand_input_paths(str(tmp_path / "*.xyz"))

        assert result == []

    def test_recursive_glob_pattern(self, tmp_path: Path) -> None:
        """Recursive glob pattern finds files in subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.mp4").write_text("root")
        (subdir / "nested.mp4").write_text("nested")

        result = _expand_input_paths(str(tmp_path / "**/*.mp4"))

        assert len(result) == 2

    def test_excludes_directories(self, tmp_path: Path) -> None:
        """Glob pattern excludes directories from results."""
        (tmp_path / "file.mp4").write_text("file")
        (tmp_path / "dir.mp4").mkdir()  # directory with .mp4 extension

        result = _expand_input_paths(str(tmp_path / "*.mp4"))

        assert len(result) == 1
        assert result[0].name == "file.mp4"


class TestEleDubbyInit:
    """Tests for EleDubby initialization."""

    @patch("eledubby.eledubby.subprocess.run")
    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
    def test_default_normalize_enabled(self, mock_run: MagicMock) -> None:
        """Normalization is enabled by default."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version")

        dubber = EleDubby(require_elevenlabs=False)

        assert dubber.normalize is True
        assert dubber.target_db == -23.0

    @patch("eledubby.eledubby.subprocess.run")
    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
    def test_normalize_can_be_disabled(self, mock_run: MagicMock) -> None:
        """Normalization can be disabled."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version")

        dubber = EleDubby(require_elevenlabs=False, normalize=False)

        assert dubber.normalize is False

    @patch("eledubby.eledubby.subprocess.run")
    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
    def test_custom_target_db(self, mock_run: MagicMock) -> None:
        """Custom target_db can be specified."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version")

        dubber = EleDubby(require_elevenlabs=False, target_db=-16.0)

        assert dubber.target_db == -16.0

    @patch("eledubby.eledubby.subprocess.run")
    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
    def test_parallel_minimum_is_one(self, mock_run: MagicMock) -> None:
        """Parallel workers cannot be less than 1."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version")

        dubber = EleDubby(require_elevenlabs=False, parallel=0)

        assert dubber.parallel == 1

    @patch("eledubby.eledubby.subprocess.run")
    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
    def test_preview_minimum_is_zero(self, mock_run: MagicMock) -> None:
        """Preview seconds cannot be negative."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version")

        dubber = EleDubby(require_elevenlabs=False, preview=-5)

        assert dubber.preview == 0


class TestDubBatchMode:
    """Tests for batch processing in dub command."""

    @patch("eledubby.eledubby.EleDubby")
    @patch("eledubby.eledubby._expand_input_paths")
    def test_batch_mode_processes_multiple_files(
        self, mock_expand: MagicMock, mock_dubber_class: MagicMock, tmp_path: Path
    ) -> None:
        """Batch mode processes all matched files."""
        # Setup mock files
        files = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
        for f in files:
            f.write_text("test")
        mock_expand.return_value = files

        # Setup mock dubber
        mock_dubber = MagicMock()
        mock_dubber_class.return_value = mock_dubber

        dub(input="*.mp4", voice="test_voice")

        assert mock_dubber.process.call_count == 3

    @patch("eledubby.eledubby.EleDubby")
    @patch("eledubby.eledubby._expand_input_paths")
    def test_batch_mode_creates_output_directory(
        self, mock_expand: MagicMock, mock_dubber_class: MagicMock, tmp_path: Path
    ) -> None:
        """Batch mode creates output directory if specified."""
        files = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        for f in files:
            f.write_text("test")
        mock_expand.return_value = files

        mock_dubber = MagicMock()
        mock_dubber_class.return_value = mock_dubber

        output_dir = tmp_path / "output"
        dub(input="*.mp4", voice="test_voice", output=str(output_dir))

        assert output_dir.exists()

    @patch("eledubby.eledubby.EleDubby")
    @patch("eledubby.eledubby._expand_input_paths")
    def test_batch_mode_continues_on_failure(
        self, mock_expand: MagicMock, mock_dubber_class: MagicMock, tmp_path: Path
    ) -> None:
        """Batch mode continues processing after a file fails."""
        files = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
        for f in files:
            f.write_text("test")
        mock_expand.return_value = files

        mock_dubber = MagicMock()
        # First call fails, others succeed
        mock_dubber.process.side_effect = [Exception("fail"), None, None]
        mock_dubber_class.return_value = mock_dubber

        # Should not raise, continues processing
        dub(input="*.mp4", voice="test_voice")

        assert mock_dubber.process.call_count == 3


class TestDubSingleFileMode:
    """Tests for single file processing in dub command."""

    @patch("eledubby.eledubby.EleDubby")
    @patch("eledubby.eledubby._expand_input_paths")
    def test_single_file_passes_normalize_param(
        self, mock_expand: MagicMock, mock_dubber_class: MagicMock, tmp_path: Path
    ) -> None:
        """Single file mode passes normalize parameter to EleDubby."""
        test_file = tmp_path / "video.mp4"
        test_file.write_text("test")
        mock_expand.return_value = [test_file]

        mock_dubber = MagicMock()
        mock_dubber_class.return_value = mock_dubber

        dub(input=str(test_file), voice="test_voice", normalize=False, target_db=-16.0)

        mock_dubber_class.assert_called_once()
        call_kwargs = mock_dubber_class.call_args.kwargs
        assert call_kwargs["normalize"] is False
        assert call_kwargs["target_db"] == -16.0

    @patch("eledubby.eledubby.EleDubby")
    @patch("eledubby.eledubby._expand_input_paths")
    def test_single_file_passes_parallel_param(
        self, mock_expand: MagicMock, mock_dubber_class: MagicMock, tmp_path: Path
    ) -> None:
        """Single file mode passes parallel parameter to EleDubby."""
        test_file = tmp_path / "video.mp4"
        test_file.write_text("test")
        mock_expand.return_value = [test_file]

        mock_dubber = MagicMock()
        mock_dubber_class.return_value = mock_dubber

        dub(input=str(test_file), voice="test_voice", parallel=4)

        mock_dubber_class.assert_called_once()
        call_kwargs = mock_dubber_class.call_args.kwargs
        assert call_kwargs["parallel"] == 4


class TestDubNoFilesFound:
    """Tests for error handling when no files are found."""

    @patch("eledubby.eledubby._expand_input_paths")
    def test_exits_when_no_files_found(self, mock_expand: MagicMock) -> None:
        """Exits with error when no files match the pattern."""
        mock_expand.return_value = []

        with pytest.raises(SystemExit) as exc_info:
            dub(input="nonexistent/*.mp4", voice="test_voice")

        assert exc_info.value.code == 1
