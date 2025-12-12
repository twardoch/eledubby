# this_file: tests/test_eledubby/utils/test_checkpoint.py
"""Tests for checkpoint system."""

import json
from pathlib import Path

import pytest

from eledubby.utils.checkpoint import CheckpointManager, ProcessingState


class TestProcessingState:
    """Tests for ProcessingState dataclass."""

    def test_to_dict_converts_tuples_to_lists(self) -> None:
        """Test that segments tuples are converted to lists for JSON."""
        state = ProcessingState(
            input_hash="abc123",
            voice_id="voice1",
            segments=[(0.0, 10.0), (10.0, 20.0)],
            processed_indices=[0],
            segment_paths={0: "/tmp/seg0.wav"},
            parameters={"seg_min": 10.0},
            preview=0.0,
        )
        d = state.to_dict()

        assert d["segments"] == [[0.0, 10.0], [10.0, 20.0]]
        assert d["input_hash"] == "abc123"
        assert d["processed_indices"] == [0]

    def test_from_dict_converts_lists_to_tuples(self) -> None:
        """Test that segments lists are converted back to tuples."""
        data = {
            "input_hash": "abc123",
            "voice_id": "voice1",
            "segments": [[0.0, 10.0], [10.0, 20.0]],
            "processed_indices": [0],
            "segment_paths": {0: "/tmp/seg0.wav"},
            "parameters": {"seg_min": 10.0},
            "preview": 0.0,
        }
        state = ProcessingState.from_dict(data)

        assert state.segments == [(0.0, 10.0), (10.0, 20.0)]
        assert state.input_hash == "abc123"


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for tests."""
        return tmp_path

    @pytest.fixture
    def manager(self, temp_dir: Path) -> CheckpointManager:
        """Create a CheckpointManager for tests."""
        return CheckpointManager(base_dir=temp_dir)

    @pytest.fixture
    def test_audio_file(self, temp_dir: Path) -> Path:
        """Create a test audio file for hashing."""
        audio_file = temp_dir / "test_input.wav"
        # Write some content to hash
        audio_file.write_bytes(b"test audio content" * 1000)
        return audio_file

    def test_compute_file_hash(self, manager: CheckpointManager, test_audio_file: Path) -> None:
        """Test file hash computation."""
        hash1 = manager._compute_file_hash(test_audio_file)
        hash2 = manager._compute_file_hash(test_audio_file)

        assert len(hash1) == 64  # SHA256 hex digest
        assert hash1 == hash2  # Same file should produce same hash

    def test_get_job_id(self, manager: CheckpointManager, test_audio_file: Path) -> None:
        """Test job ID generation."""
        job_id = manager._get_job_id(test_audio_file, "voice123")

        assert "test_input" in job_id
        assert "voice123" in job_id

    def test_create_checkpoint(self, manager: CheckpointManager, test_audio_file: Path) -> None:
        """Test checkpoint creation."""
        segments = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        parameters = {"seg_min": 10.0, "seg_max": 20.0}

        state = manager.create_checkpoint(
            test_audio_file, "voice1", segments, parameters, preview=0.0
        )

        assert state.voice_id == "voice1"
        assert state.segments == segments
        assert state.processed_indices == []
        assert len(state.input_hash) == 64

    def test_has_checkpoint(self, manager: CheckpointManager, test_audio_file: Path) -> None:
        """Test checkpoint existence check."""
        assert not manager.has_checkpoint(test_audio_file, "voice1")

        manager.create_checkpoint(test_audio_file, "voice1", [(0.0, 10.0)], {}, preview=0.0)

        assert manager.has_checkpoint(test_audio_file, "voice1")

    def test_load_checkpoint(self, manager: CheckpointManager, test_audio_file: Path) -> None:
        """Test checkpoint loading."""
        segments = [(0.0, 10.0), (10.0, 20.0)]
        manager.create_checkpoint(test_audio_file, "voice1", segments, {}, preview=5.0)

        loaded = manager.load_checkpoint(test_audio_file, "voice1")

        assert loaded.voice_id == "voice1"
        assert loaded.segments == segments
        assert loaded.preview == 5.0

    def test_update_checkpoint(
        self, manager: CheckpointManager, test_audio_file: Path, temp_dir: Path
    ) -> None:
        """Test checkpoint update with processed segment."""
        segments = [(0.0, 10.0), (10.0, 20.0)]
        manager.create_checkpoint(test_audio_file, "voice1", segments, {})

        # Create a segment file
        segment_file = temp_dir / "segment_0.wav"
        segment_file.write_bytes(b"segment audio data")

        manager.update_checkpoint(test_audio_file, "voice1", 0, str(segment_file))

        loaded = manager.load_checkpoint(test_audio_file, "voice1")
        assert 0 in loaded.processed_indices
        assert 0 in loaded.segment_paths

    def test_get_remaining_segments(
        self, manager: CheckpointManager, test_audio_file: Path, temp_dir: Path
    ) -> None:
        """Test getting remaining unprocessed segments."""
        segments = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        manager.create_checkpoint(test_audio_file, "voice1", segments, {})

        # Mark segment 0 and 2 as processed
        seg0 = temp_dir / "seg0.wav"
        seg2 = temp_dir / "seg2.wav"
        seg0.write_bytes(b"data")
        seg2.write_bytes(b"data")

        manager.update_checkpoint(test_audio_file, "voice1", 0, str(seg0))
        manager.update_checkpoint(test_audio_file, "voice1", 2, str(seg2))

        remaining = manager.get_remaining_segments(test_audio_file, "voice1")
        assert remaining == [1]

    def test_delete_checkpoint(self, manager: CheckpointManager, test_audio_file: Path) -> None:
        """Test checkpoint deletion."""
        manager.create_checkpoint(test_audio_file, "voice1", [(0.0, 10.0)], {})
        assert manager.has_checkpoint(test_audio_file, "voice1")

        manager.delete_checkpoint(test_audio_file, "voice1")
        assert not manager.has_checkpoint(test_audio_file, "voice1")

    def test_list_checkpoints(self, manager: CheckpointManager, test_audio_file: Path) -> None:
        """Test listing all checkpoints."""
        assert manager.list_checkpoints() == []

        manager.create_checkpoint(test_audio_file, "voice1", [(0.0, 10.0), (10.0, 20.0)], {})

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0]["voice_id"] == "voice1"
        assert checkpoints[0]["total_segments"] == 2
        assert checkpoints[0]["processed_segments"] == 0

    def test_checkpoint_hash_mismatch(
        self, manager: CheckpointManager, test_audio_file: Path
    ) -> None:
        """Test that modified input file invalidates checkpoint."""
        manager.create_checkpoint(test_audio_file, "voice1", [(0.0, 10.0)], {})
        assert manager.has_checkpoint(test_audio_file, "voice1")

        # Modify the input file
        test_audio_file.write_bytes(b"different content" * 1000)

        # Checkpoint should no longer be valid (hash mismatch)
        assert not manager.has_checkpoint(test_audio_file, "voice1")

    def test_load_checkpoint_not_found(
        self, manager: CheckpointManager, test_audio_file: Path
    ) -> None:
        """Test loading non-existent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            manager.load_checkpoint(test_audio_file, "nonexistent")


class TestCheckpointsCommand:
    """Tests for the checkpoints CLI command."""

    def test_checkpoints_empty(self, capsys) -> None:
        """Test checkpoints command with no checkpoints."""
        from unittest.mock import patch

        with patch("eledubby.eledubby.CheckpointManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.list_checkpoints.return_value = []

            from eledubby.eledubby import checkpoints

            checkpoints()

            captured = capsys.readouterr()
            assert "No checkpoints found" in captured.out

    def test_checkpoints_json_output(self, capsys) -> None:
        """Test checkpoints command with JSON output."""
        from unittest.mock import patch

        with patch("eledubby.eledubby.CheckpointManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.list_checkpoints.return_value = [
                {
                    "job_id": "test_job",
                    "voice_id": "voice1",
                    "total_segments": 5,
                    "processed_segments": 2,
                    "progress": "2/5",
                    "modified": 1700000000.0,
                }
            ]

            from eledubby.eledubby import checkpoints

            checkpoints(json=True)

            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert len(data) == 1
            assert data[0]["job_id"] == "test_job"

    def test_checkpoints_clean(self, capsys) -> None:
        """Test checkpoints cleanup."""
        from unittest.mock import patch

        with patch("eledubby.eledubby.CheckpointManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.cleanup_old_checkpoints.return_value = 3

            from eledubby.eledubby import checkpoints

            checkpoints(clean=True, max_age_days=5)

            mock_manager.cleanup_old_checkpoints.assert_called_once_with(5)
            captured = capsys.readouterr()
            assert "Removed 3 checkpoints" in captured.out


class TestPartialResultRecovery:
    """Tests for partial result recovery functionality."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CheckpointManager:
        """Create a checkpoint manager for testing."""
        return CheckpointManager(base_dir=tmp_path)

    @pytest.fixture
    def test_audio_file(self, tmp_path: Path) -> Path:
        """Create a test audio file."""
        audio_file = tmp_path / "test_input.wav"
        audio_file.write_bytes(b"audio data" * 1000)
        return audio_file

    def test_get_checkpoint_progress(
        self, manager: CheckpointManager, test_audio_file: Path, tmp_path: Path
    ) -> None:
        """Test getting checkpoint progress information."""
        segments = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        manager.create_checkpoint(test_audio_file, "voice1", segments, {})

        # Mark one segment as processed
        seg0 = tmp_path / "seg0.wav"
        seg0.write_bytes(b"data")
        manager.update_checkpoint(test_audio_file, "voice1", 0, str(seg0))

        progress = manager.get_checkpoint_progress(test_audio_file, "voice1")

        assert progress["total_segments"] == 3
        assert progress["processed_segments"] == 1
        assert progress["remaining_segments"] == 2
        assert progress["percent_complete"] == pytest.approx(33.33, rel=0.1)
        assert progress["processed_duration"] == 10.0
        assert progress["total_duration"] == 30.0
        assert progress["recoverable"] is True

    def test_get_checkpoint_progress_no_segments_processed(
        self, manager: CheckpointManager, test_audio_file: Path
    ) -> None:
        """Test progress when no segments processed."""
        segments = [(0.0, 10.0), (10.0, 20.0)]
        manager.create_checkpoint(test_audio_file, "voice1", segments, {})

        progress = manager.get_checkpoint_progress(test_audio_file, "voice1")

        assert progress["processed_segments"] == 0
        assert progress["recoverable"] is False

    def test_recover_partial_result_no_segments(
        self, manager: CheckpointManager, test_audio_file: Path, tmp_path: Path
    ) -> None:
        """Test that recovery fails when no segments processed."""
        segments = [(0.0, 10.0), (10.0, 20.0)]
        manager.create_checkpoint(test_audio_file, "voice1", segments, {})

        output_path = tmp_path / "output.wav"

        with pytest.raises(RuntimeError, match="No segments have been processed"):
            manager.recover_partial_result(test_audio_file, "voice1", output_path)
