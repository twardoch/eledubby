# this_file: tests/test_eledubby/test_benchmarks.py
"""Performance benchmarks for eledubby components."""

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile


class TestBenchmarks:
    """Performance benchmarks for core components."""

    @pytest.fixture
    def sample_wav_file(self, tmp_path: Path) -> Path:
        """Generate sample WAV file for benchmarks."""
        sample_rate = 44100
        duration = 60  # 1 minute of audio
        # Generate 1 minute of sine wave with some noise
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        audio = (audio * 32767).astype(np.int16)

        wav_path = tmp_path / "test_audio.wav"
        wavfile.write(str(wav_path), sample_rate, audio)
        return wav_path

    def test_silence_analyzer_performance(self, sample_wav_file: Path) -> None:
        """Benchmark silence analysis on 1 minute of audio."""
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer()

        start = time.perf_counter()
        result = analyzer.analyze(str(sample_wav_file), min_duration=10.0, max_duration=20.0)
        elapsed = time.perf_counter() - start

        # Should complete within 2 seconds for 1 minute audio
        assert elapsed < 2.0, f"Silence analysis took {elapsed:.2f}s (expected <2s)"
        assert isinstance(result, list)
        assert len(result) > 0

    def test_checkpoint_serialization_performance(
        self, tmp_path: Path, sample_wav_file: Path
    ) -> None:
        """Benchmark checkpoint save/load operations."""
        from eledubby.utils.checkpoint import CheckpointManager

        manager = CheckpointManager(base_dir=tmp_path)

        # Create initial checkpoint
        segments = [(i * 10.0, (i + 1) * 10.0) for i in range(6)]
        manager.create_checkpoint(
            input_path=sample_wav_file,
            voice_id="test_voice",
            segments=segments,
            parameters={"param1": "value1"},
        )

        # Create segment files for update test
        segment_files = []
        for i in range(6):
            seg_file = tmp_path / f"segment_{i}.wav"
            seg_file.write_bytes(b"x" * 1000)
            segment_files.append(seg_file)

        # Benchmark update operations
        start = time.perf_counter()
        for i in range(6):
            manager.update_checkpoint(
                input_path=sample_wav_file,
                voice_id="test_voice",
                segment_idx=i,
                segment_path=str(segment_files[i]),
            )
        update_elapsed = time.perf_counter() - start

        # Benchmark load operations
        start = time.perf_counter()
        for _ in range(100):
            manager.load_checkpoint(sample_wav_file, "test_voice")
        load_elapsed = time.perf_counter() - start

        # 6 updates should complete quickly
        assert update_elapsed < 1.0, f"6 updates took {update_elapsed:.2f}s (expected <1s)"
        # 100 loads should complete within 1 second
        assert load_elapsed < 1.0, f"100 loads took {load_elapsed:.2f}s (expected <1s)"

    def test_hash_computation_performance(self, tmp_path: Path) -> None:
        """Benchmark file hash computation."""
        from eledubby.utils.checkpoint import CheckpointManager

        # Create a 10MB test file
        test_file = tmp_path / "test_large.bin"
        test_file.write_bytes(b"x" * (10 * 1024 * 1024))

        manager = CheckpointManager()

        start = time.perf_counter()
        hash_result = manager._compute_file_hash(str(test_file))
        elapsed = time.perf_counter() - start

        # Hash computation should complete within 0.5 seconds for 10MB
        assert elapsed < 0.5, f"Hash computation took {elapsed:.2f}s (expected <0.5s)"
        assert len(hash_result) == 64  # Full SHA256 hex digest

    def test_progress_tracker_context_manager(self) -> None:
        """Test progress tracker context manager operations."""
        from eledubby.utils.progress import ProgressTracker

        tracker = ProgressTracker()

        start = time.perf_counter()
        with tracker.track_segments(100, "Benchmark test") as update:
            for _ in range(100):
                update(1)
        elapsed = time.perf_counter() - start

        # 100 updates should complete within 1 second
        assert elapsed < 1.0, f"100 progress updates took {elapsed:.2f}s (expected <1s)"

    def test_processing_state_serialization_performance(self) -> None:
        """Benchmark ProcessingState serialization."""
        from eledubby.utils.checkpoint import ProcessingState

        # Create state with many segments
        state = ProcessingState(
            input_hash="abc123" * 10,
            voice_id="test_voice",
            segments=[(i * 10.0, (i + 1) * 10.0) for i in range(100)],
            processed_indices=list(range(50)),
            segment_paths={i: f"/path/to/segment_{i}.mp3" for i in range(50)},
            parameters={"param1": "value1", "param2": "value2"},
        )

        # Benchmark to_dict
        start = time.perf_counter()
        for _ in range(1000):
            state_dict = state.to_dict()
        to_dict_elapsed = time.perf_counter() - start

        # Benchmark JSON serialization
        start = time.perf_counter()
        for _ in range(1000):
            _ = json.dumps(state.to_dict())
        json_elapsed = time.perf_counter() - start

        # Benchmark from_dict
        start = time.perf_counter()
        for _ in range(1000):
            ProcessingState.from_dict(state_dict)
        from_dict_elapsed = time.perf_counter() - start

        # All operations should complete within 1 second for 1000 iterations
        assert to_dict_elapsed < 1.0, f"1000 to_dict took {to_dict_elapsed:.2f}s (expected <1s)"
        assert json_elapsed < 1.0, f"1000 JSON dumps took {json_elapsed:.2f}s (expected <1s)"
        assert from_dict_elapsed < 1.0, (
            f"1000 from_dict took {from_dict_elapsed:.2f}s (expected <1s)"
        )


class TestMemoryUsage:
    """Tests related to memory efficiency."""

    def test_large_segment_list_memory(self) -> None:
        """Test memory handling with many segments."""
        from eledubby.utils.checkpoint import ProcessingState

        # Create state with 10000 segments (very long video scenario)
        state = ProcessingState(
            input_hash="abc123",
            voice_id="test_voice",
            segments=[(i * 10.0, (i + 1) * 10.0) for i in range(10000)],
            processed_indices=list(range(5000)),
            segment_paths={i: f"/path/to/segment_{i}.mp3" for i in range(5000)},
        )

        # Convert to dict and back (serialization roundtrip)
        state_dict = state.to_dict()
        restored = ProcessingState.from_dict(state_dict)

        assert len(restored.segments) == 10000
        assert len(restored.processed_indices) == 5000
        assert len(restored.segment_paths) == 5000

    def test_silence_analyzer_chunk_processing(self, tmp_path: Path) -> None:
        """Test that silence analyzer can handle large files."""
        from eledubby.audio.analyzer import SilenceAnalyzer

        # Create 5-minute audio file
        sample_rate = 44100
        duration = 300  # 5 minutes
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        wav_path = tmp_path / "long_audio.wav"
        wavfile.write(str(wav_path), sample_rate, audio)

        analyzer = SilenceAnalyzer()

        start = time.perf_counter()
        result = analyzer.analyze(str(wav_path), min_duration=10.0, max_duration=20.0)
        elapsed = time.perf_counter() - start

        # 5 minutes should process within 10 seconds
        assert elapsed < 10.0, f"5-minute analysis took {elapsed:.2f}s (expected <10s)"
        # Should have ~15-30 segments for 5 minutes at 10-20s each
        assert 10 <= len(result) <= 40, f"Expected 10-40 segments, got {len(result)}"


class TestMemoryMonitoring:
    """Memory usage monitoring tests."""

    def test_silence_analyzer_memory_usage(self, tmp_path: Path) -> None:
        """Monitor memory usage during silence analysis."""
        from eledubby.audio.analyzer import SilenceAnalyzer

        # Create 2-minute audio file
        sample_rate = 44100
        duration = 120
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        wav_path = tmp_path / "memory_test_audio.wav"
        wavfile.write(str(wav_path), sample_rate, audio)

        analyzer = SilenceAnalyzer()

        # Start memory tracking
        tracemalloc.start()
        tracemalloc.reset_peak()

        _ = analyzer.analyze(str(wav_path), min_duration=10.0, max_duration=20.0)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be under 100MB for 2-minute audio
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f}MB exceeded 100MB limit"

    def test_checkpoint_state_memory_usage(self) -> None:
        """Monitor memory usage for checkpoint state with many segments."""
        from eledubby.utils.checkpoint import ProcessingState

        tracemalloc.start()
        tracemalloc.reset_peak()

        # Create state with 1000 segments (long video scenario)
        state = ProcessingState(
            input_hash="abc123" * 10,
            voice_id="test_voice",
            segments=[(i * 10.0, (i + 1) * 10.0) for i in range(1000)],
            processed_indices=list(range(500)),
            segment_paths={i: f"/path/to/segment_{i}.mp3" for i in range(500)},
            parameters={"param": "value"},
        )

        # Serialize and deserialize
        state_dict = state.to_dict()
        json_str = json.dumps(state_dict)
        _ = ProcessingState.from_dict(json.loads(json_str))

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should use less than 5MB for 1000 segments
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 5, f"Peak memory {peak_mb:.1f}MB exceeded 5MB limit"

    def test_audio_array_memory_efficiency(self) -> None:
        """Test that audio arrays use memory efficiently."""
        tracemalloc.start()
        tracemalloc.reset_peak()

        # Create 1-minute audio array (typical processing size)
        sample_rate = 44100
        duration = 60
        samples = int(sample_rate * duration)

        # int16 audio: 2 bytes per sample
        _ = np.zeros(samples, dtype=np.int16)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Expected: ~5.3MB (44100 * 60 * 2 bytes)
        expected_mb = (samples * 2) / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)

        # Should be within 2x of expected (accounting for numpy overhead)
        assert peak_mb < expected_mb * 2, (
            f"Memory {peak_mb:.1f}MB exceeded 2x expected {expected_mb:.1f}MB"
        )

    def test_progress_tracker_no_memory_leak(self) -> None:
        """Test that progress tracker doesn't accumulate memory."""
        from eledubby.utils.progress import ProgressTracker

        tracker = ProgressTracker()

        tracemalloc.start()
        tracemalloc.reset_peak()

        # Simulate many progress updates
        for _ in range(10):
            with tracker.track_segments(100, "Test") as update:
                for _ in range(100):
                    update(1)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should use less than 10MB for 1000 total updates
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 10, f"Peak memory {peak_mb:.1f}MB exceeded 10MB limit"
