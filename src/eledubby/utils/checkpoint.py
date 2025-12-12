# this_file: src/eledubby/utils/checkpoint.py
"""Checkpoint system for resume capability."""

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class ProcessingState:
    """State of a dubbing job for checkpoint/resume."""

    input_hash: str  # SHA256 of first 1MB of input file
    voice_id: str
    segments: list[tuple[float, float]]  # (start, end) times
    processed_indices: list[int] = field(default_factory=list)  # Indices of completed segments
    segment_paths: dict[int, str] = field(default_factory=dict)  # idx -> adjusted segment path
    parameters: dict = field(default_factory=dict)  # seg_min, seg_max, normalize, etc.
    preview: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert tuple list to nested list for JSON
        d["segments"] = [[s, e] for s, e in self.segments]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingState":
        """Create from dict (loaded from JSON)."""
        # Convert nested list back to tuples
        data["segments"] = [tuple(s) for s in data["segments"]]
        # Convert string keys back to int for segment_paths (JSON doesn't support int keys)
        if "segment_paths" in data:
            data["segment_paths"] = {int(k): v for k, v in data["segment_paths"].items()}
        return cls(**data)


class CheckpointManager:
    """Manages checkpoints for resume capability."""

    CHECKPOINT_DIR = ".eledubby_checkpoints"
    STATE_FILE = "state.json"
    SEGMENTS_DIR = "segments"

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize checkpoint manager.

        Args:
            base_dir: Base directory for checkpoints (default: current directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.checkpoint_dir = self.base_dir / self.CHECKPOINT_DIR

    def _compute_file_hash(self, file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute SHA256 hash of first chunk of file.

        Args:
            file_path: Path to file
            chunk_size: Size of chunk to hash (default: 1MB)

        Returns:
            Hex digest of hash
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            chunk = f.read(chunk_size)
            hasher.update(chunk)
        return hasher.hexdigest()

    def _get_job_id(self, input_path: str | Path, voice_id: str) -> str:
        """Generate unique job ID from input and voice.

        Args:
            input_path: Path to input file
            voice_id: Voice ID being used

        Returns:
            Job ID string
        """
        input_name = Path(input_path).stem
        file_hash = self._compute_file_hash(input_path)[:8]
        voice_short = voice_id[:8] if len(voice_id) > 8 else voice_id
        return f"{input_name}_{file_hash}_{voice_short}"

    def _get_job_dir(self, job_id: str) -> Path:
        """Get directory for a specific job.

        Args:
            job_id: Job identifier

        Returns:
            Path to job directory
        """
        return self.checkpoint_dir / job_id

    def has_checkpoint(self, input_path: str | Path, voice_id: str) -> bool:
        """Check if a valid checkpoint exists for this job.

        Args:
            input_path: Path to input file
            voice_id: Voice ID being used

        Returns:
            True if valid checkpoint exists
        """
        job_id = self._get_job_id(input_path, voice_id)
        job_dir = self._get_job_dir(job_id)
        state_file = job_dir / self.STATE_FILE

        if not state_file.exists():
            return False

        # Verify the checkpoint is for the same file
        try:
            state = self.load_checkpoint(input_path, voice_id)
            current_hash = self._compute_file_hash(input_path)
            return state.input_hash == current_hash
        except Exception as e:
            logger.warning(f"Invalid checkpoint: {e}")
            return False

    def create_checkpoint(
        self,
        input_path: str | Path,
        voice_id: str,
        segments: list[tuple[float, float]],
        parameters: dict,
        preview: float = 0.0,
    ) -> ProcessingState:
        """Create a new checkpoint for a job.

        Args:
            input_path: Path to input file
            voice_id: Voice ID being used
            segments: List of (start, end) segment times
            parameters: Processing parameters
            preview: Preview duration (0 = full file)

        Returns:
            New ProcessingState
        """
        job_id = self._get_job_id(input_path, voice_id)
        job_dir = self._get_job_dir(job_id)

        # Create directories
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / self.SEGMENTS_DIR).mkdir(exist_ok=True)

        # Create state
        state = ProcessingState(
            input_hash=self._compute_file_hash(input_path),
            voice_id=voice_id,
            segments=segments,
            processed_indices=[],
            segment_paths={},
            parameters=parameters,
            preview=preview,
        )

        self._save_state(job_id, state)
        logger.info(f"Created checkpoint: {job_id}")
        return state

    def load_checkpoint(self, input_path: str | Path, voice_id: str) -> ProcessingState:
        """Load checkpoint for a job.

        Args:
            input_path: Path to input file
            voice_id: Voice ID being used

        Returns:
            ProcessingState from checkpoint

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        job_id = self._get_job_id(input_path, voice_id)
        job_dir = self._get_job_dir(job_id)
        state_file = job_dir / self.STATE_FILE

        if not state_file.exists():
            raise FileNotFoundError(f"No checkpoint found for job: {job_id}")

        with open(state_file, encoding="utf-8") as f:
            data = json.load(f)

        return ProcessingState.from_dict(data)

    def update_checkpoint(
        self,
        input_path: str | Path,
        voice_id: str,
        segment_idx: int,
        segment_path: str,
    ) -> None:
        """Update checkpoint with a completed segment.

        Args:
            input_path: Path to input file
            voice_id: Voice ID being used
            segment_idx: Index of completed segment
            segment_path: Path to processed segment file
        """
        job_id = self._get_job_id(input_path, voice_id)
        state = self.load_checkpoint(input_path, voice_id)

        # Copy segment to checkpoint directory
        job_dir = self._get_job_dir(job_id)
        dest_path = job_dir / self.SEGMENTS_DIR / f"segment_{segment_idx:04d}.wav"
        shutil.copy2(segment_path, dest_path)

        # Update state
        if segment_idx not in state.processed_indices:
            state.processed_indices.append(segment_idx)
            state.processed_indices.sort()
        state.segment_paths[segment_idx] = str(dest_path)

        self._save_state(job_id, state)
        logger.debug(f"Checkpoint updated: segment {segment_idx}")

    def get_remaining_segments(self, input_path: str | Path, voice_id: str) -> list[int]:
        """Get indices of segments not yet processed.

        Args:
            input_path: Path to input file
            voice_id: Voice ID being used

        Returns:
            List of segment indices that need processing
        """
        state = self.load_checkpoint(input_path, voice_id)
        all_indices = set(range(len(state.segments)))
        processed = set(state.processed_indices)
        return sorted(all_indices - processed)

    def get_processed_segment_paths(self, input_path: str | Path, voice_id: str) -> dict[int, str]:
        """Get paths to already-processed segments.

        Args:
            input_path: Path to input file
            voice_id: Voice ID being used

        Returns:
            Dict mapping segment index to file path
        """
        state = self.load_checkpoint(input_path, voice_id)
        # Verify paths exist
        valid_paths = {}
        for idx, path in state.segment_paths.items():
            if os.path.exists(path):
                valid_paths[idx] = path
        return valid_paths

    def delete_checkpoint(self, input_path: str | Path, voice_id: str) -> None:
        """Delete checkpoint for a job.

        Args:
            input_path: Path to input file
            voice_id: Voice ID being used
        """
        job_id = self._get_job_id(input_path, voice_id)
        job_dir = self._get_job_dir(job_id)

        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info(f"Deleted checkpoint: {job_id}")

    def _save_state(self, job_id: str, state: ProcessingState) -> None:
        """Save state to disk.

        Args:
            job_id: Job identifier
            state: State to save
        """
        job_dir = self._get_job_dir(job_id)
        state_file = job_dir / self.STATE_FILE

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """Remove checkpoints older than specified age.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of checkpoints removed
        """
        import time

        removed = 0
        if not self.checkpoint_dir.exists():
            return 0

        max_age_seconds = max_age_days * 24 * 60 * 60
        now = time.time()

        for job_dir in self.checkpoint_dir.iterdir():
            if not job_dir.is_dir():
                continue

            state_file = job_dir / self.STATE_FILE
            if state_file.exists():
                age = now - state_file.stat().st_mtime
                if age > max_age_seconds:
                    shutil.rmtree(job_dir)
                    removed += 1
                    logger.debug(f"Removed old checkpoint: {job_dir.name}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} old checkpoints")

        return removed

    def list_checkpoints(self) -> list[dict]:
        """List all existing checkpoints.

        Returns:
            List of checkpoint info dicts
        """
        checkpoints = []
        if not self.checkpoint_dir.exists():
            return checkpoints

        for job_dir in self.checkpoint_dir.iterdir():
            if not job_dir.is_dir():
                continue

            state_file = job_dir / self.STATE_FILE
            if state_file.exists():
                try:
                    with open(state_file, encoding="utf-8") as f:
                        data = json.load(f)

                    total_segments = len(data.get("segments", []))
                    processed = len(data.get("processed_indices", []))
                    checkpoints.append(
                        {
                            "job_id": job_dir.name,
                            "voice_id": data.get("voice_id", "unknown"),
                            "total_segments": total_segments,
                            "processed_segments": processed,
                            "progress": f"{processed}/{total_segments}",
                            "modified": state_file.stat().st_mtime,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Could not read checkpoint {job_dir.name}: {e}")

        return sorted(checkpoints, key=lambda x: x.get("modified", 0), reverse=True)

    def recover_partial_result(
        self, input_path: str | Path, voice_id: str, output_path: str | Path
    ) -> tuple[str, int, int]:
        """Recover partial results by concatenating available processed segments.

        Creates an audio file from all processed segments so far, with silence
        placeholders for missing segments.

        Args:
            input_path: Path to original input file
            voice_id: Voice ID used for processing
            output_path: Where to save the partial result

        Returns:
            Tuple of (output_path, processed_count, total_count)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If no segments have been processed
        """
        import subprocess

        state = self.load_checkpoint(input_path, voice_id)

        if not state.processed_indices:
            raise RuntimeError("No segments have been processed yet")

        # Get valid segment paths
        valid_paths = self.get_processed_segment_paths(input_path, voice_id)

        if not valid_paths:
            raise RuntimeError("No valid segment files found in checkpoint")

        # Build the final audio with placeholders for missing segments
        # Create a concat list file
        output_path = Path(output_path)
        concat_file = output_path.parent / f".{output_path.stem}_concat.txt"

        segments_info = []
        for idx in range(len(state.segments)):
            start, end = state.segments[idx]
            duration = end - start

            if idx in valid_paths and Path(valid_paths[idx]).exists():
                # Use the processed segment
                segments_info.append(("file", valid_paths[idx]))
            else:
                # Create a silence placeholder with duration info
                segments_info.append(("silence", duration))

        # Write concat file and generate silence segments as needed
        temp_silence_files = []
        with open(concat_file, "w", encoding="utf-8") as f:
            for item_type, item_value in segments_info:
                if item_type == "file":
                    f.write(f"file '{item_value}'\n")
                else:
                    # Generate a temporary silence file
                    duration = item_value
                    silence_file = output_path.parent / f".silence_{len(temp_silence_files)}.wav"
                    temp_silence_files.append(silence_file)

                    # Create silence using ffmpeg
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-f",
                            "lavfi",
                            "-i",
                            "anullsrc=r=44100:cl=mono",
                            "-t",
                            str(duration),
                            "-y",
                            str(silence_file),
                        ],
                        capture_output=True,
                        check=True,
                    )
                    f.write(f"file '{silence_file}'\n")

        # Concatenate all segments
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c",
                    "copy",
                    "-y",
                    str(output_path),
                ],
                capture_output=True,
                check=True,
            )
        finally:
            # Cleanup temp files
            concat_file.unlink(missing_ok=True)
            for sf in temp_silence_files:
                sf.unlink(missing_ok=True)

        logger.info(f"Recovered partial result: {len(valid_paths)}/{len(state.segments)} segments")
        return str(output_path), len(valid_paths), len(state.segments)

    def get_checkpoint_progress(self, input_path: str | Path, voice_id: str) -> dict:
        """Get detailed progress information for a checkpoint.

        Args:
            input_path: Path to input file
            voice_id: Voice ID used

        Returns:
            Dict with progress details
        """
        state = self.load_checkpoint(input_path, voice_id)
        total = len(state.segments)
        processed = len(state.processed_indices)

        # Calculate total duration processed
        processed_duration = sum(
            state.segments[idx][1] - state.segments[idx][0] for idx in state.processed_indices
        )
        total_duration = sum(end - start for start, end in state.segments)

        return {
            "total_segments": total,
            "processed_segments": processed,
            "remaining_segments": total - processed,
            "percent_complete": (processed / total * 100) if total > 0 else 0,
            "processed_duration": processed_duration,
            "total_duration": total_duration,
            "recoverable": processed > 0,
        }
