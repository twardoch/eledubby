# this_file: audio/processor.py
"""Audio processing module for timing preservation."""

import numpy as np
from loguru import logger
from pedalboard.io import AudioFile


class AudioProcessor:
    """Handles audio processing for timing preservation."""

    def measure_duration(self, audio_path: str) -> float:
        """Measure precise duration of audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            with AudioFile(audio_path) as f:
                return f.duration
        except Exception as e:
            logger.warning(f"Could not measure duration: {e}")
            return 0.0

    def adjust_duration(self, audio_path: str, target_duration: float, output_path: str) -> str:
        """Adjust audio duration to match target.

        Args:
            audio_path: Path to input audio
            target_duration: Target duration in seconds
            output_path: Path to save adjusted audio

        Returns:
            Path to adjusted audio file
        """
        current_duration = self.measure_duration(audio_path)
        difference = target_duration - current_duration

        logger.debug(
            f"Duration adjustment: current={current_duration:.3f}s, "
            f"target={target_duration:.3f}s, diff={difference:.3f}s"
        )

        if abs(difference) < 0.05:  # Within 50ms tolerance
            # Close enough, just copy
            self._copy_audio(audio_path, output_path)
            return output_path

        if difference > 0:
            # Need to pad with silence
            return self._pad_audio(audio_path, difference, output_path)
        else:
            # Need to trim
            return self._trim_audio(audio_path, target_duration, output_path)

    def _copy_audio(self, src_path: str, dst_path: str) -> None:
        """Copy audio file using pedalboard.

        Args:
            src_path: Source audio path
            dst_path: Destination audio path
        """
        with (
            AudioFile(src_path) as src,
            AudioFile(
                dst_path, "w", samplerate=src.samplerate, num_channels=src.num_channels
            ) as dst,
        ):
            chunk_size = src.samplerate  # 1 second chunks
            while src.tell() < src.frames:
                chunk = src.read(chunk_size)
                if chunk.size == 0:
                    break
                dst.write(chunk)

    def _pad_audio(self, audio_path: str, pad_duration: float, output_path: str) -> str:
        """Pad audio with silence.

        Args:
            audio_path: Path to input audio
            pad_duration: Duration of silence to add
            output_path: Path to save padded audio

        Returns:
            Path to padded audio file
        """
        with AudioFile(audio_path) as src:
            sample_rate = src.samplerate
            num_channels = src.num_channels
            audio_data = src.read(src.frames)

        # Create silence array (channels x samples)
        silence_frames = int(pad_duration * sample_rate)
        silence = np.zeros((num_channels, silence_frames), dtype=np.float32)

        # Concatenate audio with silence
        padded_audio = np.concatenate([audio_data, silence], axis=1)

        # Write padded audio
        with AudioFile(output_path, "w", samplerate=sample_rate, num_channels=num_channels) as dst:
            dst.write(padded_audio)

        logger.debug(f"Padded audio with {pad_duration:.3f}s of silence")
        return output_path

    def _trim_audio(self, audio_path: str, target_duration: float, output_path: str) -> str:
        """Trim audio to target duration.

        Args:
            audio_path: Path to input audio
            target_duration: Target duration in seconds
            output_path: Path to save trimmed audio

        Returns:
            Path to trimmed audio file
        """
        with AudioFile(audio_path) as src:
            sample_rate = src.samplerate
            num_channels = src.num_channels
            target_frames = int(target_duration * sample_rate)
            audio_data = src.read(target_frames)

        # Write trimmed audio
        with AudioFile(output_path, "w", samplerate=sample_rate, num_channels=num_channels) as dst:
            dst.write(audio_data)

        logger.debug(f"Trimmed audio to {target_duration:.3f}s")
        return output_path

    def compress_audio(
        self,
        audio_path: str,
        output_path: str,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
    ) -> str:
        """Apply dynamic range compression to audio.

        Args:
            audio_path: Path to input audio
            output_path: Path to save compressed audio
            threshold_db: Threshold in dB (default: -20)
            ratio: Compression ratio (default: 4:1)
            attack_ms: Attack time in ms (default: 5)
            release_ms: Release time in ms (default: 50)

        Returns:
            Path to compressed audio file
        """
        from pedalboard import Compressor, Pedalboard

        try:
            # Create compressor effect
            compressor = Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
            board = Pedalboard([compressor])

            # Read audio
            with AudioFile(audio_path) as src:
                sample_rate = src.samplerate
                num_channels = src.num_channels
                audio_data = src.read(src.frames)

            # Apply compression
            compressed = board(audio_data, sample_rate)

            # Write compressed audio
            with AudioFile(
                output_path, "w", samplerate=sample_rate, num_channels=num_channels
            ) as dst:
                dst.write(compressed)

            logger.debug(f"Applied compression: threshold={threshold_db}dB, ratio={ratio}:1")

        except Exception as e:
            logger.warning(f"Compression failed, copying original: {e}")
            self._copy_audio(audio_path, output_path)

        return output_path

    def normalize_audio(self, audio_path: str, output_path: str, target_db: float = -23.0) -> str:
        """Normalize audio levels using RMS-based loudness normalization.

        Note: This uses RMS normalization via pedalboard. For full EBU R128 compliance,
        FFmpeg's loudnorm filter would be needed, but this provides good results for
        most speech content.

        Args:
            audio_path: Path to input audio
            output_path: Path to save normalized audio
            target_db: Target loudness in dB (RMS-based)

        Returns:
            Path to normalized audio file
        """
        from pedalboard import Gain, Limiter, Pedalboard

        try:
            # Read audio
            with AudioFile(audio_path) as src:
                sample_rate = src.samplerate
                num_channels = src.num_channels
                audio_data = src.read(src.frames)

            # Calculate current RMS level
            rms = np.sqrt(np.mean(audio_data**2))
            current_db = 20 * np.log10(rms) if rms > 0 else -120

            # Calculate required gain adjustment
            gain_db = target_db - current_db

            # Apply gain and limiter to prevent clipping
            board = Pedalboard([Gain(gain_db=gain_db), Limiter(threshold_db=-1.5)])
            normalized = board(audio_data, sample_rate)

            # Write normalized audio
            with AudioFile(
                output_path, "w", samplerate=sample_rate, num_channels=num_channels
            ) as dst:
                dst.write(normalized)

            logger.debug(f"Applied normalization: target={target_db}dB, gain={gain_db:.1f}dB")

        except Exception as e:
            logger.warning(f"Normalization failed, copying original: {e}")
            self._copy_audio(audio_path, output_path)

        return output_path

    def reduce_noise(
        self,
        audio_path: str,
        output_path: str,
        noise_reduction_db: float = 12.0,
        noise_floor_db: float = -40.0,
    ) -> str:
        """Apply noise reduction to audio using NoiseGate.

        Uses pedalboard's NoiseGate for noise reduction. This is effective for
        reducing background noise during quiet passages.

        Args:
            audio_path: Path to input audio
            output_path: Path to save denoised audio
            noise_reduction_db: Amount of noise reduction in dB (default: 12)
            noise_floor_db: Noise floor in dB below which is considered noise (default: -40)

        Returns:
            Path to denoised audio file
        """
        from pedalboard import HighpassFilter, NoiseGate, Pedalboard

        try:
            # Read audio
            with AudioFile(audio_path) as src:
                sample_rate = src.samplerate
                num_channels = src.num_channels
                audio_data = src.read(src.frames)

            # Create noise reduction chain:
            # - HighpassFilter to remove low frequency rumble
            # - NoiseGate to reduce noise below threshold
            board = Pedalboard(
                [
                    HighpassFilter(cutoff_frequency_hz=80),
                    NoiseGate(
                        threshold_db=noise_floor_db,
                        ratio=noise_reduction_db / 6.0,  # Scale ratio based on reduction
                        attack_ms=1.0,
                        release_ms=100.0,
                    ),
                ]
            )
            denoised = board(audio_data, sample_rate)

            # Write denoised audio
            with AudioFile(
                output_path, "w", samplerate=sample_rate, num_channels=num_channels
            ) as dst:
                dst.write(denoised)

            logger.debug(
                f"Applied noise reduction: nr={noise_reduction_db}dB, nf={noise_floor_db}dB"
            )

        except Exception as e:
            logger.warning(f"Noise reduction failed, copying original: {e}")
            self._copy_audio(audio_path, output_path)

        return output_path

    def reduce_noise_advanced(
        self,
        audio_path: str,
        output_path: str,
        method: str = "gate",
        strength: float = 0.5,
    ) -> str:
        """Apply advanced noise reduction with selectable method.

        Args:
            audio_path: Path to input audio
            output_path: Path to save denoised audio
            method: Denoising method - 'gate' (NoiseGate) or 'expander' (softer gating)
            strength: Denoising strength from 0.0 (minimal) to 1.0 (maximum)

        Returns:
            Path to denoised audio file
        """
        from pedalboard import HighpassFilter, NoiseGate, Pedalboard

        strength = max(0.0, min(1.0, strength))

        try:
            # Read audio
            with AudioFile(audio_path) as src:
                sample_rate = src.samplerate
                num_channels = src.num_channels
                audio_data = src.read(src.frames)

            # Scale parameters based on strength
            threshold_db = -60 + (strength * 30)  # -60 to -30 dB threshold
            ratio = 1.5 + (strength * 8.5)  # 1.5:1 to 10:1 ratio
            highpass_hz = 60 + (strength * 60)  # 60 to 120 Hz highpass

            if method == "expander":
                # Softer expansion-style gating
                ratio = 1.2 + (strength * 2.8)  # 1.2:1 to 4:1 ratio
                release_ms = 200.0
            else:
                # Standard noise gate
                release_ms = 100.0

            board = Pedalboard(
                [
                    HighpassFilter(cutoff_frequency_hz=highpass_hz),
                    NoiseGate(
                        threshold_db=threshold_db,
                        ratio=ratio,
                        attack_ms=1.0,
                        release_ms=release_ms,
                    ),
                ]
            )
            denoised = board(audio_data, sample_rate)

            # Write denoised audio
            with AudioFile(
                output_path, "w", samplerate=sample_rate, num_channels=num_channels
            ) as dst:
                dst.write(denoised)

            logger.debug(f"Applied {method} noise reduction (strength={strength:.2f})")

        except Exception as e:
            logger.warning(f"Noise reduction ({method}) failed: {e}")
            self._copy_audio(audio_path, output_path)

        return output_path
