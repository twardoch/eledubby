# this_file: api/elevenlabs_client.py
"""ElevenLabs API client wrapper."""

import os
import time

from elevenlabs import ElevenLabs
from elevenlabs.errors import (
    BadRequestError,
    ForbiddenError,
    NotFoundError,
    UnprocessableEntityError,
)
from loguru import logger


class ElevenLabsClient:
    """Wrapper for ElevenLabs API with retry logic and error handling."""

    def __init__(self, api_key: str | None = None, max_retries: int = 3):
        """Initialize ElevenLabs client.

        Args:
            api_key: API key (defaults to ELEVENLABS_API_KEY env var)
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not provided")

        self.max_retries = max_retries
        self.client = ElevenLabs(api_key=self.api_key)
        logger.debug("ElevenLabs client initialized")

    def speech_to_speech(
        self,
        audio_path: str,
        voice_id: str,
        output_path: str,
        model_id: str = "eleven_english_sts_v2",
    ) -> str:
        """Convert speech to speech with different voice.

        Args:
            audio_path: Path to input audio file
            voice_id: Target voice ID
            output_path: Path to save converted audio
            model_id: Model ID for conversion

        Returns:
            Path to converted audio file

        Raises:
            RuntimeError: If conversion fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Speech-to-speech attempt {attempt + 1}/{self.max_retries}")

                # Open audio file
                with open(audio_path, "rb") as audio_file:
                    # Perform conversion
                    audio_generator = self.client.speech_to_speech.convert(
                        audio=audio_file,
                        voice_id=voice_id,
                        model_id=model_id,
                        output_format="mp3_44100_128",
                    )

                    # Write output
                    with open(output_path, "wb") as output_file:
                        for chunk in audio_generator:
                            if chunk:
                                output_file.write(chunk)

                logger.info(f"Converted audio saved to: {output_path}")
                return output_path

            except (BadRequestError, ForbiddenError, NotFoundError, UnprocessableEntityError) as e:
                if hasattr(e, "status_code"):
                    if e.status_code == 429:  # Rate limit
                        wait_time = 2**attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    elif e.status_code == 401:  # Auth error
                        raise RuntimeError("Authentication failed - check API key") from e
                    else:
                        logger.error(f"API error: {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep(1)
                        else:
                            raise RuntimeError(f"API error after {self.max_retries} attempts: {e}") from e
                else:
                    logger.error(f"API error: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                    else:
                        raise RuntimeError(f"API error after {self.max_retries} attempts: {e}") from e

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Conversion failed after {self.max_retries} attempts: {e}") from e

        raise RuntimeError("Conversion failed - max retries exceeded")

    def validate_voice_id(self, voice_id: str) -> bool:
        """Validate if voice ID exists.

        Args:
            voice_id: Voice ID to validate

        Returns:
            True if voice exists, False otherwise
        """
        try:
            voices = self.client.voices.get_all()
            for voice in voices.voices:
                if voice.voice_id == voice_id:
                    logger.debug(f"Voice validated: {voice.name} ({voice_id})")
                    return True
            logger.warning(f"Voice ID not found: {voice_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to validate voice: {e}")
            return False

    def list_voices(self) -> list:
        """List available voices.

        Returns:
            List of voice dictionaries with id and name
        """
        try:
            voices = self.client.voices.get_all()
            voice_list = []
            for voice in voices.voices:
                voice_list.append(
                    {
                        "id": voice.voice_id,
                        "name": voice.name,
                        "category": getattr(voice, "category", "unknown"),
                    }
                )
            return voice_list
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []
