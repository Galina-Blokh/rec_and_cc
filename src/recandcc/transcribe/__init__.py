"""Local transcription of recorded sessions with faster-whisper."""

from .engine import Transcriber, auto_parallelism, transcribe_directory
from .queue import TranscriptionQueue

__all__ = ["Transcriber", "TranscriptionQueue", "auto_parallelism", "transcribe_directory"]
