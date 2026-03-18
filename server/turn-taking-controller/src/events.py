from dataclasses import dataclass
from typing import Literal

from shared_lib.events import Role, now_ms


@dataclass
class STTChunkEvent:
    """
    Event emitted when the STT has transcribed some chunk of audio from the client.

    """

    type: Literal["stt"]

    transcript: str
    """
    Complete transcription of the user's speech.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(cls, transcript: str) -> "STTChunkEvent":
        """Factory method to create an STTChunkEvent with current timestamp."""
        return cls(type="stt", transcript=transcript, ts=now_ms())


@dataclass
class STTEndEvent:
    """
    Event emitted when the STT has finished transcribing some audio from the client.

    """

    type: Literal["stt"]

    transcript: str
    """
    Complete transcription of the user's speech.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(cls, transcript: str) -> "STTEndEvent":
        """Factory method to create an STTEndEvent with current timestamp."""
        return cls(type="stt", transcript=transcript, ts=now_ms())


STTEvent = STTChunkEvent | STTEndEvent


@dataclass
class TTSEndEvent:
    """
    Event emitted when the TTS has finished playing some audio.

    """

    type: Literal["tts"]

    role: Literal[Role.STUDENT, Role.TEACHER]
    """Role of the speaker"""

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(cls, role: Literal[Role.STUDENT, Role.TEACHER]) -> "TTSEndEvent":
        """Factory method to create an TTSEndEvent with current timestamp."""
        return cls(type="tts", role=role, ts=now_ms())


TTSEvent = TTSEndEvent
