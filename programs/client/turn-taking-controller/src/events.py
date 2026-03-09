from dataclasses import dataclass
from typing import Literal

from shared_lib.events import now_ms


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
