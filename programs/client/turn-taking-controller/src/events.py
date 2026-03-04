from dataclasses import dataclass
from typing import Literal

from shared_lib.events import now_ms


@dataclass
class STTEvent:
    """
    Event emitted when the STT has transcribed some audio from the client.

    """

    type: Literal["stt"]

    transcript: str
    """
    Complete transcription of the user's speech.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(cls, transcript: str) -> "STTEvent":
        """Factory method to create an AgentTextEvent with current timestamp."""
        return cls(type="stt", transcript=transcript, ts=now_ms())


TTCEvent = STTEvent
