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
    def create(cls, text: str) -> "STTEvent":
        """Factory method to create an AgentTextEvent with current timestamp."""
        return cls(type="stt", text=text, ts=now_ms())


TTCEvent = STTEvent


def event_to_dict(event: TTCEvent) -> dict:
    """Convert an event to a JSON-serializable dictionary."""
    if isinstance(event, STTEvent):
        return {
            "type": event.type,
            "text": event.transcript,
            "ts": event.ts,
        }
    else:
        raise ValueError(f"Unknown event type: {type(event)}")
