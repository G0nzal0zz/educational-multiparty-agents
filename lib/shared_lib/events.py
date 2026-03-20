import base64
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union


def now_ms() -> int:
    """Return current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


class Role(Enum):
    HUMAN = 1
    TEACHER = 2
    STUDENT = 3


@dataclass
class SocketHumanTranscription:
    """
    Event emitted when the TTC has finished transcribing the human's speech.

    Sent from client to server.
    """

    type: Literal["human_transcription"]

    text: str
    """
    Complete transcription of the humans's speech.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(cls, text: str) -> "SocketHumanTranscription":
        """Factory method to create an SocketHumanTranscription with current timestamp."""
        return cls(type="human_transcription", text=text, ts=now_ms())


@dataclass
class SocketAgentTextChunkEvent:
    """
    Event emitted when an agent has generated a chunk of text.

    Sent from server to client.
    """

    type: Literal["agent_text_chunk"]

    role: Literal[Role.STUDENT, Role.TEACHER]
    """
    Role of the agent who was speaking.
    """

    text: str
    """
    Agent's intervention.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(
        cls, text: str, role: Literal[Role.STUDENT, Role.TEACHER]
    ) -> "SocketAgentTextChunkEvent":
        """Factory method to create an SocketAgentTextChunkEvent with current timestamp."""
        return cls(type="agent_text_chunk", text=text, role=role, ts=now_ms())


@dataclass
class SocketAgentTextEndEvent:
    """
    Event emitted when an agent has finished generating text.

    This event implies that the agent has finished sending events of type agent_chunk_event and the TTC
    shouldn't be waiting for more.

    Sent from server to client.
    """

    type: Literal["agent_text_end"]

    role: Literal[Role.STUDENT, Role.TEACHER]
    """
    Role of the agent who was speaking.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(
        cls, role: Literal[Role.STUDENT, Role.TEACHER]
    ) -> "SocketAgentTextEndEvent":
        """Factory method to create an SocketAgentTextEndEvent with current timestamp."""
        return cls(type="agent_text_end", role=role, ts=now_ms())


@dataclass
class SocketTeacherStartEvent:
    """
    Event emitted when an agent has finished generating text.

    This event implies that the agent has finished sending events of type agent_chunk_event and the TTC
    shouldn't be waiting for more.

    Sent from server to client.
    """

    type: Literal["teacher_start_teaching"]

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(cls) -> "SocketTeacherStartEvent":
        """Factory method to create an SocketTeacherStartEvent with current timestamp."""
        return cls(type="teacher_start_teaching", ts=now_ms())


# Union types for type-safe event handling

SocketClientEvent = SocketHumanTranscription | SocketTeacherStartEvent
"""Events sent from the turn-taking controller (client) to the server."""

SocketServerEvent = SocketAgentTextChunkEvent | SocketAgentTextEndEvent
"""Events sent from the server to the turn-taking controller (client)."""

SocketEvent = SocketClientEvent | SocketServerEvent


def event_to_dict(event: SocketEvent) -> dict:
    """Convert an event to a JSON-serializable dictionary."""
    if isinstance(event, SocketHumanTranscription):
        return {
            "type": event.type,
            "text": event.text,
            "ts": event.ts,
        }
    elif isinstance(event, SocketTeacherStartEvent):
        return {
            "type": event.type,
            "ts": event.ts,
        }
    elif isinstance(event, SocketAgentTextChunkEvent):
        return {
            "type": event.type,
            "role": event.role.value,
            "text": event.text,
            "ts": event.ts,
        }
    elif isinstance(event, SocketAgentTextEndEvent):
        return {
            "type": event.type,
            "role": event.role.value,
            "ts": event.ts,
        }
    else:
        raise ValueError(f"Unknown event type: {type(event)}")


def dict_to_event(data: dict) -> SocketClientEvent | SocketServerEvent:
    """
    Convert a dictionary to the appropriate event type.

    Args:
        data: Dictionary containing event data (typically from JSON parsing)

    Returns:
        The appropriate event object based on the "type" field

    Raises:
        ValueError: If the event type is unknown or required fields are missing
    """
    event_type = data.get("type")

    if event_type == "human_transcription":
        return SocketHumanTranscription(
            type=event_type, text=data.get("text", ""), ts=data.get("ts", now_ms())
        )

    elif event_type == "teacher_start_teaching":
        return SocketTeacherStartEvent(type=event_type, ts=data.get("ts", now_ms()))

    elif event_type == "agent_text_chunk":
        role_value = data.get("role", Role.TEACHER.value)
        role = Role(role_value) if isinstance(role_value, int) else Role[role_value]
        return SocketAgentTextChunkEvent(
            type=event_type,
            text=data.get("text", ""),
            role=role,
            ts=data.get("ts", now_ms()),
        )

    elif event_type == "agent_text_end":
        role_value = data.get("role", Role.TEACHER.value)
        role = Role(role_value) if isinstance(role_value, int) else Role[role_value]
        return SocketAgentTextEndEvent(
            type=event_type,
            role=role,
            ts=data.get("ts", now_ms()),
        )

    else:
        raise ValueError(f"Unknown event type: {event_type}")


def bytes_to_event(data: bytes | str) -> SocketClientEvent | SocketServerEvent:
    """
    Convert bytes or string containing JSON to the appropriate event type.

    This is a convenience wrapper around dict_to_event that handles
    JSON parsing.

    Args:
        data: Bytes or string containing JSON event data

    Returns:
        The appropriate event object based on the "type" field

    Raises:
        ValueError: If the JSON is invalid or event type is unknown
        json.JSONDecodeError: If the input is not valid JSON
    """
    import json

    if isinstance(data, bytes):
        data = data.decode("utf-8")

    # Handle newline-delimited JSON (multiple JSON objects separated by newlines)
    data = data.strip()
    if "\n" in data:
        # Take the first complete JSON object
        data = data.split("\n")[0]

    parsed = json.loads(data)
    return dict_to_event(parsed)
