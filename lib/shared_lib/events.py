"""
Voice Agent Event Types

Python implementation of the voice agent event system.
All events in the pipeline share common properties to enable
consistent handling, logging, and debugging across the system.

This module defines typed dataclasses for all events that flow through
the voice agent pipeline, from user audio input through STT, agent
processing, and TTS output.
"""

import base64
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union


def now_ms() -> int:
    """Return current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


class Role(Enum):
    TEACHER = 1
    AGENT_STUDENT = 2
    HUMAN_STUDENT = 3


@dataclass
class SocketTurnDecisionEvent:
    """
    Event emitted when the Turn Taking Controller has taken a decision
    about who should speak next.

    Sent from TTC to both agents after analyzing (one of the following):
    - User speech transcription (local STT)
    - Agent speech transcription (received from server)
    """

    type: Literal["turn_decision"]

    role: Role
    """
    Role of the party who should take the turn.
    """

    # text_role: Role
    # """
    # Role of the party who that led to the decision.
    # """

    text: str
    """
    Text of the conversation that led to the decision.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(cls, text: str, role: Role) -> "SocketTurnDecisionEvent":
        """Factory method to create a TurnDecisionEvent with current timestamp."""
        return cls(type="turn_decision", text=text, role=role, ts=now_ms())


@dataclass
class SocketTurnCancelledEvent:
    """
    Event emitted when the TTC cancels the turn of an agent.
    Human student's turn cannot be cancelled.

    Sent from TTC to one of the agents when the controller decides to interrupt
    an agent mid-turn (e.g., when the human starts speaking).
    """

    type: Literal["turn_cancelled"]

    role: Literal[Role.AGENT_STUDENT, Role.TEACHER]
    """
    Role of the agent whose turn has been cancelled.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(
        cls, role: Literal[Role.AGENT_STUDENT, Role.TEACHER]
    ) -> "SocketTurnCancelledEvent":
        """Factory method to create a TurnCancelledEvent with current timestamp."""
        return cls(type="turn_cancelled", role=role, ts=now_ms())


@dataclass
class SocketAgentAudioChunkEvent:
    """
    Event emitted when an agent is speaking.

    Sent from server to client. Contains audio data that the client
    should play through the speakers so the user can hear the agent.
    The audio is streamed in chunks for low-latency playback.
    """

    type: Literal["agent_audio_chunk"]

    audio: bytes
    """
    Raw PCM audio bytes from the agent's TTS output.
    Format: float32, mono channel, 24kHz sample rate.
    Encoded as base64 when serialized to JSON.
    """

    role: Literal[Role.AGENT_STUDENT, Role.TEACHER]
    """
    Role of the agent who is speaking.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(
        cls, audio: bytes, role: Literal[Role.AGENT_STUDENT, Role.TEACHER]
    ) -> "SocketAgentAudioChunkEvent":
        """Factory method to create an AgentAudioChunkEvent with current timestamp."""
        return cls(type="agent_audio_chunk", audio=audio, role=role, ts=now_ms())


@dataclass
class SocketAgentTextEvent:
    """
    Event emitted when an agent has finished speaking.

    Sent from server to client. Contains the full transcription of what
    the agent said. This event triggers the turn-taking controller to
    decide who should speak next.
    """

    type: Literal["agent_text"]

    text: str
    """
    Complete transcription of the agent's speech.
    """

    role: Literal[Role.AGENT_STUDENT, Role.TEACHER]
    """
    Role of the agent who was speaking.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(
        cls, text: str, role: Literal[Role.AGENT_STUDENT, Role.TEACHER]
    ) -> "SocketAgentTextEvent":
        """Factory method to create an AgentTextEvent with current timestamp."""
        return cls(type="agent_text", text=text, role=role, ts=now_ms())


# Union types for type-safe event handling

SocketClientEvent = SocketTurnDecisionEvent | SocketTurnCancelledEvent
"""Events sent from the turn-taking controller (client) to the server."""

SocketServerEvent = SocketAgentAudioChunkEvent | SocketAgentTextEvent
"""Events sent from the server to the turn-taking controller (client)."""


def event_to_dict(event: SocketClientEvent | SocketServerEvent) -> dict:
    """Convert an event to a JSON-serializable dictionary."""
    if isinstance(event, SocketTurnDecisionEvent):
        return {
            "type": event.type,
            "role": event.role.value,
            "text": event.text,
            "ts": event.ts,
        }
    elif isinstance(event, SocketTurnCancelledEvent):
        return {
            "type": event.type,
            "role": event.role.value,
            "ts": event.ts,
        }
    elif isinstance(event, SocketAgentAudioChunkEvent):
        return {
            "type": event.type,
            "audio": base64.b64encode(event.audio).decode("ascii"),
            "role": event.role.value,
            "ts": event.ts,
        }
    elif isinstance(event, SocketAgentTextEvent):
        return {
            "type": event.type,
            "text": event.text,
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

    if event_type == "turn_decision":
        role_value = data.get("role", Role.HUMAN_STUDENT.value)
        role = Role(role_value) if isinstance(role_value, int) else Role[role_value]
        return SocketTurnDecisionEvent(
            type="turn_decision",
            text=data.get("text", ""),
            role=role,
            ts=data.get("ts", now_ms()),
        )

    elif event_type == "turn_cancelled":
        role_value = data.get("role", Role.TEACHER.value)
        role = Role(role_value) if isinstance(role_value, int) else Role[role_value]
        return SocketTurnCancelledEvent(
            type="turn_cancelled",
            role=role,
            ts=data.get("ts", now_ms()),
        )

    elif event_type == "agent_audio_chunk":
        audio_str = data.get("audio", "")
        audio_bytes = base64.b64decode(audio_str) if audio_str else b""
        role_value = data.get("role", Role.TEACHER.value)
        role = Role(role_value) if isinstance(role_value, int) else Role[role_value]
        return SocketAgentAudioChunkEvent(
            type="agent_audio_chunk",
            audio=audio_bytes,
            role=role,
            ts=data.get("ts", now_ms()),
        )

    elif event_type == "agent_text":
        role_value = data.get("role", Role.TEACHER.value)
        role = Role(role_value) if isinstance(role_value, int) else Role[role_value]
        return SocketAgentTextEvent(
            type="agent_text",
            text=data.get("text", ""),
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
