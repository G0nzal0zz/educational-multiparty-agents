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


# TURN TAKING CONTROLLER EVENTS (client -> server)


@dataclass
class TurnDecisionEvent:
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
    def create(cls, text: str, role: Role) -> "TurnDecisionEvent":
        """Factory method to create a TurnDecisionEvent with current timestamp."""
        return cls(type="turn_decision", text=text, role=role, ts=now_ms())


@dataclass
class TurnCancelledEvent:
    """
    Event emitted when the TTC cancels the turn of an agent.
    Human student's turn cannot be cancelled.

    Sent from TTC to one of the agents when the controller decides to interrupt
    an agent mid-turn (e.g., when the human starts speaking).
    """

    type: Literal["turn_cancelled"]

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(
        cls, role: Literal[Role.AGENT_STUDENT, Role.TEACHER]
    ) -> "TurnCancelledEvent":
        """Factory method to create a TurnCancelledEvent with current timestamp."""
        return cls(type="turn_cancelled", role=role, ts=now_ms())


# AGENT EVENTS (server -> client)


@dataclass
class AgentAudioChunkEvent:
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
    ) -> "AgentAudioChunkEvent":
        """Factory method to create an AgentAudioChunkEvent with current timestamp."""
        return cls(type="agent_audio_chunk", audio=audio, role=role, ts=now_ms())


@dataclass
class AgentTextEvent:
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
    ) -> "AgentTextEvent":
        """Factory method to create an AgentTextEvent with current timestamp."""
        return cls(type="agent_text", text=text, role=role, ts=now_ms())


# Union types for type-safe event handling

ClientEvent = TurnDecisionEvent | TurnCancelledEvent
"""Events sent from the turn-taking controller (client) to the server."""

ServerEvent = AgentAudioChunkEvent | AgentTextEvent
"""Events sent from the server to the turn-taking controller (client)."""


def event_to_dict(event: ClientEvent | ServerEvent) -> dict:
    """Convert an event to a JSON-serializable dictionary."""
    if isinstance(event, TurnDecisionEvent):
        return {
            "type": event.type,
            "role": event.role.value,
            "text": event.text,
            "ts": event.ts,
        }
    elif isinstance(event, TurnCancelledEvent):
        return {
            "type": event.type,
            "ts": event.ts,
        }
    elif isinstance(event, AgentAudioChunkEvent):
        return {
            "type": event.type,
            "audio": base64.b64encode(event.audio).decode("ascii"),
            "role": event.role.value,
            "ts": event.ts,
        }
    elif isinstance(event, AgentTextEvent):
        return {
            "type": event.type,
            "text": event.text,
            "role": event.role.value,
            "ts": event.ts,
        }
    else:
        raise ValueError(f"Unknown event type: {type(event)}")
