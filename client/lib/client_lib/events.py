"""
Voice Agent Event Types

Python implementation of the voice agent event system.
All events in the pipeline share common properties to enable
consistent handling, logging, and debugging across the system.

This module defines typed dataclasses for all events that flow through
the voice agent pipeline, from user audio input through STT, agent
processing, and TTS output.
"""

from dataclasses import dataclass
from typing import Literal

from shared_lib.events import now_ms


@dataclass
class AgentChunkEvent:
    """
    Event emitted during agent response generation for streaming text chunks.

    As the LLM generates its response, it streams tokens incrementally.
    These chunks enable real-time display of the agent's response and allow
    the TTS stage to begin synthesis before the complete response is generated,
    reducing overall latency.
    """

    type: Literal["agent_chunk"]

    text: str
    """
    Partial text chunk from the agent's streaming response.
    Multiple chunks combine to form the complete agent output.
    """

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(cls, text: str) -> "AgentChunkEvent":
        """Factory method to create an AgentChunkEvent event with current timestamp."""
        return cls(type="agent_chunk", text=text, ts=now_ms())


@dataclass
class AgentEndEvent:
    """
    Event emitted when the agent has finished generating its response for a turn.

    This signals downstream consumers (like TTS) that no more text is coming
    for this turn and they should flush any buffered content.
    """

    type: Literal["agent_end"]

    ts: int
    """Unix timestamp (milliseconds since epoch) when the event was created."""

    @classmethod
    def create(cls) -> "AgentEndEvent":
        """Factory method to create an AgentEndEvent event with current timestamp."""
        return cls(type="agent_end", ts=now_ms())


AgentEvent = AgentChunkEvent | AgentEndEvent
"""
Union type of all agent-related events.

This type encompasses all events emitted during agent processing, including
streaming text chunks, tool invocations, and completion signals. It enables
type-safe handling of the various stages of agent response generation.
"""


ServerEvent = AgentEvent
