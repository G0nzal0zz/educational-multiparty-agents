"""
Llama Agent - Turn-taking decision logic.

This module contains the client-side agent that processes events
from both the local user (microphone) and remote agents (server),
and makes decisions about who should take the turn to speak.
"""

from shared_lib.events import AgentTextEvent, ClientEvent, Role, TurnDecisionEvent

from events import STTEvent


class LlamaAgent:
    """
    Client-side agent that processes events and makes turn-taking decisions.

    The agent receives:
    - UserSTTEvent: When the user speaks (transcribed locally)
    - AgentTextEvent: When an agent finishes speaking (from server)

    The agent decides:
    - TurnDecisionEvent: Who should speak next (sent to server)
    """

    def __init__(self):
        print("LlamaAgent initialized")

    async def process_event(self, event: STTEvent | AgentTextEvent) -> ClientEvent:
        """
        Process an event and return a turn-taking decision.

        Args:
            event: Either UserSTTEvent (user spoke) or AgentTextEvent (agent finished)

        Returns:
            TurnDecisionEvent if a decision is made, None otherwise
        """
        if isinstance(event, STTEvent):
            return await self._process_user_speech(event)
        elif isinstance(event, AgentTextEvent):
            return await self._process_agent_text(event)

    async def _process_user_speech(self, event: STTEvent) -> TurnDecisionEvent:
        """
        Process user speech transcription.

        When the user speaks, the turn is given to the human student.
        """
        return TurnDecisionEvent.create(
            text=event.transcript,
            role=Role.HUMAN_STUDENT,
        )

    async def _process_agent_text(self, event: AgentTextEvent) -> TurnDecisionEvent:
        """
        Process agent text transcription.

        After an agent finishes speaking, decide who should speak next.
        Currently gives the turn to the human student.
        """
        return TurnDecisionEvent.create(
            text=event.text,
            role=Role.HUMAN_STUDENT,
        )
