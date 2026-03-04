"""
Turn Taking Controller Agent - Decision logic using LLM.

This module contains the client-side agent that processes events
from both the local user (microphone) and remote agents (server),
and makes decisions about who should take the turn to speak.
"""

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from shared_lib.events import (
    Role,
    SocketAgentTextEvent,
    SocketClientEvent,
    SocketTurnDecisionEvent,
)

SYSTEM_PROMPT = """You are a turn-taking controller for an educational dialogue system.
Your role is to decide who should speak next in a conversation between:
- HUMAN_STUDENT: The human user/student
- TEACHER: The AI teacher agent
- AGENT_STUDENT: Another AI student agent

You must analyze the conversation context and decide who should speak next.
Consider:
1. Who was the last speaker?
2. Was the last speech complete (ended with a question or natural pause)?
3. Does someone need to respond to a question?
4. Is there a natural transition to another speaker?
"""


def _parse_role(role_str: str) -> Role:
    """Convert string role to Role enum, defaulting to HUMAN_STUDENT."""
    role_str = role_str.strip().upper()
    if "HUMAN" in role_str:
        return Role.HUMAN_STUDENT
    elif "TEACHER" in role_str:
        return Role.TEACHER
    elif "AGENT" in role_str or "STUDENT" in role_str:
        return Role.AGENT_STUDENT
    else:
        print(f"Unknown role: {role_str}, defaulting to HUMAN_STUDENT")
        return Role.HUMAN_STUDENT


class _model_output(BaseModel):
    """
    Structured output from the LLM specifying who should speak next and why.

    This output is used by the turn-taking controller to decide which party
    should take the next turn in the conversation.
    """

    role: str = Field(
        ...,
        description="""The party who should speak next in the conversation.
        Valid values:
        - HUMAN_STUDENT: The human user/student should speak next (e.g., after teacher explains something)
        - TEACHER: The AI teacher agent should speak next (e.g., to answer a question)
        - AGENT_STUDENT: The AI student agent should speak next (e.g., to respond to the human)""",
    )
    reason: str = Field(
        ...,
        description="""A brief explanation (1-2 sentences) for why this role was chosen.
        This helps track the decision-making process.""",
    )
    confidence: float = Field(
        default=0.5,
        description="""Confidence score from 0.0 to 1.0 indicating how certain
        you are about this decision. Higher values mean more confident.""",
    )


class OllamaLLM:
    """
    Client-side agent that processes events and makes turn-taking decisions using LLM.

    The agent receives:
    - STTEvent: When the user speaks (transcribed locally)
    - AgentTextEvent: When an agent finishes speaking (from server)

    The agent decides:
    - TurnDecisionEvent: Who should speak next (sent to server)
    """

    model: Runnable[LanguageModelInput, dict | BaseModel]

    def __init__(self):
        ollama = ChatOllama(
            model="llama3.2:1b",
            temperature=0.3,
        )
        self.model = ollama.with_structured_output(_model_output)

    async def process_event(
        self, event: "STTEvent | SocketAgentTextEvent"
    ) -> SocketClientEvent | None:
        """
        Process an event and return a turn-taking decision.

        Args:
            event: Either STTEvent (user spoke) or AgentTextEvent (agent finished)

        Returns:
            TurnDecisionEvent if a decision is made, None otherwise
        """
        if isinstance(event, STTEvent):
            return await self._process_user_speech(event)
        elif isinstance(event, SocketAgentTextEvent):
            return await self._process_agent_text(event)
        return None

    async def _process_user_speech(self, event: "STTEvent") -> SocketTurnDecisionEvent:
        """
        Process user speech transcription.

        When the user speaks, analyze their speech to decide who should respond.
        The user might be:
        - Asking a question (teacher should answer)
        - Making a statement (could be responded by either agent)
        - Responding to an agent (agent should continue or human takes turn)
        """
        print("Processing user speech with transcript:", event.transcript)

        prompt = f"""{SYSTEM_PROMPT}

User said: "{event.transcript}"

Based on the user's speech, decide who should speak next. Consider:
- If the user asked a question, the TEACHER should likely respond
- If the user made a statement, consider if AGENT_STUDENT should respond or if the human is done speaking"""

        try:
            result: _model_output = self.model.invoke(prompt)
            parsed_role = _parse_role(result.role)
            print(
                f"LLM Decision: role={parsed_role.name}, reason={result.reason}, confidence={result.confidence}"
            )
        except Exception as e:
            print(f"LLM error: {e}, defaulting to HUMAN_STUDENT")
            parsed_role = Role.HUMAN_STUDENT
            result = _model_output(
                role="HUMAN_STUDENT",
                reason="Default: human student takes turn after speaking",
                confidence=0.5,
            )

        return SocketTurnDecisionEvent.create(
            text=event.transcript,
            role=parsed_role,
        )

    async def _process_agent_text(
        self, event: SocketAgentTextEvent
    ) -> SocketTurnDecisionEvent:
        """
        Process agent text transcription.

        After an agent finishes speaking, decide who should speak next.
        Consider:
        - Did the agent ask a question? (likely human should respond)
        - Did the agent make a statement? (could be responded by another agent)
        - Is this a teaching moment? (could go to student agent)
        """
        agent_type = "Teacher" if event.role == Role.TEACHER else "Student Agent"
        print(f"Processing {agent_type} speech with transcript: {event.text}")

        prompt = f"""{SYSTEM_PROMPT}

{agent_type} said: "{event.text}"

Based on the agent's speech, decide who should speak next. Consider:
- If the agent asked a question, the HUMAN_STUDENT should likely respond
- If the agent made a statement and is waiting, AGENT_STUDENT might elaborate
- If this is educational content, consider if another agent should add perspective"""

        try:
            result: _model_output = self.model.invoke(prompt)
            parsed_role = _parse_role(result.role)
            print(
                f"LLM Decision: role={parsed_role.name}, reason={result.reason}, confidence={result.confidence}"
            )
        except Exception as e:
            print(f"LLM error: {e}, defaulting to HUMAN_STUDENT")
            parsed_role = Role.HUMAN_STUDENT
            result = _model_output(
                role="HUMAN_STUDENT",
                reason="Default: human student takes turn after agent speaks",
                confidence=0.5,
            )

        return SocketTurnDecisionEvent.create(
            text=event.text,
            role=parsed_role,
        )


# Import STTEvent from local events module
from events import STTEvent
