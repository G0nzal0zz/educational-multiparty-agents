import asyncio
from collections.abc import AsyncIterator

from client_lib.config import config
from client_lib.events import AgentChunkEvent, AgentEndEvent, AgentEvent, ServerEvent
from client_lib.handler import ClientHandler
from client_lib.ollama_llm import OLlamaLLM
from client_lib.prompts import TTS_SYSTEM_PROMPT
from langchain_core.runnables import RunnableGenerator
from shared_lib.events import (
    Role,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketAgentTurnCancelledEvent,
    SocketAgentTurnEvent,
    SocketEvent,
    SocketHumanTranscription,
    SocketServerEvent,
)

system_prompt = f"""
You are a patient teacher conducting a lesson about the Spanish Empire. \
Your students include a human student (with speech transcription that may have errors) \
and an AI student agent (with clear, grammatically correct questions).

## Message Types You Will Receive

You will receive three types of messages, each prefixed to identify the source:

1. **Teaching Initiation** - `Start teaching:`: When the lesson begins, you will receive this prefix followed by a request to teach about a topic. Begin with a brief, engaging introduction.

2. **Human Interaction** - `The human student said:`: Real-time speech transcription that may contain errors (homophones, misheard words, typos). Interpret the intent and ask for clarification if needed.

3. **Agent Question** - `The student agent said:`: Clear, grammatically correct question from the AI student.

## Response Guidelines

- For teaching initiation: Give a brief, engaging introduction (1-2 sentences).
- For human input: Be patient with unclear phrasing, ask for clarification if needed.
- For agent questions: Answer directly and concisely.

## Plain Text Output

Your response will be converted to speech and displayed as text. Always output:
- Plain, readable text (no markdown, no emojis, no special characters)
- Natural spoken language
- Proper punctuation

{TTS_SYSTEM_PROMPT}
""".strip()

agent = OLlamaLLM(system_prompt)


async def _ollama_agent_stream(
    event_stream: AsyncIterator[SocketEvent],
) -> AsyncIterator[ServerEvent]:
    """Process incoming client events and generate LLM responses.

    Reads client events via a background task so that a SocketTurnCancelledEvent
    can interrupt an in-progress LLM generation between chunks.
    """

    pending: asyncio.Queue[SocketEvent | None] = asyncio.Queue()
    output: asyncio.Queue[AgentEvent] = asyncio.Queue()
    cancelled = asyncio.Event()

    async def _feed_events():
        """Continuously read client events into a queue, flagging cancellations."""
        async for event in event_stream:
            if isinstance(event, SocketAgentTurnCancelledEvent):
                print("Cancelling event")
                cancelled.set()
            else:
                await pending.put(event)
        await pending.put(None)

    feeder = asyncio.create_task(_feed_events())

    first_turn = True

    async def generate_output(message: str):
        try:
            async for chunk in agent.generate_response(message):
                await output.put(chunk)
        except asyncio.CancelledError:
            print("LLM generation cancelled")
            raise

    current_task = None

    try:
        while True:
            event = await pending.get()
            if event is None:
                break

            if isinstance(event, SocketAgentTurnEvent):
                if first_turn:
                    current_task = asyncio.create_task(
                        generate_output(
                            "Start teaching: Teach something about the Spanish Empire. Keep it very concise and engaging."
                        )
                    )
                    first_turn = False

                while True:
                    agent_event = await output.get()

                    yield agent_event

                    if isinstance(agent_event, AgentEndEvent):
                        print("Agent has finished generating text")
                        current_task = None
                        break

            elif isinstance(event, SocketHumanTranscription):
                current_task = asyncio.create_task(
                    generate_output(f"The human student said: {event.text}")
                )

            elif isinstance(event, SocketAgentTextEndEvent):
                print(f"event.text: {event.text}")
                current_task = asyncio.create_task(
                    generate_output(f"The student agent said: {event.text}")
                )

            else:
                print(f"WARNING: Received unexpected event type: {type(event)}")

    finally:
        if current_task:
            _ = current_task.cancel()

        _ = feeder.cancel()
        try:
            await feeder
        except asyncio.CancelledError:
            pass


async def _to_socket_events(
    event_stream: AsyncIterator[ServerEvent],
) -> AsyncIterator[SocketServerEvent]:
    """Convert internal pipeline events to socket events for transmission to the client."""
    async for event in event_stream:
        if isinstance(event, AgentChunkEvent):
            yield SocketAgentTextChunkEvent.create(text=event.text, role=Role.TEACHER)
        elif isinstance(event, AgentEndEvent):
            print("SENDING SocketAgentTextEndEvent")

            yield SocketAgentTextEndEvent.create(role=Role.TEACHER, text=event.text)
        else:
            print(f"WARNING: Unknown event type in pipeline: {type(event)}")


pipeline = RunnableGenerator(_ollama_agent_stream) | RunnableGenerator(
    _to_socket_events
)


async def start_client():
    client_handler = ClientHandler(pipeline)

    reader, writer = await asyncio.open_connection(
        config.TTC_SERVER_HOST, config.TTC_SERVER_PORT
    )

    print("Teacher client connected to TTC")
    await client_handler.handle(reader, writer)


if __name__ == "__main__":
    asyncio.run(start_client())
