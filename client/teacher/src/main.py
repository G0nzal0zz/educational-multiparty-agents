import asyncio
from collections.abc import AsyncIterator

from client_lib.config import config
from client_lib.events import AgentChunkEvent, AgentEndEvent, ServerEvent
from client_lib.handler import ClientHandler
from client_lib.ollama_llm import OLlamaLLM
from client_lib.prompts import TTS_SYSTEM_PROMPT
from langchain_core.runnables import RunnableGenerator
from shared_lib.events import (
    Role,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketClientEvent,
    SocketEvent,
    SocketHumanTranscription,
    SocketServerEvent,
    SocketTeacherStartEvent,
)

system_prompt = f"""
You are a helpful teacher. Your goal is to teach about a topic to some students.
Be concise and friendly.

Available topics: history, science, sports and culture.

{TTS_SYSTEM_PROMPT}
"""

agent = OLlamaLLM(system_prompt)


async def _ollama_agent_stream(
    event_stream: AsyncIterator[SocketEvent],
) -> AsyncIterator[ServerEvent]:
    """Process incoming client events and generate LLM responses.

    Reads client events via a background task so that a SocketTurnCancelledEvent
    can interrupt an in-progress LLM generation between chunks.
    """

    cancelled = asyncio.Event()
    pending: asyncio.Queue[SocketEvent | None] = asyncio.Queue()

    async def _feed_events():
        """Continuously read client events into a queue, flagging cancellations."""
        async for event in event_stream:
            # if isinstance(event, SocketTurnCancelledEvent):
            #     cancelled.set()
            # else:
            await pending.put(event)
        await pending.put(None)  # Sentinel: client stream ended

    feeder = asyncio.create_task(_feed_events())

    try:
        while True:
            event = await pending.get()
            if event is None:
                break

            # TODO: Change this
            if isinstance(event, SocketTeacherStartEvent):
                async for chunk in agent.generate_response(
                    "Teach me the Spanish Empire"
                ):
                    if cancelled.is_set():
                        print("Turn cancelled, aborting generation")
                        break
                    print(f"Sending AgentChunkEvent {chunk}")
                    yield chunk
                yield AgentEndEvent.create()

            if not isinstance(event, SocketHumanTranscription):
                print(f"WARNING: Received unexpected event type: {type(event)}")
                continue

            cancelled.clear()
            async for chunk in agent.generate_response(event.text):
                if cancelled.is_set():
                    print("Turn cancelled, aborting generation")
                    break
                print(f"Sending AgentChunkEvent {chunk}")
                yield chunk

            # Always emit AgentEndEvent so downstream stages flush their buffers
            yield AgentEndEvent.create()
    finally:
        feeder.cancel()
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
            yield SocketAgentTextEndEvent.create(role=Role.TEACHER)
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
