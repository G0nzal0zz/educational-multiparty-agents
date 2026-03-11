import asyncio
from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableGenerator
from server_lib.chatterbox_tts import ChatterboxTTS
from server_lib.events import AgentChunkEvent, AgentEndEvent, ServerEvent
from server_lib.handler import ClientHandler
from server_lib.ollama_llm import OLlamaLLM
from server_lib.prompts import TTS_SYSTEM_PROMPT
from shared_lib.events import (
    Role,
    SocketAgentAudioChunkEvent,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketClientEvent,
    SocketHumanTranscription,
    SocketServerEvent,
    SocketTurnCancelledEvent,
)

system_prompt = f"""
You are a helpful teacher. Your goal is to teach about a topic to some students.
Be concise and friendly.

Available topics: history, science, sports and culture.

{TTS_SYSTEM_PROMPT}
"""

agent = OLlamaLLM(system_prompt)


async def _ollama_agent_stream(
    event_stream: AsyncIterator[SocketClientEvent],
) -> AsyncIterator[ServerEvent]:
    """Process incoming client events and generate LLM responses.

    Reads client events via a background task so that a SocketTurnCancelledEvent
    can interrupt an in-progress LLM generation between chunks.
    """

    cancelled = asyncio.Event()
    pending: asyncio.Queue[SocketClientEvent | None] = asyncio.Queue()

    async def _feed_events():
        """Continuously read client events into a queue, flagging cancellations."""
        async for event in event_stream:
            if isinstance(event, SocketTurnCancelledEvent):
                cancelled.set()
            else:
                await pending.put(event)
        await pending.put(None)  # Sentinel: client stream ended

    feeder = asyncio.create_task(_feed_events())

    try:
        while True:
            event = await pending.get()
            if event is None:
                break

            if not isinstance(event, SocketHumanTranscription):
                print(f"WARNING: Received unexpected event type: {type(event)}")
                continue

            cancelled.clear()
            async for chunk in agent.generate_response(event.text):
                if cancelled.is_set():
                    print("Turn cancelled, aborting generation")
                    break
                yield chunk

            # Always emit AgentEndEvent so downstream stages flush their buffers
            yield AgentEndEvent.create()
    finally:
        feeder.cancel()
        try:
            await feeder
        except asyncio.CancelledError:
            pass


async def _chatterbox_tts_stream(
    event_stream: AsyncIterator[ServerEvent],
) -> AsyncIterator[ServerEvent | SocketAgentAudioChunkEvent]:
    """Buffer agent text chunks and synthesize TTS audio when the agent finishes."""
    tts = ChatterboxTTS()

    buffer: list[str] = []
    async for event in event_stream:
        # Pass through all events (they'll be converted to socket events downstream)
        yield event

        if isinstance(event, AgentChunkEvent):
            buffer.append(event.text)

        if isinstance(event, AgentEndEvent):
            full_text = " ".join(buffer)
            print(f"TTS synthesizing: {full_text}")
            tts_event = await tts.generate(full_text)
            yield tts_event
            buffer = []


async def _to_socket_events(
    event_stream: AsyncIterator[ServerEvent | SocketAgentAudioChunkEvent],
) -> AsyncIterator[SocketServerEvent]:
    """Convert internal pipeline events to socket events for transmission to the client."""
    async for event in event_stream:
        if isinstance(event, AgentChunkEvent):
            yield SocketAgentTextChunkEvent.create(
                text=event.text, role=Role.TEACHER
            )
        elif isinstance(event, AgentEndEvent):
            yield SocketAgentTextEndEvent.create(role=Role.TEACHER)
        elif isinstance(event, SocketAgentAudioChunkEvent):
            yield event
        else:
            print(f"WARNING: Unknown event type in pipeline: {type(event)}")


pipeline = (
    RunnableGenerator(_ollama_agent_stream)
    | RunnableGenerator(_chatterbox_tts_stream)
    | RunnableGenerator(_to_socket_events)
)


async def server():
    client_handler = ClientHandler(pipeline)
    srv = await asyncio.start_server(client_handler.handle, "127.0.0.1", 9000)
    print("Teacher server listening on 127.0.0.1:9000")
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(server())
