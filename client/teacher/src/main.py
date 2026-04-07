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
from shared_lib.utils import empty_queue

from lesson import lesson_generator
from prompt import EXPERIMENT_ONE_SYSTEM_PROMPT

agent = OLlamaLLM(EXPERIMENT_ONE_SYSTEM_PROMPT)


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

    async def generate_initial_output():
        for chunk in lesson_generator():
            await output.put(chunk)

    current_task = None

    try:
        while True:
            event = await pending.get()
            if event is None:
                break

            if isinstance(event, SocketAgentTurnEvent):
                if first_turn:
                    print("[INFO] Generating first lesson.")
                    await generate_initial_output()
                    first_turn = False

                while True:
                    agent_event = await output.get()

                    yield agent_event

                    if isinstance(agent_event, AgentEndEvent):
                        print(
                            "[INFO] Finished generating text, sending it to the TURN-TAKING-CONTROLLER."
                        )
                        print(f"[INFO] TEACHER's intervention: {agent_event.text}")
                        current_task = None
                        break

            elif isinstance(event, SocketHumanTranscription):
                print("[INFO] Received HUMAN transcription. Generating a response...")
                empty_queue(output)
                current_task = asyncio.create_task(generate_output(event.text))

            elif isinstance(event, SocketAgentTextEndEvent):
                print(
                    "[INFO] Received AGENTIC STUDENT intervention. Generating a response..."
                )
                empty_queue(output)
                current_task = asyncio.create_task(generate_output(event.text))

            else:
                print(f"[WARNING] Received unexpected event type: {type(event)}")

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

    print("TEACHER client connected to TTC")
    await client_handler.handle(reader, writer)


if __name__ == "__main__":
    asyncio.run(start_client())
