import asyncio
from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableGenerator
from server_lib.chatterbox_tts import ChatterboxTTS
from server_lib.events import AgentChunkEvent, AgentEndEvent
from server_lib.handler import ClientHandler, ServerEvent
from server_lib.ollama_llm import OLlamaLLM
from server_lib.prompts import TTS_SYSTEM_PROMPT
from shared_lib.events import Role, SocketClientEvent, SocketTurnCancelledEvent

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

    async for event in event_stream:
        if isinstance(event, SocketTurnCancelledEvent):
            # TODO: Handle cancelled turn
            continue

        if event.role != Role.TEACHER:
            print("ERROR: Received turn event, but role isn't TEACHER")
            continue

        response = agent.generate_response(event.text)
        async for chunk in response:
            yield chunk


async def _chatterbox_tts_stream(
    event_stream: AsyncIterator[ServerEvent],
) -> AsyncIterator[ServerEvent]:
    tts = ChatterboxTTS()

    buffer: list[str] = []
    async for event in event_stream:
        yield event

        if isinstance(event, AgentChunkEvent):
            buffer.append(event.text)

        if isinstance(event, AgentEndEvent):
            print(" ".join(buffer))
            ttsEvent = await tts.generate(" ".join(buffer))
            yield ttsEvent
            buffer = []


pipeline = RunnableGenerator(_ollama_agent_stream) | RunnableGenerator(
    _chatterbox_tts_stream
)


async def server():
    clientHandler = ClientHandler(pipeline)
    server = await asyncio.start_server(clientHandler.handle, "127.0.0.1", 9000)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(server())
