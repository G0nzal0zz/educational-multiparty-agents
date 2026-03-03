import asyncio
from collections.abc import AsyncIterator
from typing import Literal

from langchain_core.runnables import RunnableGenerator
from server_lib.chatterbox_tts import ChatterboxTTS
from server_lib.events import AgentChunkEvent, AgentEndEvent, VoiceAgentEvent
from server_lib.handler import ClientHandler
from server_lib.ollama_llm import OllamaLLM
from server_lib.prompts import TEACHER_PROMPT

role: Literal["TEACHER", "STUDENT"] = "TEACHER"
state: Literal["TEACHING", "LISTENING"] = "TEACHING"


async def _ollama_agent_stream(
    event_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    ollamaLLM = OllamaLLM()

    # if role == "TEACHER" and state == "TEACHING":
    #     _ = ollamaLLM.generate_response(TEACHER_PROMPT)
    #     role = "STUDENT"
    #     state = "LISTENING"

    async for event in event_stream:
        # Pass through all events to downstream consumers
        yield event

        # When we receive a final transcript, invoke the agent
        if event.type == "stt_output":
            _ = ollamaLLM.generate_response(event.transcript)


async def _chatterbox_tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    tts = ChatterboxTTS()

    buffer: list[str] = []
    async for event in event_stream:
        # Pass through all events to downstream consumers
        yield event
        # Buffer agent text chunks
        if event.type == "agent_chunk":
            buffer.append(event.text)
        # Send all buffered text to Chatterbox TTS finishes
        if event.type == "agent_end":
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
