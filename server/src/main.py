import asyncio
from collections.abc import AsyncIterator
from typing import Literal

import numpy as np
from langchain_core.runnables import RunnableGenerator

from src.lib.chatterbox_tts import ChatterboxTTS
from src.lib.events import AgentChunkEvent, AgentEndEvent, VoiceAgentEvent
from src.lib.ollama_llm import OllamaLLM
from src.lib.prompts import TEACHER_PROMPT
from src.lib.utils import ClientHandler
from src.lib.whisper_stt import WhisperSTT

role: Literal["TEACHER", "STUDENT"] = "TEACHER"
state: Literal["TEACHING", "LISTENING"] = "TEACHING"

SAMPLE_RATE = 16000
MAX_SECONDS = 5
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS


async def _whisper_stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Continuously collect raw PCM audio and transcribe with Whisper.
    Assumes:
        - mono
        - 16-bit PCM
        - 16 kHz
    """

    if role == "TEACHER" and state == "TEACHING":
        return

    pcm_buffer = bytearray()
    stt = WhisperSTT(sample_rate=16000)

    async for chunk in audio_stream:
        if not chunk:
            continue

        pcm_buffer.extend(chunk)

        if len(pcm_buffer) >= MAX_SAMPLES * 2:  # 2 bytes per int16
            # Copy buffer to NumPy safely
            audio_int16 = np.frombuffer(bytes(pcm_buffer), dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            async for event in stt.transcribe(audio_float32):
                yield event

            # Reset buffer for next segment
            pcm_buffer.clear()


async def _ollama_agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    ollamaLLM = OllamaLLM()

    if role == "TEACHER" and state == "TEACHING":
        global role, state
        _ = ollamaLLM.generate_response(TEACHER_PROMPT)
        role = "STUDENT"
        state = "LISTENING"

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


pipeline = (
    RunnableGenerator(_whisper_stt_stream)  # Audio -> STT events
    | RunnableGenerator(_ollama_agent_stream)  # STT events -> STT + Agent events
    | RunnableGenerator(_chatterbox_tts_stream)  # STT + Agent events -> All events
)


async def start_server():
    clientHandler = ClientHandler(pipeline)
    server = await asyncio.start_server(clientHandler.handle, "127.0.0.1", 9000)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(start_server())
