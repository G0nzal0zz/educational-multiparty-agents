"""
Turn Taking Controller - Client-side event processing pipeline.

This module runs on the client and manages the turn-taking logic for a
voice-based educational system. It coordinates between:

1. Local user input (microphone) - transcribed via Whisper STT
2. Remote agent input (from server) - audio to play + transcription

The controller listens for events from both sources and decides who should
take the turn to speak. When a decision is made, it sends the decision
back to the server.
"""

import argparse
import asyncio
import base64
import json
import socket
from collections.abc import AsyncIterator
from sys import platform

import numpy as np
import shared_lib.events
import sounddevice as sd
import speech_recognition as sr
from shared_lib.events import (
    AgentAudioChunkEvent,
    AgentTextEvent,
    ClientEvent,
    Role,
    ServerEvent,
    TurnDecisionEvent,
    event_to_dict,
)

from events import STTEvent
from llama_agent import LlamaAgent
from whisper_stt import WhisperSTT

HOST = "127.0.0.1"
PORT = 9000


async def receive_server_events(
    sock: socket.socket,
) -> AsyncIterator[ServerEvent]:
    """
    Receive events from the server (agents).

    Parses newline-delimited JSON events from the server socket.
    Handles AgentAudioChunkEvent (audio to play) and AgentTextEvent
    (transcription when agent finishes speaking).
    """
    loop = asyncio.get_running_loop()
    buffer = ""

    stream = sd.OutputStream(samplerate=24000, channels=1, dtype="float32")
    stream.start()

    try:
        while True:
            data = await loop.sock_recv(sock, 4096)
            if not data:
                break

            buffer += data.decode("utf-8")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue

                try:
                    event_dict = json.loads(line)
                    event_type = event_dict.get("type")

                    if event_type == "agent_audio_chunk":
                        audio_data = event_dict.get("audio", "")
                        if audio_data:
                            audio_bytes = base64.b64decode(audio_data)
                            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                            stream.write(audio_np)
                            role_value = event_dict.get("role", Role.TEACHER.value)
                            role = Role(role_value)
                            yield AgentAudioChunkEvent.create(audio_bytes, role)

                    elif event_type == "agent_text":
                        text = event_dict.get("text", "")
                        role_value = event_dict.get("role", Role.TEACHER.value)
                        role = Role(role_value)
                        yield AgentTextEvent.create(text, role)

                except json.JSONDecodeError:
                    continue

    except ConnectionResetError:
        pass

    finally:
        stream.stop()
        stream.close()


async def send_event_to_server(sock: socket.socket, event: ClientEvent):
    """Send an event from the client to the server over the socket."""
    loop = asyncio.get_running_loop()
    json_data = json.dumps(event_to_dict(event)) + "\n"
    await loop.sock_sendall(sock, json_data.encode())


async def event_pipeline(whisper: WhisperSTT, llama: LlamaAgent, sock: socket.socket):
    """
    Main event processing pipeline.

    Concurrently listens to:
    - Local user speech (via Whisper STT)
    - Server events (agent audio and text)

    When events arrive, processes them through the LlamaAgent to
    make turn-taking decisions, and sends decisions back to the server.
    """
    stt_queue: asyncio.Queue[STTEvent] = asyncio.Queue()
    server_queue: asyncio.Queue[ServerEvent] = asyncio.Queue()

    async def user_listener():
        """Listen to local microphone and emit user speech events."""
        async for event in whisper.STT():
            await stt_queue.put(event)

    async def server_listener():
        """Listen to server for agent audio and text events."""
        async for event in receive_server_events(sock):
            await server_queue.put(event)

    async def process_events():
        """Process events from both sources and make turn decisions."""
        while True:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(stt_queue.get()),
                    asyncio.create_task(server_queue.get()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                try:
                    event = task.result()
                except asyncio.CancelledError:
                    continue

                decision = await llama.process_event(event)

                if decision:
                    await send_event_to_server(sock, decision)
                    print(
                        f"Sent turn decision: {decision.role.name} - '{decision.text}'"
                    )

            for task in pending:
                task.cancel()

    user_task = asyncio.create_task(user_listener())
    server_task = asyncio.create_task(server_listener())
    process_task = asyncio.create_task(process_events())

    try:
        await asyncio.gather(user_task, server_task, process_task)
    except asyncio.CancelledError:
        user_task.cancel()
        server_task.cancel()
        process_task.cancel()


async def start(whisper: WhisperSTT, llama: LlamaAgent):
    """Initialize socket connection and start the event pipeline."""
    loop = asyncio.get_running_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    await loop.sock_connect(sock, (HOST, PORT))

    await event_pipeline(whisper, llama, sock)


async def main():
    parser = argparse.ArgumentParser(description="Turn Taking Controller Client")
    _ = parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model to use for speech-to-text",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    _ = parser.add_argument(
        "--non_english",
        action="store_true",
        help="Don't use the English-only model",
    )
    _ = parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for microphone to detect speech",
        type=int,
    )
    _ = parser.add_argument(
        "--record_timeout",
        default=2,
        help="Maximum recording duration in seconds",
        type=float,
    )
    _ = parser.add_argument(
        "--phrase_timeout",
        default=3,
        help="Silence duration (seconds) to consider as end of phrase",
        type=float,
    )
    if "linux" in platform:
        _ = parser.add_argument(
            "--default_microphone",
            default="pulse",
            help="Default microphone name for SpeechRecognition. "
            + "Run with 'list' to view available microphones",
            type=str,
        )
    args = parser.parse_args()

    source = sr.Microphone(sample_rate=16000)

    if "linux" in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphone devices:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'  Microphone "{name}" found')
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break

    whisper = WhisperSTT(args, source)
    llama = LlamaAgent()

    await start(whisper, llama)


if __name__ == "__main__":
    asyncio.run(main())
