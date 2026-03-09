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
import sounddevice as sd
import speech_recognition as sr
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from shared_lib.events import (Role, SocketAgentTextChunkEvent,
                               SocketAgentTextEndEvent, SocketClientEvent,
                               SocketEvent, SocketServerEvent, event_to_dict)
from shared_lib.utils import stream_reader_to_event

from events import STTChunkEvent, STTEndEvent, STTEvent
from ollama_llm import OllamaLLM
from whisper_stt import WhisperSTT

HOST = "127.0.0.1"
PORT = 9000


async def send_event_to_server(sock: socket.socket, event: SocketClientEvent):
    """Send an event from the client to the server over the socket."""
    loop = asyncio.get_running_loop()
    json_data = json.dumps(event_to_dict(event)) + "\n"
    await loop.sock_sendall(sock, json_data.encode())


async def event_pipeline(whisper: WhisperSTT, llama: OllamaLLM, sock: socket.socket):
    """
    Main event processing pipeline.

    Concurrently listens to:
    - Local user speech (via Whisper STT)
    - Server events (agent audio and text)

    When events arrive, processes them through the OllamaAgent to
    make turn-taking decisions, and sends decisions back to the server.
    """

    user_task = asyncio.create_task(user_listener())
    server_task = asyncio.create_task(server_listener())
    process_task = asyncio.create_task(process_events())

    try:
        await asyncio.gather(user_task, server_task, process_task)
    except asyncio.CancelledError:
        user_task.cancel()
        server_task.cancel()
        process_task.cancel()


async def user_listener(whisper: WhisperSTT, queue: asyncio.Queue[STTEvent]):
    """Listen to local microphone and emit user speech events."""
    async for event in whisper.STT():
        await queue.put(event)


async def server_listener(
    reader: asyncio.StreamReader, queue: asyncio.Queue[SocketServerEvent]
):
    """Listen to local microphone and emit user speech events."""
    async for event in stream_reader_to_event(reader):
        if isinstance(event, SocketClientEvent):
            print("Received SocketClientEvent in the clinet, skipping for now.")
            continue
        await queue.put(event)


async def process_events(
    stt_queue: asyncio.Queue[STTEvent],
    teacher_queue: asyncio.Queue[SocketServerEvent],
    student_queue: asyncio.Queue[SocketServerEvent],
):
    """Process events from both sources and make turn decisions."""
    while True:
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(stt_queue.get()),
                asyncio.create_task(teacher_queue.get()),
                asyncio.create_task(student_queue.get()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            try:
                event = task.result()
            except asyncio.CancelledError:
                continue

            if isinstance(event, STTEndEvent):
                print("test")
                # TODO: send transcription to teacher and student
                # handle_STT_end(event)

            if isinstance(event, SocketAgentTextChunkEvent):
                print("test")
                # TODO: play audio through TTS
                # handle_agent_text_chunk(event)

            if isinstance(event, SocketAgentTextEndEvent):
                # TODO: send transcribed text to the other server
                if event.role == Role.TEACHER:
                    # Wait for the user to talk
                    try:
                        sttEvent = await asyncio.wait_for(stt_queue.get(), 1.0)
                        if isinstance(sttEvent, STTEndEvent):
                            print("test")
                            # handle_STT_end(sttEvent)
                        else:
                            continue

                    except asyncio.TimeoutError:

                    # TODO


            #
            # decision = await llama.process_event(event)
            #
            # if decision == None:
            #     break
            #
            # if isinstance(decision, SocketTurnDecisionEvent):
            #     await send_event_to_server(sock, decision)
            #     print(f"Sent turn decision: {decision.role.name}")

            # if isintance(decision, TurnCancelledEvent):
            #     TODO: Handle turn cancellation

        for task in pending:
            task.cancel()


async def start(whisper: WhisperSTT, llama: OllamaLLM):
    """Initialize socket connection and start the event pipeline."""
    stt_queue: asyncio.Queue[STTEvent] = asyncio.Queue()
    teacher_queue: asyncio.Queue[SocketServerEvent] = asyncio.Queue()
    student_queue: asyncio.Queue[SocketServerEvent] = asyncio.Queue()

    teacherReader, teacherWriter = await asyncio.open_connection("127.0.0.1", 8000)
    studentReader, studentWriter = await asyncio.open_connection("127.0.0.1", 8001)

    user_listener_task = asyncio.create_task(user_listener(whisper, stt_queue))
    teacher_listener_task = asyncio.create_task(
        server_listener(teacherReader, teacher_queue)
    )
    student_listener_task = asyncio.create_task(
        server_listener(studentReader, student_queue)
    )
    process_events_taks = asyncio.create_task(
        process_events(stt_queue, teacher_queue, student_queue)
    )

    try:
        _ = await asyncio.gather(
            user_listener_task,
            teacher_listener_task,
            student_listener_task,
            process_events_taks,
        )
    except asyncio.CancelledError:
        _ = user_listener_task.cancel()
        _ = teacher_listener_task.cancel()
        _ = student_listener_task.cancel()
        _ = process_events_taks.cancel()


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
    else:
        source = sr.Microphone(sample_rate=16000)

    whisper = WhisperSTT(args, source)
    llama = OllamaLLM()

    await start(whisper, llama)


if __name__ == "__main__":
    asyncio.run(main())
