import argparse
import asyncio
import json
import socket
from collections.abc import AsyncIterator
from sys import platform

import numpy as np
import shared_lib.events
import sounddevice as sd
import speech_recognition as sr
from shared_lib.events import TurnTakingEvent, event_to_dict

from llama_agent import LlamaAgent
from whisper_stt import WhisperSTT

HOST = "127.0.0.1"
PORT = 9000


async def stream_turn_events_to_server(sock: socket.socket, whisper: WhisperSTT, llama: LlamaAgent):
    async def generate_turn_events(
        whisper: WhisperSTT, llama: LlamaAgent
    ) -> AsyncIterator[TurnTakingEvent]:

        stt_events = whisper.STT()

        async for event in llama.process_turn(stt_events):
            yield event

    loop = asyncio.get_running_loop()
    async for data in generate_turn_events(whisper, llama):
        json_data = json.dumps(event_to_dict(data))
        await loop.sock_sendall(sock, json_data.encode())


async def receive_server_data(sock: socket.socket):
    loop = asyncio.get_running_loop()

    # Match the server audio: float32, mono
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype="float32")
    stream.start()

    try:
        while True:
            data = await loop.sock_recv(sock, 4096)
            if not data:
                break

            audio_chunk = np.frombuffer(data, dtype=np.float32)
            _ = stream.write(audio_chunk)

    except ConnectionResetError:
        pass

    finally:
        stream.stop()
        stream.close()

async def start(whisper: WhisperSTT, llama: LlamaAgent):
    loop = asyncio.get_running_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    await loop.sock_connect(sock, (HOST, PORT))

    send_task = asyncio.create_task(stream_turn_events_to_server(sock, whisper, llama))
    recv_task = asyncio.create_task(receive_server_data(sock))

    try:
        _ = await asyncio.gather(send_task, recv_task)
    finally:
        _ = send_task.cancel()
        _ = recv_task.cancel()

async def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--model",
        default="medium",
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    _ = parser.add_argument(
        "--non_english", action="store_true", help="Don't use the english model."
    )
    _ = parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect.",
        type=int,
    )
    _ = parser.add_argument(
        "--record_timeout",
        default=2,
        help="How real time the recording is in seconds.",
        type=float,
    )
    _ = parser.add_argument(
        "--phrase_timeout",
        default=3,
        help="How much empty space between recordings before we "
        + "consider it a new line in the transcription.",
        type=float,
    )
    if "linux" in platform:
        _ = parser.add_argument(
            "--default_microphone",
            default="pulse",
            help="Default microphone name for SpeechRecognition. "
            + "Run this with 'list' to view available Microphones.",
            type=str,
        )
    args = parser.parse_args()

    # Set default microphone
    source = sr.Microphone(sample_rate=16000)

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if "linux" in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'Microphone with name "{name}" found')
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break

    whisper = WhisperSTT(args, source)
    llama = LlamaAgent()

    await start(whisper)


if __name__ == "__main__":
    asyncio.run(main())
