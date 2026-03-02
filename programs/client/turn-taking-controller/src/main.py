import argparse
import asyncio
import socket
from sys import platform

import numpy as np
import sounddevice as sd
import speech_recognition as sr

from whisper_stt import WhisperSTT

HOST = "127.0.0.1"
PORT = 9000

async def send_audio(whisper: WhisperSTT, sock: socket.socket):
    whisper.audio_model
    loop = asyncio.get_running_loop()
    async for data in whisper.STT():
        await loop.sock_sendall(sock, data)


async def receive_and_play(sock):
    loop = asyncio.get_running_loop()

    # Match the server audio: float32, mono
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype="float32")
    stream.start()

    try:
        while True:
            data = await loop.sock_recv(sock, 4096)
            if not data:
                break

            # Interpret the bytes as float32, not int16
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            stream.write(audio_chunk)

    except ConnectionResetError:
        # Server closed abruptly → normal end of audio
        pass

    finally:
        stream.stop()
        stream.close()


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
                    args.source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        args.source = sr.Microphone(sample_rate=16000)

    loop = asyncio.get_running_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    await loop.sock_connect(sock, (HOST, PORT))

    send_task = asyncio.create_task(send_audio(sock))
    recv_task = asyncio.create_task(receive_and_play(sock))

    try:
        await asyncio.gather(send_task, recv_task)
    finally:
        send_task.cancel()
        recv_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
