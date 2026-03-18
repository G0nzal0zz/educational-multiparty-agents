import argparse
import asyncio
import queue
from sys import platform

import speech_recognition as sr
from shared_lib.events import (
    Role,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketClientEvent,
    SocketServerEvent,
)
from shared_lib.stream import read_event

from config import config
from event_handlers import (
    AgentTextChunkHandler,
    AgentTextEndHandler,
    STTEndEventHandler,
    TTSEndEventHandler,
)
from events import STTEndEvent, STTEvent, TTSEvent
from turn_manager import TurnManager
from whisper_stt import WhisperSTT

HOST = "127.0.0.1"
PORT = 9000


async def process_events(
    stt_queue: asyncio.Queue[STTEvent],
    tts_queue: asyncio.Queue[TTSEvent],
    text_queue: queue.Queue[str | None],
    server_queue: asyncio.Queue[SocketServerEvent],
    server_writer: dict[Role, asyncio.StreamWriter],
):
    turn_manager = TurnManager()

    stt_handler = STTEndEventHandler(server_writer, turn_manager)
    tts_handler = TTSEndEventHandler(turn_manager, tts_queue)

    chunk_handler = AgentTextChunkHandler(turn_manager, tts_queue, text_queue)
    end_handler = AgentTextEndHandler(server_writer, turn_manager, text_queue)

    while True:
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(stt_queue.get()),
                asyncio.create_task(tts_queue.get()),
                asyncio.create_task(server_queue.get()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            try:
                event = task.result()
            except asyncio.CancelledError:
                # TODO: Log exception
                continue

            if isinstance(event, STTEndEvent):
                stt_handler.handle(event)
            elif isinstance(event, TTSEvent):
                await tts_handler.handle(event)
            elif isinstance(event, SocketAgentTextChunkEvent):
                chunk_handler.handle(event)
            elif isinstance(event, SocketAgentTextEndEvent):
                end_handler.handle(event)

        for task in pending:
            _ = task.cancel()


async def user_listener(whisper: WhisperSTT, queue: asyncio.Queue[STTEvent]):
    async for event in whisper.STT():
        await queue.put(event)


async def server_listener(
    reader: asyncio.StreamReader, queue: asyncio.Queue[SocketServerEvent]
):
    async for event in read_event(reader):
        if isinstance(event, SocketClientEvent):
            print("Received SocketClientEvent in the client, skipping for now.")
            continue
        await queue.put(event)


async def start(whisper: WhisperSTT):
    stt_queue: asyncio.Queue[STTEvent] = asyncio.Queue()
    tts_queue: asyncio.Queue[TTSEvent] = asyncio.Queue()
    text_queue: queue.Queue[str | None] = queue.Queue()
    server_queue: asyncio.Queue[SocketServerEvent] = asyncio.Queue()
    server_writers: dict[Role, asyncio.StreamWriter] = {}

    user_task = asyncio.create_task(user_listener(whisper, stt_queue))
    process_events_task = asyncio.create_task(
        process_events(
            stt_queue,
            tts_queue,
            text_queue,
            server_queue,
            server_writers,
        )
    )

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        if len(server_writers) == 0:
            role = Role.TEACHER
        else:
            role = Role.STUDENT

        server_writers[role] = writer
        server_task = asyncio.create_task(server_listener(reader, server_queue))

        try:
            _ = await server_task
        except asyncio.CancelledError:
            _ = server_task.cancel()
        finally:
            server_writers.pop(role, None)

    server = await asyncio.start_server(handle, config.HOST, config.PORT)
    print(f"TTC server listening on {config.HOST}:{config.PORT}")

    try:
        _ = await asyncio.gather(user_task, process_events_task, server.serve_forever())
    except asyncio.CancelledError:
        _ = user_task.cancel()
        _ = process_events_task.cancel()


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
    _ = parser.add_argument(
        "--intervention_timeout",
        default=5,
        help="Maximum time (seconds) to wait for user intervention",
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
    llama = None

    await start(whisper)


if __name__ == "__main__":
    asyncio.run(main())
