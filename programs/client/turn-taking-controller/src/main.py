import argparse
import asyncio
from sys import platform

from shared_lib.events import (
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketClientEvent,
    SocketServerEvent,
)
from shared_lib.stream import read_event

from config import config
from event_handlers import AgentTextChunkHandler, AgentTextEndHandler, STTEventHandler
from events import STTEndEvent, STTEvent
from turn_manager import TurnManager
from whisper_stt import WhisperSTT

HOST = "127.0.0.1"
PORT = 9000


async def process_events(
    stt_queue: asyncio.Queue[STTEvent],
    teacher_queue: asyncio.Queue[SocketServerEvent],
    student_queue: asyncio.Queue[SocketServerEvent],
    teacher_writer: asyncio.StreamWriter,
    student_writer: asyncio.StreamWriter,
):
    turn_manager = TurnManager()

    stt_handler = STTEventHandler(teacher_writer, student_writer, turn_manager)
    chunk_handler = AgentTextChunkHandler(turn_manager)
    end_handler = AgentTextEndHandler(
        teacher_writer, student_writer, turn_manager, stt_queue, student_queue
    )

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
                # TODO: Log exception
                continue

            if isinstance(event, STTEndEvent):
                stt_handler.handle(event)
            elif isinstance(event, SocketAgentTextChunkEvent):
                chunk_handler.handle(event)
            elif isinstance(event, SocketAgentTextEndEvent):
                await end_handler.handle(event)

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
    teacher_queue: asyncio.Queue[SocketServerEvent] = asyncio.Queue()
    student_queue: asyncio.Queue[SocketServerEvent] = asyncio.Queue()

    # -------- SOCKET CONNECTIONS --------
    teacher_reader, teacher_writer = await asyncio.open_connection(
        config.TEACHER_SERVER_HOST, config.TEACHER_SERVER_PORT
    )
    student_reader, student_writer = await asyncio.open_connection(
        config.STUDENT_SERVER_HOST, config.STUDENT_SERVER_PORT
    )

    # -------- TASKS --------
    user_task = asyncio.create_task(user_listener(whisper, stt_queue))
    teacher_task = asyncio.create_task(server_listener(teacher_reader, teacher_queue))
    student_task = asyncio.create_task(server_listener(student_reader, student_queue))
    process_events_task = asyncio.create_task(
        process_events(
            stt_queue, teacher_queue, student_queue, teacher_writer, student_writer
        )
    )

    try:
        _ = await asyncio.gather(
            user_task,
            teacher_task,
            student_task,
            process_events_task,
        )
    except asyncio.CancelledError:
        _ = user_task.cancel()
        _ = teacher_task.cancel()
        _ = student_task.cancel()
        _ = process_events_task.cancel()


async def main():
    import speech_recognition as sr

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
    llama = None

    await start(whisper)


if __name__ == "__main__":
    asyncio.run(main())
