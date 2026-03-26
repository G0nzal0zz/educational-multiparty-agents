import argparse
import asyncio
import queue
from sys import platform

import speech_recognition as sr
from shared_lib.events import (
    Role,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketAgentTurnEvent,
    SocketClientEvent,
    SocketServerEvent,
)
from shared_lib.stream import read_event, write_event

from chatterbox_tts import ChatterboxTTS
from config import config
from event_handlers import (
    AgentTextChunkHandler,
    AgentTextEndHandler,
    EventContext,
    STTEventHandler,
    TTSEndEventHandler,
)
from events import STTChunkEvent, STTEndEvent, STTEvent, TTSEndEvent, TTSEvent
from turn_manager import TurnManager
from whisper_stt import WhisperSTT

HANDLERS = {
    STTChunkEvent: STTEventHandler(),
    STTEndEvent: STTEventHandler(),
    TTSEndEvent: TTSEndEventHandler(),
    SocketAgentTextChunkEvent: AgentTextChunkHandler(),
    SocketAgentTextEndEvent: AgentTextEndHandler(),
}


async def process_events(
    server_writer: dict[Role, asyncio.StreamWriter],
    server_event_queue: asyncio.Queue[SocketServerEvent],
    stt_event_queue: asyncio.Queue[STTEvent],
    tts_event_queue: asyncio.Queue[TTSEvent],
    agent_chunk_event_queue: queue.Queue[str | None],
):
    turn_manager = TurnManager()
    tts = ChatterboxTTS(tts_event_queue, agent_chunk_event_queue)
    context = EventContext(
        turn_manager=turn_manager,
        server_writers=server_writer,
        tts=tts,
        stt_event_queue=stt_event_queue,
        agents_chunk_event_queue=agent_chunk_event_queue,
    )

    while True:
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(stt_event_queue.get()),
                asyncio.create_task(tts_event_queue.get()),
                asyncio.create_task(server_event_queue.get()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            try:
                event = task.result()
            except asyncio.CancelledError:
                continue

            handler = HANDLERS.get(type(event))
            if handler:
                result = handler.handle(event, context)
                if asyncio.iscoroutine(result):
                    await result

        for task in pending:
            _ = task.cancel()


async def user_listener(whisper: WhisperSTT, queue: asyncio.Queue[STTEvent]):
    async for event in whisper.STT():
        await queue.put(event)


async def server_listener(
    reader: asyncio.StreamReader, queue: asyncio.Queue[SocketServerEvent]
):
    async for event in read_event(reader):
        print(f"RECEIVED event in TTC: {event}")
        if isinstance(event, SocketClientEvent):
            print("Received SocketClientEvent in the client, skipping for now.")
            continue
        await queue.put(event)


async def start(whisper: WhisperSTT):
    stt_event_queue: asyncio.Queue[STTEvent] = asyncio.Queue()
    tts_event_queue: asyncio.Queue[TTSEvent] = asyncio.Queue()
    agents_chunk_event_queue: queue.Queue[str | None] = queue.Queue()
    server_event_queue: asyncio.Queue[SocketServerEvent] = asyncio.Queue()
    server_writers: dict[Role, asyncio.StreamWriter] = {}
    count = 0

    user_task = asyncio.create_task(user_listener(whisper, stt_event_queue))
    process_events_task = asyncio.create_task(
        process_events(
            server_writers,
            server_event_queue,
            stt_event_queue,
            tts_event_queue,
            agents_chunk_event_queue,
        )
    )

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        if len(server_writers) == 0:
            role = Role.TEACHER
        else:
            role = Role.STUDENT
        nonlocal count
        count = count + 1
        # if count == 2:
        #     write_event(server_writers[Role.TEACHER], SocketAgentTurnEvent.create())

        server_writers[role] = writer
        server_task = asyncio.create_task(server_listener(reader, server_event_queue))

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
        type=int,
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
