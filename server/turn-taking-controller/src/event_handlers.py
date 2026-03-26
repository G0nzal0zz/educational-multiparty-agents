import asyncio
import queue
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, ParamSpec, TypeVar

from shared_lib.events import (Role, SocketAgentTextChunkEvent,
                               SocketAgentTextEndEvent, SocketAgentTurnEvent,
                               SocketHumanTranscription)
from shared_lib.stream import write_event

from chatterbox_tts import ChatterboxTTS
from config import config
from events import STTEndEvent, STTEvent, TTSEndEvent
from turn_manager import Turn, TurnManager

P = ParamSpec("P")
R = TypeVar("R")


async def poll_callback(
    timeout: float,
    sleep: float,
    callback: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> bool:
    loop = asyncio.get_running_loop()
    start = loop.time()

    while loop.time() - start < timeout:
        if callback(*args, **kwargs):
            return True
        await asyncio.sleep(sleep)

    return False


@dataclass
class EventContext:
    turn_manager: TurnManager
    server_writers: dict[Role, asyncio.StreamWriter]
    tts: ChatterboxTTS

    stt_event_queue: asyncio.Queue[STTEvent]
    agents_chunk_event_queue: queue.Queue[str | None]


class STTEventHandler:
    def handle(self, event: STTEvent, context: EventContext) -> None:
        print(f"STT: Current role speaking = f{context.turn_manager.current_turn}")
        if context.turn_manager.current_turn in [Turn.TEACHER, Turn.STUDENT]:
            print(
                f"Stopping audio player. Role speaking = {context.turn_manager.current_turn}"
            )
            context.tts.audio_player_stop = True

        if isinstance(event, STTEndEvent):
            self._handle_end_event(event, context)

    def _handle_end_event(self, event: STTEndEvent, context: EventContext):
        human_event = SocketHumanTranscription.create(event.transcript)
        turn_event = SocketAgentTurnEvent.create()

        print(f"STTTT: Sending human event: {human_event}")
        write_event(context.server_writers[Role.TEACHER], human_event)
        asyncio.sleep(0.1)
        write_event(context.server_writers[Role.TEACHER], turn_event)

        write_event(context.server_writers[Role.STUDENT], human_event)

        context.turn_manager.set_turn(Turn.TEACHER)


class TTSEndEventHandler:
    def handle(
        self, event: TTSEndEvent, context: EventContext
    ) -> Awaitable[None] | None:
        if event.role == Role.TEACHER:
            return self._handle_teacher_end(context)
        elif event.role == Role.STUDENT:
            return self._handle_student_end(context)

    def _human_has_talked(self, stt_queue: asyncio.Queue[STTEvent]) -> bool:
        if not stt_queue.empty():
            return True
        return False

    async def _handle_teacher_end(self, context: EventContext) -> None:
        if context.turn_manager.current_turn != Turn.TEACHER:
            print("Teacher finished speaking but it was not his turn.")
            return

        context.turn_manager.set_turn(Turn.IDLE)

        if await poll_callback(
            config.USER_TURN_TIMEOUT,
            0.1,
            self._human_has_talked,
            context.stt_event_queue,
        ):
            context.turn_manager.set_turn(Turn.HUMAN)
            print("Teacher finished speaking. Setting TURN to HUMAN")

        turn_event = SocketAgentTurnEvent.create()

        context.turn_manager.set_turn(Turn.STUDENT)
        write_event(context.server_writers[Role.STUDENT], turn_event)
        print("Teacher finished speaking. Setting TURN to STUDENT")

    def _handle_student_end(self, context: EventContext) -> None:
        if context.turn_manager.current_turn != Turn.STUDENT:
            print("Student finished speaking but it was not his turn.")
            return

        turn_event = SocketAgentTurnEvent.create()

        context.turn_manager.set_turn(Turn.TEACHER)
        write_event(context.server_writers[Role.TEACHER], turn_event)
        print("Student finished speaking. Setting TURN to TEACHER")


class AgentTextChunkHandler:
    _tts_task: asyncio.Task[None] | None = None

    def _turn_has_changed(self, turn_manager: TurnManager):
        if turn_manager.current_turn != Turn.IDLE:
            return True
        return False

    async def handle(
        self, event: SocketAgentTextChunkEvent, context: EventContext
    ) -> None:
        if not context.turn_manager.is_role_turn(event.role):
            print(
                f"Received SocketAgentTextChunkEvent, but audio couldn't be reproduced. "
                f"Role = {event.role}, Turn = {context.turn_manager.current_turn}"
            )
            return

        if self._needs_tts_start():
            self._start_tts(event.role, context)

        context.agents_chunk_event_queue.put(event.text)

    def _needs_tts_start(self) -> bool:
        return (
            AgentTextChunkHandler._tts_task is None
            or AgentTextChunkHandler._tts_task.done()
        )

    def _start_tts(
        self, role: Literal[Role.STUDENT, Role.TEACHER], context: EventContext
    ) -> None:
        thread = asyncio.to_thread(
            context.tts.start,
            role,
            asyncio.get_running_loop(),
        )

        AgentTextChunkHandler._tts_task = asyncio.create_task(thread)


class AgentTextEndHandler:
    def handle(self, event: SocketAgentTextEndEvent, context: EventContext) -> None:
        print("Received AgentTextEndHandler")
        if not context.turn_manager.is_role_turn(event.role):
            print(f"{event.role.name} finished STREAMING text but it was not his turn.")
            return

        context.agents_chunk_event_queue.put(None)
        writer_role = Role.TEACHER if event.role == Role.STUDENT else Role.STUDENT
        write_event(context.server_writers[writer_role], event)
