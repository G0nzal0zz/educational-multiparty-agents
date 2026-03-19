import asyncio
import queue
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, ParamSpec, TypeVar

from shared_lib.events import (
    Role,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketHumanTranscription,
)
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
        if context.turn_manager.current_turn not in [Turn.TEACHER, Turn.STUDENT]:
            print(
                f"Stopping audio player. Role speaking = {context.turn_manager.current_turn}"
            )
            context.tts.audio_player_stop = True

        if isinstance(event, STTEndEvent):
            self._handle_end_event(event, context)

    def _handle_end_event(self, event: STTEndEvent, context: EventContext):
        human_event = SocketHumanTranscription.create(event.transcript)

        write_event(context.server_writers[Role.TEACHER], human_event)
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

        context.turn_manager.set_turn(Turn.STUDENT)
        print("Teacher finished speaking. Setting TURN to STUDENT")

    def _handle_student_end(self, context: EventContext) -> None:
        if context.turn_manager.current_turn != Turn.STUDENT:
            print("Student finished speaking but it was not his turn.")
            return

        context.turn_manager.set_turn(Turn.TEACHER)
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
        # If Turn is set to IDLE, is because we are waiting for the human to talk.
        # Meaning, that if the human doesn't talk, we should let the agent speak.
        if context.turn_manager.current_turn == Turn.IDLE:
            print("Waiting for the human to speak before playing agent audio")
            # Waiting at most config.USER_TURN_TIMEOUT for the user to speak.
            _ = await poll_callback(
                config.USER_TURN_TIMEOUT,
                0.1,
                self._turn_has_changed,
                context.turn_manager,
            )

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
        if event.role == Role.TEACHER:
            self._handle_teacher_end(event, context)
        elif event.role == Role.STUDENT:
            self._handle_student_end(event, context)

    def _handle_teacher_end(
        self, event: SocketAgentTextEndEvent, context: EventContext
    ) -> None:
        if context.turn_manager.current_turn != Turn.TEACHER:
            print("Teacher finished STREAMING text but it was not his turn.")
            return

        context.agents_chunk_event_queue.put(None)
        write_event(context.server_writers[Role.STUDENT], event)

    def _handle_student_end(
        self, event: SocketAgentTextEndEvent, context: EventContext
    ) -> None:
        if context.turn_manager.current_turn != Turn.STUDENT:
            print("Student finished STREAMING text but it was not his turn.")
            return

        context.agents_chunk_event_queue.put(None)
        write_event(context.server_writers[Role.TEACHER], event)
