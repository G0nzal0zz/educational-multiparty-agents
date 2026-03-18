import asyncio
import queue

from shared_lib.events import (
    Role,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketHumanTranscription,
)
from shared_lib.stream import write_event

from chatterbox_tts import ChatterboxTTS
from config import config
from events import STTEndEvent, TTSEndEvent, TTSEvent
from turn_manager import Turn, TurnManager


class STTEndEventHandler:
    turn_manager: TurnManager
    server_writers: dict[Role, asyncio.StreamWriter]

    def __init__(
        self,
        server_writers: dict[Role, asyncio.StreamWriter],
        turn_manager: TurnManager,
    ):
        self.server_writers = server_writers
        self.turn_manager = turn_manager

    def handle(self, event: STTEndEvent) -> None:
        human_event = SocketHumanTranscription.create(event.transcript)
        write_event(self.server_writers[Role.TEACHER], human_event)
        if Role.STUDENT in self.server_writers:
            write_event(self.server_writers[Role.STUDENT], human_event)
        self.turn_manager.set_turn(Turn.TEACHER)


class TTSEndEventHandler:
    turn_manager: TurnManager
    stt_queue: asyncio.Queue[TTSEvent]

    def __init__(
        self,
        turn_manager: TurnManager,
        stt_queue: asyncio.Queue[TTSEvent],
    ):
        self.turn_manager = turn_manager
        self.stt_queue = stt_queue

    async def handle(self, event: TTSEndEvent) -> None:
        if event.role == Role.TEACHER:
            await self._handle_teacher_end()
        elif event.role == Role.STUDENT:
            self._handle_student_end()

    async def _handle_teacher_end(self) -> None:
        if self.turn_manager.current_turn != Turn.TEACHER:
            print("Teacher finished speaking but it was not his turn.")
            return

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < config.USER_TURN_TIMEOUT:
            if not self.stt_queue.empty():
                self.turn_manager.set_turn(Turn.HUMAN)
                print("Teacher finished speaking. Setting TURN to HUMAN")
                return
            await asyncio.sleep(0.1)

        # No STT event has been received, let agentic student speak
        self.turn_manager.set_turn(Turn.STUDENT)
        print("Teacher finished speaking. Setting TURN to STUDENT")

    def _handle_student_end(self) -> None:
        if self.turn_manager.current_turn != Turn.STUDENT:
            print("Student finished speaking but it was not his turn.")
            return

        self.turn_manager.set_turn(Turn.TEACHER)
        print("Student finished speaking. Setting TURN to TEACHER")


class AgentTextChunkHandler:
    turn_manager: TurnManager
    text_queue: queue.Queue[str | None]
    chatterbox: ChatterboxTTS
    tts_thread: asyncio.Task | None

    def __init__(
        self,
        turn_manager: TurnManager,
        tts_queue: asyncio.Queue[TTSEvent],
        text_queue: queue.Queue[str | None],
    ):
        self.turn_manager = turn_manager
        self.chatterbox = ChatterboxTTS(tts_queue, text_queue)
        self.tts_thread = None
        self.text_queue = text_queue

    def handle(self, event: SocketAgentTextChunkEvent) -> None:
        if self.turn_manager.is_role_turn(event.role):
            # If TTS thread isn't running intialize it
            if self.tts_thread is None or self.tts_thread.done():
                thread = asyncio.to_thread(
                    self.chatterbox.start, event.role, asyncio.get_running_loop()
                )
                self.tts_thread = asyncio.create_task(thread)

            self.text_queue.put(event.text)
        else:
            print(
                f"Received SocketAgentTextChunkEvent, but audio couldn't be reproduced. Role = {event.role}, Turn = {self.turn_manager.current_turn}"
            )


class AgentTextEndHandler:
    server_writers: dict[Role, asyncio.StreamWriter]
    turn_manager: TurnManager
    text_queue: queue.Queue[str | None]

    def __init__(
        self,
        server_writers: dict[Role, asyncio.StreamWriter],
        turn_manager: TurnManager,
        text_queue: queue.Queue[str | None],
    ):
        self.server_writers = server_writers
        self.turn_manager = turn_manager
        self.text_queue = text_queue

    def handle(self, event: SocketAgentTextEndEvent) -> None:
        if event.role == Role.TEACHER:
            self._handle_teacher_end(event)
        elif event.role == Role.STUDENT:
            self._handle_student_end(event)

    def _handle_teacher_end(self, event: SocketAgentTextEndEvent) -> None:
        if self.turn_manager.current_turn != Turn.TEACHER:
            print("Teacher finished STREAMING text but it was not his turn.")
            return

        self.text_queue.put(None)
        if Role.STUDENT in self.server_writers:
            write_event(self.server_writers[Role.STUDENT], event)

    def _handle_student_end(self, event: SocketAgentTextEndEvent) -> None:
        if self.turn_manager.current_turn != Turn.STUDENT:
            print("Student finished STREAMING text but it was not his turn.")
            return

        write_event(self.server_writers[Role.TEACHER], event)
