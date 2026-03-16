import asyncio
import threading
from dataclasses import dataclass
from queue import Queue

from shared_lib.events import (
    Role,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketHumanTranscription,
    SocketServerEvent,
)
from shared_lib.stream import write_event

from chatterbox_tts import ChatterboxTTS
from config import config
from events import STTEndEvent, STTEvent
from turn_manager import Turn, TurnManager


class STTEventHandler:
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
        # write_event(self.server_writers[Role.STUDENT], human_event)
        self.turn_manager.set_turn(Turn.TEACHER)


class AgentTextChunkHandler:
    turn_manager: TurnManager
    text_queue: Queue[str]
    chatterbox: ChatterboxTTS

    def __init__(self, turn_manager: TurnManager):
        self.turn_manager = turn_manager
        self.text_queue = Queue[str]()
        self.chatterbox = ChatterboxTTS()
        tts_thread = threading.Thread(
            target=self.chatterbox.start, args=(self.text_queue,)
        )
        tts_thread.daemon = True
        tts_thread.start()

    def handle(self, event: SocketAgentTextChunkEvent) -> None:
        if (
            event.role == Role.TEACHER
            and self.turn_manager.current_turn == Turn.TEACHER
        ):
            print("Teacher is talking")
            self.text_queue.put(event.text)
        elif (
            event.role == Role.STUDENT
            and self.turn_manager.current_turn == Turn.STUDENT
        ):
            print("Student is talking")
        else:
            print(
                f"Received SocketAgentTextChunkEvent, but audio couldn't be reproduced. Role = {event.role}, Turn = {self.turn_manager.current_turn}"
            )


class AgentTextEndHandler:
    server_writers: dict[Role, asyncio.StreamWriter]
    turn_manager: TurnManager
    stt_queue: asyncio.Queue[STTEvent]

    def __init__(
        self,
        server_writers: dict[Role, asyncio.StreamWriter],
        turn_manager: TurnManager,
        stt_queue: asyncio.Queue[STTEvent],
    ):
        self.server_writers = server_writers
        self.turn_manager = turn_manager
        self.stt_queue = stt_queue

    async def handle(self, event: SocketAgentTextEndEvent) -> None:
        if event.role == Role.TEACHER:
            await self._handle_teacher_end(event)
        elif event.role == Role.STUDENT:
            self._handle_student_end(event)

    async def _handle_teacher_end(self, event: SocketAgentTextEndEvent) -> None:
        if self.turn_manager.current_turn != Turn.TEACHER:
            print("Teacher finished speaking but it was not his turn.")
            return

        # write_event(self.server_writers[Role.STUDENT], event)
        self.turn_manager.set_turn(Turn.IDLE)

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

    def _handle_student_end(self, event: SocketAgentTextEndEvent) -> None:
        if self.turn_manager.current_turn != Turn.STUDENT:
            print("Student finished speaking but it was not his turn.")
            return

        write_event(self.server_writers[Role.TEACHER], event)
        self.turn_manager.set_turn(Turn.TEACHER)
        print("Student finished speaking. Setting TURN to TEACHER")
