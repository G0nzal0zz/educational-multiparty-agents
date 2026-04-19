import asyncio
import json

from client_lib.config import config
from client_lib.events import AgentChunkEvent, AgentEndEvent, AgentEvent
from client_lib.ollama_llm import OLlamaLLM
from client_lib.prompts import TTS_SYSTEM_PROMPT
from shared_lib.events import (
    Role,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketAgentTurnCancelledEvent,
    SocketAgentTurnEvent,
    SocketHumanTranscription,
    event_to_dict,
)
from shared_lib.stream import read_event, write_event
from shared_lib.utils import empty_queue

from prompt import STUDENT_SYSTEM_PROMPT
from student_state import StudentState

agent = OLlamaLLM(STUDENT_SYSTEM_PROMPT, num_gpu=0)


def _build_lesson_context(state: StudentState) -> str:
    parts = ["Lesson transcript so far:"]
    for entry in state.transcript:
        role_label = entry["role"]
        parts.append(f"{role_label}: {entry['text']}")
    return "\n".join(parts)


def _build_question_prompt(state: StudentState) -> str:
    context = _build_lesson_context(state)
    human_note = ""
    if state.human_level_hints:
        hints = " ".join(state.human_level_hints[-3:])
        human_note = (
            f"\n\nObservations about the human student's level: {hints}\n"
            "Adapt your question to suit that level."
        )
    already_asked = ""
    if state.questions_asked:
        already_asked = (
            "\n\nQuestions you have already asked (do not repeat): "
            + "; ".join(state.questions_asked[-5:])
        )
    return (
        f"{context}{human_note}{already_asked}\n\n"
        "Based on the last thing the teacher said, ask one short clarifying question "
        "that would help the human student understand better."
    )


def handle_teacher_end(
    text: str,
    output_queue: asyncio.Queue[AgentEvent],
):
    empty_queue(output_queue)

    async def add_output_to_queue():
        try:

            async for ste in agent.generate_response(text):
                await output_queue.put(ste)
                if isinstance(ste, AgentEndEvent):
                    print(f"[INFO] Finished generating question: {ste.text[:100]}...")
        except asyncio.CancelledError:
            raise

    _ = asyncio.create_task(add_output_to_queue())


async def handle_agent_turn(
    writer: asyncio.StreamWriter,
    output_queue: asyncio.Queue[AgentEvent],
) -> None:
    while True:
        agent_event = await output_queue.get()

        if isinstance(agent_event, AgentEndEvent):
            event = SocketAgentTextEndEvent.create(Role.STUDENT, agent_event.text)
            write_event(writer, event)
            break

        event = SocketAgentTextChunkEvent.create(agent_event.text, role=Role.STUDENT)
        write_event(writer, event)


async def event_loop(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    state = StudentState()
    output_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async for event in read_event(reader):
        if isinstance(event, SocketAgentTextEndEvent):
            print("[INFO] Received TEACHER intervention. Generating a question...")
            state.append_transcript(Role.TEACHER, event.text)
            handle_teacher_end(event.text, output_queue)

        elif isinstance(event, SocketHumanTranscription):
            state.append_transcript(Role.HUMAN, event.text)
            state.note_human_input(event.text)

        elif isinstance(event, SocketAgentTurnEvent):
            print("[INFO] Sending generated question to the TURN-TAKING-CONTROLLER.")
            await handle_agent_turn(writer, output_queue)

        elif isinstance(event, SocketAgentTurnCancelledEvent):
            # TODO: Handle turn cancellation (e.g., keep track of what was said in order to reexplain it if necessary)
            print("[INFO] Turn cancelled")

        else:
            print(f"[WARN] Unhandled event type {type(event)}")


async def start_client() -> None:
    reader, writer = await asyncio.open_connection(
        config.TTC_SERVER_HOST, config.TTC_SERVER_PORT
    )
    print("[INFO] STUDENT client connected to TTC")
    try:
        await event_loop(reader, writer)
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    asyncio.run(start_client())
