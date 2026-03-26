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
    SocketAgentTurnEvent,
    SocketHumanTranscription,
    event_to_dict,
)
from shared_lib.stream import read_event, write_event

from student_state import StudentState

TEACHER_PROMPT = """
The teacher will be teaching about the Spanish Empire.
The learning objectives for this lesson are:
1. Understand how Spain built the first global empire
2. Analyse the political, economic and religious drivers of expansion
3. Evaluate the impact on indigenous peoples and colonised societies
4. Trace the causes of Spanish imperial decline
The Syllabus and flow of the lesson is as follows:
1. Context & foundations of expansion
2. Conquest & the empire at its height
3. Administration & colonial society
4. Decline & collapse
5. Legacy & reflection
""".strip()

STUDENT_SYSTEM_PROMPT = f"""
You are a curious and engaged student attending a lesson given by a teacher. \
Another human student is also attending the lesson alongside you.

Your role is to ask clarifying questions that help the human student understand the material better. \
Your questions should reflect the human student's level of understanding. \
You will be informed when the human student speaks, and you must adapt your future questions \
to match the vocabulary, depth, and pace that suits that student. \
For context about the teacher and the lesson, {TEACHER_PROMPT}  \

When formulating questions:
- Focus on concepts that were potentially confusing or under-explained.
- Prioritise questions a curious but non-expert student would genuinely ask.
- Keep each question short, natural, and conversational.
- Ask only one question per turn.
- Do not repeat questions already asked or already answered.
- Never address the human student directly. Your questions are always directed at the teacher.

{TTS_SYSTEM_PROMPT}
""".strip()

agent = OLlamaLLM(STUDENT_SYSTEM_PROMPT)


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
    state: StudentState,
    writer: asyncio.StreamWriter,
    output_queue: asyncio.Queue[AgentEvent],
) -> None:
    async def add_output_to_queue():
        trancripts = _build_lesson_context(state)
        async for ste in agent.generate_response(
            f"Based on the teacher's last statements,{trancripts}, ask one short clarifying question that would help the human student understand better."
        ):
            await output_queue.put(ste)

    asyncio.create_task(add_output_to_queue())


async def handle_agent_turn(
    writer: asyncio.StreamWriter, output_queue: asyncio.Queue[AgentEvent]
) -> None:
    while True:
        agent_event = await output_queue.get()
        await asyncio.sleep(0.1)  # small delay to avoid flooding (optional)
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
        print(f"Student received event: {event}")

        if isinstance(event, SocketAgentTextEndEvent):
            state.append_transcript(Role.TEACHER, event.text)
            handle_teacher_end(state, writer, output_queue)

        elif isinstance(event, SocketHumanTranscription):
            state.append_transcript(Role.HUMAN, event.text)
            state.note_human_input(event.text)

        elif isinstance(event, SocketAgentTurnEvent):
            await handle_agent_turn(writer, output_queue)

        else:
            print(f"Student: unhandled event type {type(event)}")


async def start_client() -> None:
    reader, writer = await asyncio.open_connection(
        config.TTC_SERVER_HOST, config.TTC_SERVER_PORT
    )
    print("Student client connected to TTC")
    try:
        await event_loop(reader, writer)
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    asyncio.run(start_client())
