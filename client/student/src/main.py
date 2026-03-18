import asyncio
import json
from collections.abc import AsyncIterator

from client_lib.config import config
from client_lib.ollama_llm import OLlamaLLM
from client_lib.prompts import TTS_SYSTEM_PROMPT
from shared_lib.events import (
    Role,
    SocketAgentTextChunkEvent,
    SocketAgentTextEndEvent,
    SocketAgentTextEndEvent,
    SocketEvent,
    SocketHumanTranscription,
    SocketServerEvent,
    bytes_to_event,
    event_to_dict,
)
from shared_lib.stream import read_event

from student_state import StudentState

STUDENT_SYSTEM_PROMPT = f"""
You are a curious and engaged student attending a lesson given by a teacher. \
Another human student is also attending the lesson alongside you.

Your role is to ask clarifying questions that help the human student understand the material better. \
Your questions should reflect the human student's level of understanding. \
You will be informed when the human student speaks, and you must adapt your future questions \
to match the vocabulary, depth, and pace that suits that student.

When formulating questions:
- Focus on concepts that were potentially confusing or under-explained.
- Prioritise questions a curious but non-expert student would genuinely ask.
- Keep each question short, natural, and conversational.
- Ask only one question per turn.
- Do not repeat questions already asked or already answered.
- Never address the human student directly. Your questions are always directed at the teacher.

{TTS_SYSTEM_PROMPT}
""".strip()

PREVENTIVE_SYSTEM_PROMPT = f"""
You are a curious student attending a lesson. You are about to ask the teacher a question. \
Write a single short sentence (one line, no punctuation beyond a period) that warns the teacher \
you have a question coming, without revealing what the question is yet. \
This is purely a conversational heads-up, like "I have a question about that last point." \
Keep it natural and brief.

{TTS_SYSTEM_PROMPT}
""".strip()

agent = OLlamaLLM(STUDENT_SYSTEM_PROMPT)
preventive_agent = OLlamaLLM(PREVENTIVE_SYSTEM_PROMPT)


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
        already_asked = "\n\nQuestions you have already asked (do not repeat): " + "; ".join(
            state.questions_asked[-5:]
        )
    return (
        f"{context}{human_note}{already_asked}\n\n"
        "Based on the last thing the teacher said, ask one short clarifying question "
        "that would help the human student understand better."
    )


def _build_preventive_prompt(state: StudentState) -> str:
    context = _build_lesson_context(state)
    return (
        f"{context}\n\n"
        "You are about to ask a question. Write a single brief sentence "
        "notifying the teacher that a question is coming."
    )


async def _generate_text(llm: OLlamaLLM, prompt: str) -> str:
    collected = []
    async for event in llm.generate_response(prompt):
        if hasattr(event, "text"):
            collected.append(event.text)
    return "".join(collected).strip()


async def _stream_text_as_events(
    text: str,
    role: Role,
    writer: asyncio.StreamWriter,
) -> None:
    chunk_event = SocketAgentTextChunkEvent.create(text=text, role=role)
    json_data = json.dumps(event_to_dict(chunk_event)) + "\n"
    writer.write(json_data.encode())
    await writer.drain()

    end_event = SocketAgentTextEndEvent.create(role=role)
    json_data = json.dumps(event_to_dict(end_event)) + "\n"
    writer.write(json_data.encode())
    await writer.drain()


async def handle_teacher_end(
    state: StudentState,
    writer: asyncio.StreamWriter,
) -> None:
    preventive_prompt = _build_preventive_prompt(state)
    preventive_text = await _generate_text(preventive_agent, preventive_prompt)

    question_prompt = _build_question_prompt(state)
    question_task = asyncio.create_task(
        _generate_text(agent, question_prompt)
    )

    await _stream_text_as_events(preventive_text, Role.STUDENT, writer)

    question_text = await question_task
    state.questions_asked.append(question_text)
    await _stream_text_as_events(question_text, Role.STUDENT, writer)


async def event_loop(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    state = StudentState()

    async for event in read_event(reader):
        print(f"Student received event: {event}")

        if isinstance(event, SocketAgentTextChunkEvent) and event.role == Role.TEACHER:
            state.append_transcript("Teacher", event.text)

        elif isinstance(event, SocketAgentTextEndEvent) and event.role == Role.TEACHER:
            await handle_teacher_end(state, writer)

        elif isinstance(event, SocketHumanTranscription):
            state.append_transcript("Human student", event.text)
            state.note_human_input(event.text)

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
