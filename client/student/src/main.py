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

from student_state import StudentState

STUDENT_SYSTEM_PROMPT = f"""
You are a STUDENT, not a teacher. Your ONLY job is to ask questions. \
You must NEVER explain, teach, or provide answers. You only ask clarifying questions directed at the teacher. \

When the teacher finishes speaking, IMMEDIATELY ask one short clarifying question. \
Do not provide any explanation, summary, or teaching content. Only ask questions. \

Examples of correct responses:
- "Can you explain that in simpler terms?"
- "What do you mean by that?"
- "Why did that happen?"
- "How does that work?"

Examples of INCORRECT responses (do not do these):
- "The Spanish Empire was formed because..."
- "To understand this, we need to consider..."
- Any explanation, teaching, or informational content.

## Message Types You Will Handle

You will receive the following types of messages:
1. **Teacher messages**: Transcript excerpts of what the teacher has said in the lesson.
2. **Human student messages**: Transcriptions of what the human student has spoken aloud.

## Your Output Format

You must output exactly one clarifying question directed at the teacher. Your output will be:
- Sent as streaming text chunks followed by an end-of-turn signal.
- Converted to speech using text-to-speech (TTS).
- Used by the turn-taking controller to manage conversation flow.

## Question Guidelines

- Focus on concepts that were potentially confusing or under-explained.
- Prioritise questions a curious but non-expert student would genuinely ask.
- Keep your question short, natural, and conversational (suitable for TTS).
- Ask only one question per turn.
- Do not repeat questions already asked or already answered.
- Never address the human student directly. Your questions are always directed at the teacher.
- End your question with a question mark.

## Response Format for TTS

Your response will be:
- Spoken aloud using text-to-speech (TTS)
- Displayed as plain text

Output only:
- One question, 10-15 words maximum
- Natural spoken language (no markdown, no emojis)
- A single question mark at the end

{TTS_SYSTEM_PROMPT}
""".strip()

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


def _empty_queue(queue: asyncio.Queue):
    while not queue.empty():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            break


def handle_teacher_end(
    state: StudentState,
    output_queue: asyncio.Queue[AgentEvent],
):
    _empty_queue(output_queue)

    async def add_output_to_queue():
        try:
            trancripts = _build_lesson_context(state)

            async for ste in agent.generate_response(trancripts):
                await output_queue.put(ste)
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
        print(f"Student received event: {event}")

        if isinstance(event, SocketAgentTextEndEvent):
            state.append_transcript(Role.TEACHER, event.text)
            handle_teacher_end(state, output_queue)

        elif isinstance(event, SocketHumanTranscription):
            state.append_transcript(Role.HUMAN, event.text)
            state.note_human_input(event.text)

        elif isinstance(event, SocketAgentTurnEvent):
            await handle_agent_turn(writer, output_queue)

        elif isinstance(event, SocketAgentTurnCancelledEvent):
            # TODO: Handle turn cancellation (e.g., keep track of what was said in order to reexplain it if necessary)
            print("INFO: Turn cancelled")

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
