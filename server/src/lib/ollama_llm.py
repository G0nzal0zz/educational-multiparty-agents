from langchain_ollama import ChatOllama

from src.lib.events import AgentChunkEvent, AgentEndEvent
from src.lib.prompts import TTS_SYSTEM_PROMPT

ollama = ChatOllama(model="llama3.2:1b", temperature=0)

system_prompt = f"""
You are a helpful teacher. Your goal is to teach about a topic to some students.
Be concise and friendly.

Available topics: history, science, sports and culture.

{TTS_SYSTEM_PROMPT}
"""


def build_ollama_prompt(message: str):
    return [
        ("system", system_prompt),
        ("user", message),
    ]


class OllamaLLM:
    async def generate_response(self, user_prompt: str):
        # Stream the agent's response using LangChain's astream method.
        # stream_mode="messages" yields message chunks as they're generated.
        prompt = build_ollama_prompt(user_prompt)
        stream = ollama.astream(
            prompt,
        )
        # Iterate through the agent's streaming response. The stream yields
        # tuples of (message, metadata), but we only need the message.
        async for message in stream:
            # Emit agent chunks (AI messages)
            # Extract and yield the text content from each message chunk
            yield AgentChunkEvent.create(message.text)

        # Signal that the agent has finished responding for this turn
        yield AgentEndEvent.create()
