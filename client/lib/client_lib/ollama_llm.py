import asyncio
from asyncio import Queue

from langchain_ollama import ChatOllama

from client_lib.events import AgentChunkEvent, AgentEndEvent
from client_lib.prompts import TTS_SYSTEM_PROMPT

PHRASES_IN_CHUNK = 5


class OLlamaLLM:
    model: ChatOllama
    system_prompt: str

    def __init__(self, system_prompt: str):
        self.model = ChatOllama(model="steamdj/llama3.1-cpu-only", temperature=0)
        self.system_prompt = system_prompt

    def build_ollama_prompt(self, message: str):
        return [
            ("system", self.system_prompt),
            ("user", message),
        ]

    async def generate_response(self, message: str):
        # Stream the agent's response using LangChain's astream method.
        # stream_mode="messages" yields message chunks as they're generated.
        prompt = self.build_ollama_prompt(message)
        stream = self.model.astream(prompt)
        chunk = ""
        phrases_in_chunk = 0

        async for message_chunk in stream:
            chunk = chunk + " " + message_chunk.text
            if any(c in message_chunk.text for c in ".!?"):
                print(f"Phrase completed, phrases in chunk = {phrases_in_chunk}")
                phrases_in_chunk = phrases_in_chunk + 1

            if phrases_in_chunk >= PHRASES_IN_CHUNK:
                print(f"Sending AgentChunkEvent: {chunk}")
                yield AgentChunkEvent.create(chunk)
                chunk = ""
                phrases_in_chunk = 0

        if len(chunk) > 0:
            yield AgentChunkEvent.create(chunk)

        yield AgentEndEvent.create()
