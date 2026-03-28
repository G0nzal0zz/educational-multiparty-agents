import asyncio
from asyncio import Queue

from langchain_ollama import ChatOllama

from client_lib.events import AgentChunkEvent, AgentEndEvent
from client_lib.prompts import TTS_SYSTEM_PROMPT

PHRASES_IN_CHUNK = 1


class OLlamaLLM:
    model: ChatOllama
    system_prompt: str

    def __init__(
        self, system_prompt: str, model: str = "llama3.2:3b", num_gpu: int | None = None
    ):
        self.model = ChatOllama(model=model, num_gpu=num_gpu)
        self.system_prompt = system_prompt

    def build_ollama_prompt(self, message: str):
        return [
            ("system", self.system_prompt),
            ("human", message),
        ]

    async def generate_response(self, message: str):
        prompt = self.build_ollama_prompt(message)
        stream = self.model.astream(prompt)
        response = ""
        chunk = ""
        phrases_in_chunk = 0

        async for message_chunk in stream:
            chunk = chunk + message_chunk.text
            response = response + message_chunk.text
            if any(c in message_chunk.text for c in ".!?"):
                phrases_in_chunk = phrases_in_chunk + 1

            if phrases_in_chunk >= PHRASES_IN_CHUNK:
                print(f"Sending AgentChunkEvent: {chunk}")
                yield AgentChunkEvent.create(chunk)
                chunk = ""
                phrases_in_chunk = 0

        if len(chunk) > 0:
            yield AgentChunkEvent.create(chunk)

        print(f"Sending AgentEndEvent: {response}")
        yield AgentEndEvent.create(response)
