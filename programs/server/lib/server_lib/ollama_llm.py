from langchain_ollama import ChatOllama

from server_lib.events import AgentChunkEvent, AgentEndEvent
from server_lib.prompts import TTS_SYSTEM_PROMPT


class OLlamaLLM:
    model: ChatOllama
    system_prompt: str

    def __init__(self, system_prompt: str):
        self.model = ChatOllama(model="llama3.2:1b", temperature=0)
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

        async for chunk in stream:
            yield AgentChunkEvent.create(chunk.text)

        yield AgentEndEvent.create()
