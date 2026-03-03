import asyncio

from langchain_core.runnables import RunnableSerializable

from src.lib.events import TTSChunkEvent, VoiceAgentEvent


class ClientHandler:
    def __init__(self, pipeline: RunnableSerializable[bytes, VoiceAgentEvent]):
        self.pipeline: RunnableSerializable[bytes, VoiceAgentEvent] = pipeline

    async def socket_audio_stream(self, reader: asyncio.StreamReader):
        while True:
            data = await reader.read(4096)
            if not data:
                break
            yield data

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        output_stream = self.pipeline.atransform(self.socket_audio_stream(reader))

        try:
            async for event in output_stream:
                if isinstance(event, TTSChunkEvent):
                    writer.write(event.audio)
                    await writer.drain()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # TODO: Improve error handling
            print("client error:", e)
        finally:
            writer.close()
            await writer.wait_closed()
