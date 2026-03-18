import asyncio
import json
from collections.abc import AsyncGenerator

from langchain_core.runnables import RunnableSerializable
from shared_lib.events import (
    SocketClientEvent,
    SocketEvent,
    SocketServerEvent,
    bytes_to_event,
    event_to_dict,
)
from shared_lib.stream import read_event

from client_lib.events import ServerEvent


class ClientHandler:
    def __init__(self, pipeline: RunnableSerializable[SocketEvent, SocketServerEvent]):
        self.pipeline: RunnableSerializable[SocketEvent, SocketServerEvent] = pipeline

    async def socket_reader_to_event(
        self, reader: asyncio.StreamReader
    ) -> AsyncGenerator[SocketEvent]:
        async for event in read_event(reader):
            # if isinstance(event, SocketServerEvent):
            #     print("Received a SockerServerEvent in a server, skipping for now.")
            #     continue
            print(f"Received event: {event}")
            yield event

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        output_stream = self.pipeline.atransform(self.socket_reader_to_event(reader))

        try:
            async for event in output_stream:
                if not isinstance(event, SocketServerEvent):
                    print("ERROR: Tried to send an invalid event to the server")
                    continue
                dict = event_to_dict(event)
                json_data = json.dumps(event_to_dict(event)) + "\n"

                writer.write(json_data.encode())
                await writer.drain()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # TODO: Improve error handling
            print("client error:", e)
        finally:
            writer.close()
            await writer.wait_closed()
