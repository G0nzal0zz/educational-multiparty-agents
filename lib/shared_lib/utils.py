import asyncio

from shared_lib.events import SocketServerEvent, bytes_to_event


async def stream_reader_to_event(reader: asyncio.StreamReader):
    while True:
        data = await reader.read(4096)
        if not data:
            break
        event = bytes_to_event(data)
        yield event
