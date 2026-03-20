import asyncio
import json

from shared_lib.events import (SocketEvent, SocketServerEvent, bytes_to_event,
                               event_to_dict)


def write_event(writer: asyncio.StreamWriter, event: SocketEvent) -> None:
    event_dict = event_to_dict(event)
    event_bytes = json.dumps(event_dict).encode()
    writer.write(event_bytes)


async def read_event(reader: asyncio.StreamReader):
    while True:
        data = await reader.read(4096)
        if not data:
            break
        event = bytes_to_event(data)
        yield event
