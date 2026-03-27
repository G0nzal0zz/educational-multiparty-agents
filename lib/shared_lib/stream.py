import asyncio
import json

from shared_lib.events import SocketEvent, bytes_to_event, event_to_dict


def write_event(writer: asyncio.StreamWriter, event: SocketEvent) -> None:
    event_dict = event_to_dict(event)
    event_bytes = (json.dumps(event_dict) + "\n").encode()
    writer.write(event_bytes)


async def read_event(reader: asyncio.StreamReader):
    while True:
        line = await reader.readline()
        if not line:
            break
        yield bytes_to_event(line.rstrip(b"\n"))
